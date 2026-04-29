"""
Alignment test: HF Transformers vs vLLM-Omni for Qwen3-Omni Thinker.

Verifies that vLLM-Omni produces the same logprobs as HF transformers
for the same model weights and input tokens.

Usage (run on 4-GPU machine):
  # Phase 1: HF forward pass
  python tests/test_qwen3_omni_alignment.py --mode hf --model_path /path/to/Qwen3-Omni

  # Phase 2: vLLM-Omni inference
  python tests/test_qwen3_omni_alignment.py --mode vllm --model_path /path/to/Qwen3-Omni --tp_size 4

  # Phase 3: Compare
  python tests/test_qwen3_omni_alignment.py --mode compare
"""

import argparse
import gc
import json
from pathlib import Path

import torch

SAVE_DIR = Path("/tmp/alignment_test")
HF_OUTPUT = SAVE_DIR / "hf_logprobs.pt"
VLLM_OUTPUT = SAVE_DIR / "vllm_logprobs.pt"
META_FILE = SAVE_DIR / "meta.json"

TEST_PROMPTS = [
    "The capital of France is",
    "1 + 1 =",
    "Explain quantum computing in one sentence:",
]


def run_hf(model_path: str, dtype: str = "bfloat16"):
    """Phase 1: HF Transformers forward pass, save logprobs."""
    from transformers import AutoConfig, AutoTokenizer
    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
        Qwen3OmniMoeForConditionalGeneration,
    )

    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    torch_dtype = getattr(torch, dtype)

    print(f"[HF] Loading tokenizer from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Qwen3-Omni: top-level model is Qwen3OmniMoeForConditionalGeneration
    # which contains thinker + talker + code2wav.
    # We load the full model then strip talker/code2wav to save memory,
    # just like veRL does.
    print(f"[HF] Loading model from {model_path}")
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    # Disable audio output to avoid loading talker/code2wav if possible
    if hasattr(config, "enable_audio_output"):
        config.enable_audio_output = False
        print("[HF] Set enable_audio_output=False to skip talker/code2wav")

    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    # Access the thinker sub-model for forward pass
    if hasattr(model, "thinker"):
        thinker = model.thinker
        print(f"[HF] Using model.thinker: {type(thinker).__name__}")
    else:
        thinker = model
        print(f"[HF] Model has no .thinker, using directly: {type(model).__name__}")

    all_results = {}

    for i, prompt in enumerate(TEST_PROMPTS):
        print(f"\n[HF] Prompt {i}: {prompt!r}")
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(thinker.device if hasattr(thinker, "device") else "cuda")
        attention_mask = inputs["attention_mask"].to(input_ids.device)

        with torch.no_grad():
            outputs = thinker(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits  # [1, seq_len, vocab_size]

        # Compute log_softmax
        logprobs = torch.log_softmax(logits.float(), dim=-1)  # float32 for precision

        # Save per-position top-k logprobs (k=20) and the full last-position logprobs
        top_k = 20
        top_values, top_indices = logprobs[0].topk(top_k, dim=-1)  # [seq_len, top_k]

        result = {
            "input_ids": input_ids[0].cpu(),
            "top_logprob_values": top_values.cpu(),  # [seq_len, top_k]
            "top_logprob_indices": top_indices.cpu(),  # [seq_len, top_k]
            "last_pos_logprobs": logprobs[0, -1].cpu(),  # [vocab_size] - full logprobs at last position
        }
        all_results[i] = result

        # Print top-5 next token predictions
        last_top5_indices = top_indices[-1, :5]
        last_top5_values = top_values[-1, :5]
        decoded = [tokenizer.decode([idx]) for idx in last_top5_indices]
        print(f"  Top-5 next tokens: {list(zip(decoded, last_top5_values.tolist()))}")

    torch.save(all_results, HF_OUTPUT)
    meta = {
        "prompts": TEST_PROMPTS,
        "dtype": dtype,
        "model_path": model_path,
        "input_ids": {i: all_results[i]["input_ids"].tolist() for i in all_results},
    }
    with open(META_FILE, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n[HF] Saved logprobs to {HF_OUTPUT}")
    print(f"[HF] Saved metadata to {META_FILE}")

    # Cleanup
    del model, thinker
    gc.collect()
    torch.cuda.empty_cache()


def run_vllm(model_path: str, dtype: str = "bfloat16", tp_size: int = 8):
    """Phase 2: vLLM-Omni inference, save logprobs."""
    from vllm import SamplingParams
    from vllm_omni.entrypoints import Omni

    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    # Load metadata from HF phase to use exact same input_ids
    with open(META_FILE) as f:
        meta = json.load(f)

    print(f"[vLLM] Initializing Omni engine: model={model_path}, tp={tp_size}")

    stage_configs_path = str(
        Path(__file__).parent / "vllm-omni/examples/online_serving/qwen3_omni/qwen3_omni_moe_thinking.yaml"
    )
    engine = Omni(
        model=model_path,
        dtype=dtype,
        tensor_parallel_size=tp_size,
        max_model_len=2048,
        trust_remote_code=True,
        enforce_eager=True,
        stage_configs_path=stage_configs_path,
    )

    all_results = {}

    for i, prompt in enumerate(TEST_PROMPTS):
        input_ids = meta["input_ids"][str(i)]
        seq_len = len(input_ids)
        print(f"\n[vLLM] Prompt {i}: {prompt!r} (tokens: {seq_len})")

        # Generate 1 token with prompt_logprobs to get logprobs at every position
        sampling_params = SamplingParams(
            max_tokens=1,
            temperature=0.0,
            logprobs=20,  # top-20 logprobs for generated token
            prompt_logprobs=20,  # top-20 logprobs for each prompt token
        )

        outputs = engine.generate(
            prompts={"prompt_token_ids": input_ids},
            sampling_params_list=[sampling_params],
        )

        output = outputs[0]
        req_output = output.request_output

        # Extract prompt logprobs (positions 1..seq_len-1, predicting each token)
        # prompt_logprobs[0] is None (no prediction for first token)
        prompt_lps = req_output.prompt_logprobs  # list of dict or None
        # Extract generated token logprobs
        gen_lps = req_output.outputs[0].logprobs  # list of dict

        # Build comparable structure: for each position, store top-k logprobs
        # Position i predicts token at position i+1
        top_k = 20
        top_values = torch.full((seq_len, top_k), float("-inf"))
        top_indices = torch.full((seq_len, top_k), -1, dtype=torch.long)

        # Prompt logprobs: positions 0..(seq_len-2) predict tokens 1..(seq_len-1)
        if prompt_lps is not None:
            for pos in range(1, seq_len):
                if prompt_lps[pos] is not None:
                    sorted_lps = sorted(prompt_lps[pos].items(), key=lambda x: x[1].logprob, reverse=True)[:top_k]
                    for k, (token_id, logprob_obj) in enumerate(sorted_lps):
                        top_values[pos - 1, k] = logprob_obj.logprob
                        top_indices[pos - 1, k] = token_id

        # Generated token logprobs: position seq_len-1 predicts next token
        if gen_lps and gen_lps[0] is not None:
            sorted_lps = sorted(gen_lps[0].items(), key=lambda x: x[1].logprob, reverse=True)[:top_k]
            for k, (token_id, logprob_obj) in enumerate(sorted_lps):
                top_values[seq_len - 1, k] = logprob_obj.logprob
                top_indices[seq_len - 1, k] = token_id

        result = {
            "input_ids": torch.tensor(input_ids),
            "top_logprob_values": top_values,
            "top_logprob_indices": top_indices,
        }
        all_results[i] = result

        # Print top-5 next token predictions
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        last_top5_indices = top_indices[-1, :5]
        last_top5_values = top_values[-1, :5]
        decoded = [tokenizer.decode([idx]) for idx in last_top5_indices if idx >= 0]
        print(f"  Top-5 next tokens: {list(zip(decoded, last_top5_values.tolist()))}")

    torch.save(all_results, VLLM_OUTPUT)
    print(f"\n[vLLM] Saved logprobs to {VLLM_OUTPUT}")


def run_compare(atol: float = 1e-2, rtol: float = 1e-2):
    """Phase 3: Compare HF and vLLM logprobs."""
    print("=" * 60)
    print("Alignment Comparison: HF vs vLLM-Omni")
    print("=" * 60)

    with open(META_FILE) as f:
        meta = json.load(f)

    hf_results = torch.load(HF_OUTPUT, weights_only=True)
    vllm_results = torch.load(VLLM_OUTPUT, weights_only=True)

    all_passed = True

    for i, prompt in enumerate(meta["prompts"]):
        print(f"\n--- Prompt {i}: {prompt!r} ---")
        hf = hf_results[i]
        vllm = vllm_results[i]

        # Verify same input
        assert torch.equal(hf["input_ids"], vllm["input_ids"]), "Input IDs mismatch!"

        hf_vals = hf["top_logprob_values"]     # [seq_len, top_k]
        hf_idxs = hf["top_logprob_indices"]     # [seq_len, top_k]
        vllm_vals = vllm["top_logprob_values"]  # [seq_len, top_k]
        vllm_idxs = vllm["top_logprob_indices"]  # [seq_len, top_k]

        seq_len = hf_vals.shape[0]

        # Compare position by position
        max_diff = 0.0
        mismatches = 0
        top1_agree = 0
        worst = None  # (pos, token_id, hf_logprob, vllm_logprob, hf_rank, vllm_rank)

        for pos in range(seq_len):
            # Check if top-1 token matches
            if hf_idxs[pos, 0] == vllm_idxs[pos, 0]:
                top1_agree += 1

            # For positions where vllm has valid logprobs
            if vllm_idxs[pos, 0] < 0:
                continue

            # Compare logprob values for tokens that appear in both top-k
            hf_token_set = set(hf_idxs[pos].tolist())
            vllm_token_set = set(vllm_idxs[pos].tolist())
            common = hf_token_set & vllm_token_set - {-1}

            for token_id in common:
                hf_idx = (hf_idxs[pos] == token_id).nonzero(as_tuple=True)[0]
                vllm_idx = (vllm_idxs[pos] == token_id).nonzero(as_tuple=True)[0]
                if len(hf_idx) > 0 and len(vllm_idx) > 0:
                    hf_lp = hf_vals[pos, hf_idx[0]].item()
                    vllm_lp = vllm_vals[pos, vllm_idx[0]].item()
                    diff = abs(hf_lp - vllm_lp)
                    if diff > max_diff:
                        max_diff = diff
                        worst = (pos, token_id, hf_lp, vllm_lp, hf_idx[0].item(), vllm_idx[0].item())
                    if diff > atol:
                        mismatches += 1

        valid_positions = sum(1 for pos in range(seq_len) if vllm_idxs[pos, 0] >= 0)
        print(f"  Positions: {seq_len}, Valid: {valid_positions}")
        print(f"  Top-1 agreement: {top1_agree}/{valid_positions}")
        print(f"  Max logprob diff: {max_diff:.6f}")
        print(f"  Mismatches (>{atol}): {mismatches}")
        if worst is not None:
            pos, token_id, hf_lp, vllm_lp, hf_rank, vllm_rank = worst
            print(f"  Worst mismatch: pos={pos}, token={token_id}, "
                  f"hf={hf_lp:.4f} (rank {hf_rank}), vllm={vllm_lp:.4f} (rank {vllm_rank})")

        passed = max_diff < atol and top1_agree == valid_positions
        status = "PASS" if passed else "FAIL"
        print(f"  Status: {status}")
        if not passed:
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("RESULT: ALL PASSED - HF and vLLM-Omni are aligned!")
    else:
        print("RESULT: SOME FAILED - check logprob differences above.")
        print(f"  Note: small diffs (<{atol}) in bf16 are normal.")
        print("  Top-1 disagreement may indicate a weight loading issue.")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Qwen3-Omni HF vs vLLM-Omni alignment test")
    parser.add_argument("--mode", choices=["hf", "vllm", "compare"], required=True)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--tp_size", type=int, default=8)
    parser.add_argument("--atol", type=float, default=0.5)
    args = parser.parse_args()

    if args.mode == "hf":
        assert args.model_path, "--model_path required for hf mode"
        run_hf(args.model_path, args.dtype)
    elif args.mode == "vllm":
        assert args.model_path, "--model_path required for vllm mode"
        run_vllm(args.model_path, args.dtype, args.tp_size)
    elif args.mode == "compare":
        run_compare(args.atol)


if __name__ == "__main__":
    main()