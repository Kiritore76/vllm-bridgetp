"""Run a bounded TP1 live probe; its diagnostic data is not benefit training."""

import argparse
import hashlib
import json
import os
from pathlib import Path

from run_predictor_capture import load_requests
from predictor_progress import RequestProgress, emit


def finish_probe(llm):
    """Drain using a named RPC, then shut down even if the diagnostic fails."""
    try:
        drained = llm.collective_rpc(
            "bridge_tp_drain_predictor_diagnostics", timeout=60
        )
        print(json.dumps({"drain": drained}), flush=True)
    finally:
        llm.llm_engine.engine_core.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() != args.expected_sha256:
        parser.error("checkpoint SHA differs")
    if args.out_dir.exists():
        parser.error("probe output must be new")
    requests = load_requests(args.input, None)
    if len(requests) != 2:
        parser.error("this pilot requires exactly two paired requests")
    args.out_dir.mkdir(parents=True)
    for key in list(os.environ):
        if key.startswith("BRIDGETP_PREDICTOR_"):
            del os.environ[key]
    os.environ.update(
        {
            "BRIDGETP_PREDICTOR_LIVE_CHECKPOINT": str(args.checkpoint.resolve()),
            "BRIDGETP_PREDICTOR_LIVE_SHA256": args.expected_sha256,
            "BRIDGETP_PREDICTOR_LIVE_EVENTS": str(args.out_dir / "events.jsonl"),
            "BRIDGETP_PREDICTOR_DIAGNOSTIC_DIR": str(args.out_dir / "probes"),
            "BRIDGETP_PREDICTOR_DIAGNOSTIC_LIMIT": "128",
        }
    )
    from vllm import LLM, SamplingParams

    emit("正在加载在线特征对照模型")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=1,
        async_scheduling=False,
        enforce_eager=True,
        max_model_len=16384,
        max_num_seqs=1,
        gpu_memory_utilization=0.85,
        enable_prefix_caching=False,
    )
    tokenizer = llm.get_tokenizer()
    labels = []
    for number, row in enumerate(requests, 1):
        prompt = tokenizer.apply_chat_template(
            row["messages"], tokenize=False, add_generation_prompt=True
        )
        with RequestProgress(number, len(requests), label="在线特征对照") as progress:
            result = llm.generate(
                [prompt],
                SamplingParams(max_tokens=384, temperature=0, ignore_eos=False),
                use_tqdm=False,
            )[0]
            progress.finish(
                len(result.outputs[0].token_ids), result.outputs[0].finish_reason
            )
        completion = result.outputs[0]
        labels.append(
            {
                "input_id": row["id"],
                "request_id": result.request_id,
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "output_token_ids": list(completion.token_ids),
                "output_tokens": len(completion.token_ids),
                "finish_reason": completion.finish_reason,
            }
        )
    (args.out_dir / "labels.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in labels), encoding="utf-8"
    )
    finish_probe(llm)


if __name__ == "__main__":
    main()
