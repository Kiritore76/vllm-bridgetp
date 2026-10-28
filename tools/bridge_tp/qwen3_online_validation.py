# SPDX-License-Identifier: Apache-2.0
"""Read-only finite predictor checks on live source streams; no policy trigger."""

import json
import time
from pathlib import Path

from tools.bridge_tp.prepare_4090d_qwen3 import sha
from tools.bridge_tp.pressure_cpu import load

CHECKPOINT_SHA = "f51f6a8fd2e299f2ebffae71706f7c9dcd93154eeffa18cf008bf0ee710f738f"


def background_completion(io, url, payload, timeout, directory, index):
    """Keep per-token client timing for background load, including migration gaps."""
    path = directory / f"background-{index}.tokens.jsonl"
    with path.open("x", encoding="utf-8", buffering=1) as sink:

        def token(position, token_id, unix_s):
            sink.write(
                json.dumps(dict(index=position, token_id=token_id, unix_s=unix_s))
                + "\n"
            )

        return io.post_streaming_completion(
            url,
            payload,
            timeout,
            token,
            directory / f"background-{index}-failure.json",
            retain_chunks=False,
        )


def latest_inventory(path):
    if not path.exists():
        return None
    # Read a bounded suffix, not the growing journal on every SSE polling tick.
    with path.open("rb") as source:
        source.seek(0, 2)
        end = source.tell()
        size = min(end, 65536)
        while size:
            source.seek(end - size)
            data = source.read(size)
            if size < end:
                data = data.partition(b"\n")[2]
            complete = data.splitlines(keepends=True)
            complete = [line for line in complete if line.endswith(b"\n")]
            if complete:
                return json.loads(complete[-1])
            if size == end:
                return None
            size = min(end, size * 2)
    return None


def configure_predictor(plan, checkpoint, model):
    checkpoint = Path(checkpoint).resolve()
    if sha(checkpoint) != CHECKPOINT_SHA:
        raise ValueError("requires the audited frozen Qwen3 checkpoint")
    predictor = load("distribution_predictor").DistributionPredictor(
        checkpoint,
        checkpoint_sha256=CHECKPOINT_SHA,
        model_config_sha256=sha(Path(model) / "config.json"),
        device="cpu",
    )
    if (
        predictor.overflow_category
        or predictor.category_upper_edges.numel() != 62
        or predictor.category_upper_edges[-1].item() != 8192
        or predictor.feature_mean.numel() != 4096
        or predictor.network_depth != 4
    ):
        raise ValueError("Qwen3 finite62 predictor geometry differs")
    plan["source_env"].update(
        BRIDGETP_PREDICTOR_LIVE_CHECKPOINT=str(checkpoint),
        BRIDGETP_PREDICTOR_LIVE_SHA256=CHECKPOINT_SHA,
        BRIDGETP_PREDICTOR_LIVE_EVENTS=str(plan["run_dir"] / "predictor_events.jsonl"),
    )


def probe_requests(tokenizer):
    jobs = []
    for name, text, cap in (
        ("short", "Compute 17 * 23. Give the result and a short explanation.", 512),
        (
            "code",
            (
                "Write a Python asyncio bounded worker queue with cancellation, "
                "timeouts, shutdown, examples and tests. Explain race conditions."
            ),
            2048,
        ),
        (
            "long",
            (
                "Write a detailed design for an LLM serving scheduler. Discuss "
                "admission, KV memory, migration, ownership, failures, fairness, "
                "overload, telemetry, testing and experiments. Include pseudocode."
            ),
            6144,
        ),
    ):
        prompt = tokenizer.apply_chat_template(
            [dict(role="user", content=text)],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        if len(prompt) + cap > 8192:
            raise ValueError("probe exceeds TP1 context")
        payload = load("sampling_contract").freeze_strict_greedy_sampling(
            dict(
                model="bridgetp-model",
                prompt=prompt,
                max_tokens=cap,
                ignore_eos=False,
                request_id="qwen3-online-" + name,
                stream=True,
                return_token_ids=True,
            )
        )
        jobs.append(dict(name=name, payload=payload))
    return jobs


def natural_probe(url, directory, model, io):
    from transformers import AutoTokenizer

    jobs = probe_requests(AutoTokenizer.from_pretrained(model, local_files_only=True))
    reader = load("manager_m5").PredictorEventReader(
        directory / "predictor_events.jsonl", CHECKPOINT_SHA
    )
    bounded = load("bounded_length")
    records = []
    responses = []
    with (directory / "online_advisories.jsonl").open("x", encoding="utf-8") as sink:
        for job in jobs:
            payload = job["payload"]
            available_before = len(records)
            print(f"[在线预测器] {job['name']} 开始", flush=True)

            def token(index, token_id, unix_s, payload=payload, job=job):
                if index % 20:
                    return
                inv = latest_inventory(directory / "source_inventory.jsonl")
                if inv is None:
                    return
                for request in inv["requests"]:
                    request_id = request["request_id"]
                    if not request_id.startswith("cmpl-" + payload["request_id"] + "-"):
                        continue
                    row = reader.advisory(
                        request_id,
                        index + 1,
                        max(0, 8192 - len(payload["prompt"]) - index - 1),
                        max_output_tokens=payload["max_tokens"],
                        ignore_eos=False,
                    )
                    row.update(probe=job["name"], observed_unix_s=time.time())
                    if row["status"] == "AVAILABLE":
                        distribution = bounded.from_advisory(row, payload["max_tokens"])
                        row["remaining_mean"] = distribution.mean
                        row["remaining_p50_aux"] = distribution.quantile(0.5)
                        row["point_method"] = (
                            "weighted integer-bin means; survival/lag/stop aligned"
                        )
                        records.append(row)
                    sink.write(json.dumps(row) + "\n")
                    sink.flush()

            response = io.post_streaming_completion(
                url,
                payload,
                900,
                token,
                directory / (job["name"] + "_failure.json"),
                retain_chunks=False,
            )
            response.update(
                probe=job["name"],
                max_tokens=payload["max_tokens"],
                prompt_tokens=len(payload["prompt"]),
            )
            responses.append(response)
            if len(response["token_ids"]) >= 40 and len(records) == available_before:
                raise RuntimeError(
                    "no fresh finite62 advisory for a sustained natural probe"
                )
    if not records:
        raise RuntimeError("no live finite62 predictions consumed")
    receipt = dict(
        status="COMPLETED",
        checkpoint_sha256=CHECKPOINT_SHA,
        fresh_advisories=len(records),
        responses=responses,
        automatic_policy=False,
        accuracy_evaluation=False,
        mean_method="weighted bucket midpoint with survival/lag/stop",
    )
    (directory / "online_predictor_receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    return receipt
