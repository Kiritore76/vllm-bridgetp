"""Collect twelve independent natural-EOS tasks to test long-output recipes."""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

from predictor_progress import emit

MAX_MODEL_LEN = 24576
MAX_OUTPUT_TOKENS = 18432
RECIPE_VERSION = "tail12-natural-v1-20261011"

# Each task has a distinct subject; language partners are not translations.
TASKS = [
    (
        "en",
        "8193-12288",
        "tutorial",
        "Design a small relational database engine",
        (
            "page format; buffer pool; write-ahead log; crash recovery; row layout; "
            "heap files; B-tree leaves; B-tree splits; query parsing; query planning; "
            "join execution; transactional integration"
        ),
    ),
    (
        "zh",
        "8193-12288",
        "tutorial",
        "构建一个可测试的离散事件模拟器",
        (
            "事件模型；优先队列；模拟时钟；服务站；排队规则；资源竞争；随机到达；"
            "故障与修复；数据统计；实验重复；结果可视化；系统集成"
        ),
    ),
    (
        "en",
        "8193-12288",
        "worked_cases",
        "A workbook on numerical optimization",
        (
            "quadratic minimization; line search; Newton iteration; ill conditioning; "
            "conjugate gradient; constrained projection; penalty methods; logistic "
            "regression; least squares; robust fitting; trust regions; "
            "convergence testing"
        ),
    ),
    (
        "zh",
        "8193-12288",
        "worked_cases",
        "图算法完整案例教材",
        (
            "图的存储；广度优先搜索；深度优先搜索；拓扑排序；强连通分量；最短路径；"
            "最小生成树；二分图匹配；网络最大流；割集；动态规划路径；综合调度案例"
        ),
    ),
    (
        "en",
        "8193-12288",
        "implementation",
        "Implement an incremental static-site builder",
        (
            "project layout; content parser; metadata validation; template rendering; "
            "dependency graph; incremental invalidation; asset processing; "
            "URL handling; "
            "search index; development watcher; error reporting; end-to-end tests"
        ),
    ),
    (
        "zh",
        "8193-12288",
        "implementation",
        "实现一个本地任务队列与工作进程系统",
        (
            "项目布局；任务协议；持久化存储；入队；调度；工作进程；重试；幂等；"
            "超时恢复；可观测性；管理接口；集成测试"
        ),
    ),
    (
        "en",
        "12289-16384",
        "tutorial",
        "A complete guide to building a toy compiler",
        (
            "language specification; lexer; source locations; parser; syntax tree; "
            "name resolution; type checking; intermediate representation; "
            "basic blocks; "
            "control-flow graph; data-flow analysis; constant propagation; dead-code "
            "elimination; register allocation; code generation; integration tests"
        ),
    ),
    (
        "zh",
        "12289-16384",
        "tutorial",
        "从零构建可验证的数字通信仿真链路",
        (
            "比特源；帧结构；校验码；编码；交织；符号映射；脉冲成形；信道；"
            "噪声；接收滤波；定时同步；载波同步；判决；解交织；译码；误码实验"
        ),
    ),
    (
        "en",
        "12289-16384",
        "worked_cases",
        "An applied probability workbook",
        (
            "finite sample spaces; conditional probability; Bayes updates; discrete "
            "variables; continuous variables; expectation; variance; transforms; "
            "joint distributions; sums of variables; Poisson processes; Markov chains; "
            "stationary distributions; first-passage times; Monte Carlo; "
            "confidence intervals"
        ),
    ),
    (
        "zh",
        "12289-16384",
        "worked_cases",
        "线性代数数值计算案例工作手册",
        (
            "向量空间；基变换；投影；正交化；LU分解；QR分解；特征值；奇异值；"
            "最小二乘；病态矩阵；矩阵范数；条件数；低秩近似；迭代解法；稀疏矩阵；综合实验"
        ),
    ),
    (
        "en",
        "12289-16384",
        "implementation",
        "Implement a reproducible data-processing toolkit",
        (
            "repository structure; record schema; CSV ingestion; JSON ingestion; "
            "validation; type conversion; filtering; aggregation; joins; external "
            "sorting; partitioning; checkpoints; command-line interface; "
            "configuration; "
            "error recovery; comprehensive tests"
        ),
    ),
    (
        "zh",
        "12289-16384",
        "implementation",
        "实现一个完整的日志检索与聚合工具",
        (
            "目录结构；记录协议；文本解析；JSON解析；时间标准化；字段校验；过滤；"
            "索引；检索；分组统计；流式聚合；磁盘分片；检查点；命令行；错误恢复；完整测试"
        ),
    ),
]


def recipe():
    """Return fixed tasks; target ranges express intentions, never labels."""
    rows = []
    for number, (lang, target, kind, title, components) in enumerate(TASKS, 1):
        sections = 12 if target == "8193-12288" else 16
        if lang == "en":
            instruction = (
                f"Produce the complete deliverable: {title}.\n"
                f"Write {sections} substantial numbered sections in this order: "
                f"{components}.\n"
                "For EACH section, provide a self-contained explanation of the "
                "design or mathematical principle, a fully worked concrete example "
                "with intermediate steps, an executable Python implementation "
                "where applicable, tests with expected results, and an explanation "
                "of limitations and common mistakes. Aim for approximately 550-700 "
                "words per section, excluding code. Integrate the sections into "
                "one coherent deliverable and include the complete implementations "
                "rather than stubs. Write the actual content immediately, rather "
                "than a proposed outline or a promise to continue in another turn. "
                "Avoid repeated boilerplate; each section must add new substance. "
                "Finish naturally when the deliverable is complete."
            )
        else:
            instruction = (
                f"请一次性完成以下完整交付物：{title}。\n"
                f"依次写出{sections}个内容充分的编号章节：{components}。\n"
                "每章都要包含：原理或设计的完整解释、带中间过程的具体案例、适用时"
                "可直接执行的Python实现、测试及预期结果、局限和常见错误的分析。"
                "每章正文目标为1200至1600个汉字，代码另计。各章要连成完整的教材"
                "或项目，实现要给出完整函数，不能用省略号或待实现占位。"
                "直接输出实际内容，不要只给目录或承诺下一轮继续；不要重复套话，"
                "每章都应增加新的实质内容。完成交付物后自然结束。"
            )
        identity = f"predictor-{RECIPE_VERSION}-task{number:02d}"
        rows.append(
            {
                "id": identity,
                "source_tree_id": identity,
                "source": "constructed_long_output_recipe_pilot",
                "split": "pilot",
                "collection_role": "recipe_development_not_formal_holdout",
                "lang": lang,
                "task_kind": kind,
                "target_output_range": target,
                "title": title,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a careful technical writer.",
                    },
                    {"role": "user", "content": instruction},
                ],
            }
        )
    return rows


def serialized(rows):
    """Serialize the preregistered recipe identically on every machine."""
    return "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    ).encode("utf-8")


def prepare(out_dir, tokenizer, config):
    """Check every actual prompt budget before launching the capture engine."""
    if config.get("max_position_embeddings", 0) < MAX_MODEL_LEN:
        raise ValueError("model configuration cannot support pilot context length")
    if out_dir.exists() and any(out_dir.iterdir()):
        raise ValueError("pilot output must be new or empty; old data are preserved")
    rows = recipe()
    budgets = []
    for row in rows:
        prompt_ids = tokenizer.apply_chat_template(
            row["messages"], tokenize=True, add_generation_prompt=True
        )
        count = len(prompt_ids)
        # Capture's string encode may include BOS; leave a small explicit margin.
        if count + MAX_OUTPUT_TOKENS + 16 > MAX_MODEL_LEN:
            raise ValueError(f"prompt/output budget exceeded for {row['id']}")
        budgets.append({"id": row["id"], "prompt_tokens": count})
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = serialized(rows)
    input_path = out_dir / "input_requests.jsonl"
    input_path.write_bytes(payload)
    manifest = {
        "recipe_version": RECIPE_VERSION,
        "requests": len(rows),
        "input_sha256": hashlib.sha256(payload).hexdigest(),
        "max_model_len": MAX_MODEL_LEN,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "ignore_eos": False,
        "min_tokens": 0,
        "training_started": False,
        "split": "pilot",
        "formal_holdout_eligible": False,
        "feature_layer": "decoder:31",
        "capture_interval": 20,
        "prompt_budgets": budgets,
        "targets_are_not_labels": True,
    }
    (out_dir / "input_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def length_band(tokens):
    """Classify observed output, without converting censored lengths to EOS."""
    for edge, name in [
        (4096, "<=4096"),
        (8192, "4097-8192"),
        (12288, "8193-12288"),
        (16384, "12289-16384"),
    ]:
        if tokens <= edge:
            return name
    return ">16384"


def summarize(out_dir, tokenizer, *, capture_returncode):
    """Preserve partial responses and report natural and censored counts apart."""
    inputs = {row["id"]: row for row in recipe()}
    label_path = out_dir / "capture/labels.jsonl"
    labels = (
        [
            json.loads(line)
            for line in label_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if label_path.exists()
        else []
    )
    seen = set()
    natural, censored, target_hits = Counter(), Counter(), Counter()
    responses = []
    for row in labels:
        identity = row["input_id"]
        if identity not in inputs or identity in seen:
            raise ValueError("unknown or repeated response identity")
        seen.add(identity)
        ids = row.get("output_token_ids")
        if ids is None or len(ids) != row["output_tokens"]:
            raise ValueError("response lacks complete saved token IDs")
        is_natural = row["finish_reason"] == "stop"
        if row["natural_finish"] != is_natural:
            raise ValueError("natural-EOS label disagrees with termination")
        band = length_band(row["output_tokens"])
        (natural if is_natural else censored)[band] += 1
        intended = inputs[identity]["target_output_range"]
        if is_natural and intended == band:
            target_hits[intended] += 1
        responses.append(
            {
                **row,
                "target_output_range": intended,
                "observed_output_band": band,
                "output_text": tokenizer.decode(ids, skip_special_tokens=True),
            }
        )
    (out_dir / "responses.jsonl").write_bytes(serialized(responses))
    complete = (
        capture_returncode == 0
        and len(seen) == len(inputs)
        and (out_dir / "capture/summary.json").is_file()
        and (out_dir / "capture/sample_index.jsonl").is_file()
    )
    # This is a recipe-development gate, not an acceptance test for a predictor.
    lower_hits = sum(natural[b] for b in ("8193-12288", "12289-16384", ">16384"))
    upper_hits = sum(natural[b] for b in ("12289-16384", ">16384"))
    promising = complete and lower_hits >= 6 and upper_hits >= 3
    result = {
        "status": "COLLECTION_COMPLETE" if complete else "INCOMPLETE_DIAGNOSTIC",
        "capture_returncode": capture_returncode,
        "planned_requests": len(inputs),
        "completed_requests": len(seen),
        "natural_eos_by_output_band": dict(natural),
        "censored_observed_lower_bounds_by_band": dict(censored),
        "natural_target_hits": dict(target_hits),
        "natural_gt8192": lower_hits,
        "natural_gt12288": upper_hits,
        "recipe_promising_for_expansion": promising,
        "recipe_gate": "at least 6 natural outputs >8192 and 3 >12288",
        "next_action": (
            "review response quality and length diversity before formal collection"
            if promising
            else "review premature EOS/censoring and revise recipe"
        ),
        "training_started": False,
        "formal_holdout_eligible": False,
        "censored_are_not_exact_lengths": True,
    }
    (out_dir / "pilot_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    emit(
        f"已完成{len(seen)}/12条；自然输出>8192有{lower_hits}条，"
        f">12288有{upper_hits}条；配方覆盖{'达到' if promising else '未达到'}试采目标"
    )
    return result


def capture_command(args, manifest):
    """Reuse the existing serial single-GPU capture and progress mechanism."""
    return [
        sys.executable,
        str(Path(__file__).with_name("run_predictor_capture.py")),
        "--input",
        str(args.out_dir / "input_requests.jsonl"),
        "--model",
        str(args.model),
        "--out-dir",
        str(args.out_dir / "capture"),
        "--expected-revision",
        args.expected_revision,
        "--expected-input-sha256",
        manifest["input_sha256"],
        "--expected-gpu-name",
        "NVIDIA A100-PCIE-40GB",
        "--expected-gpu-count",
        str(args.expected_gpu_count),
        "--interval",
        "20",
        "--feature-layer",
        "decoder:31",
        "--max-model-len",
        str(MAX_MODEL_LEN),
        "--max-tokens",
        str(MAX_OUTPUT_TOKENS),
        "--gpu-memory-utilization",
        "0.85",
        "--temperature",
        "0",
        "--save-token-ids",
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--expected-recipe-sha256", required=True)
    parser.add_argument("--expected-gpu-count", required=True, type=int)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    actual_sha = hashlib.sha256(serialized(recipe())).hexdigest()
    if actual_sha != args.expected_recipe_sha256:
        parser.error("preregistered recipe SHA differs")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if revision != args.expected_revision:
        parser.error("HEAD differs from expected revision")
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        parser.error("worktree has uncommitted files")
    config = json.loads((args.model / "config.json").read_text(encoding="utf-8"))
    from transformers import AutoTokenizer

    emit("正在核对12条新配方与实际tokenizer上下文预算")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    manifest = prepare(args.out_dir, tokenizer, config)
    emit("预算通过：12条，decoder:31，20 token采样，自然EOS；本轮只采集")
    if args.prepare_only:
        return
    started = time.monotonic()
    code = subprocess.call(capture_command(args, manifest))
    result = summarize(args.out_dir, tokenizer, capture_returncode=code)
    result["collection_wall_time_s"] = time.monotonic() - started
    (args.out_dir / "pilot_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if result["status"] != "COLLECTION_COMPLETE":
        raise RuntimeError("pilot incomplete; retrieve the diagnostic archive")


if __name__ == "__main__":
    main()
