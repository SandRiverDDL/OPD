#!/usr/bin/env python3
"""Run a resumable checkpoint-selection evaluation.

The evaluator deliberately treats the filesystem as the source of truth:

* every completed checkpoint evaluation is written to ``state.json`` and TSV;
* every state transition is appended to ``events.jsonl``;
* each stage conclusion is written before the next stage starts;
* the final report is written atomically.

If the process is killed while one checkpoint is being evaluated, completed
checkpoints and the last persisted stage conclusion remain readable.  A later
invocation with the same ``--run-name`` resumes from the existing files.
"""

from __future__ import annotations

import argparse
import collections
import copy
import datetime as dt
import json
import math
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
EVAL_DIR = REPO_ROOT / "scripts" / "val" / "eval"
EVAL_OUTPUT_ROOT = EVAL_DIR / "justrl_eval_outputs"
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

DEFAULT_STEPS = list(range(20, 301, 20))
DEFAULT_GPUS = "0,1,2,3,4,5,6,7"
DEFAULT_TOKENIZER = (
    "/apdcephfs_szcf/share_304335953/hunyuan/halanchen/.cache/huggingface/"
    "hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/"
    "snapshots/ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562"
)

# These are calibration values from the preceding AIME25 repeated-evaluation
# measurements.  avg@4 was only a rough estimate from sliced avg@16 rows, so
# it is explicitly marked as provisional in all reports.
REFERENCE_SD_PP = {
    "avg4": 1.60,
    "avg16": 1.016,
}


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat(timespec="seconds")


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with tmp.open("w", encoding="utf-8") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
    )


def append_event(path: Path, event: str, **fields: Any) -> None:
    record = {"time": now(), "event": event, **fields}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def command_text(command: list[str]) -> str:
    import shlex

    return shlex.join(command)


def run_logged(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    log_path: Path,
    events_path: Path,
    label: str,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    append_event(events_path, "command_start", label=label, command=command_text(command))
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{now()}] COMMAND {command_text(command)}\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        return_code = process.wait()
        log.write(f"[{now()}] RETURN_CODE {return_code}\n")
        log.flush()
    append_event(
        events_path,
        "command_end",
        label=label,
        command=command_text(command),
        return_code=return_code,
        log=str(log_path),
    )
    if return_code != 0:
        raise RuntimeError(
            f"{label} failed with return code {return_code}; see {log_path}"
        )


def validate_rollout_file(
    output_path: Path,
    expected_questions: int,
    expected_n: int,
) -> dict[str, Any]:
    counts: collections.Counter[int] = collections.Counter()
    rows = 0
    with output_path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            item = json.loads(line)
            example_id = int(item["example_id"])
            if not 0 <= example_id < expected_questions:
                raise ValueError(
                    f"{output_path}: line {line_no}: invalid example_id={example_id}"
                )
            counts[example_id] += 1
            rows += 1

    missing = [
        example_id
        for example_id in range(expected_questions)
        if counts[example_id] != expected_n
    ]
    if rows != expected_questions * expected_n or missing:
        raise ValueError(
            f"{output_path}: expected {expected_questions * expected_n} rows, "
            f"got {rows}; per-question mismatch={missing[:10]}"
        )
    return {
        "rows": rows,
        "questions": expected_questions,
        "n": expected_n,
    }


def load_rollouts(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            rows.append(json.loads(line))
    return rows


def analyze_rollouts(
    output_path: Path,
    analysis_path: Path,
    *,
    task: str,
    step: int,
    n: int,
) -> dict[str, Any]:
    """Grade rollouts question-by-question using the benchmark rule grader."""
    # Import lazily so a state-only inspection does not initialize any model
    # or tokenizer machinery from grade.py.
    from utils import grade_answer_verl

    rows = load_rollouts(output_path)
    grouped: dict[int, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        grouped[int(row["example_id"])].append(row)

    question_rows = []
    rollout_correct = 0
    format_errors = 0
    output_lengths = []
    for example_id in sorted(grouped):
        candidates = sorted(
            grouped[example_id],
            key=lambda item: (int(item.get("seed", 0)), str(item.get("response", ""))),
        )
        correct = []
        lengths = []
        per_rollout = []
        for candidate in candidates:
            response = str(candidate.get("response", ""))
            answer = candidate.get("answer")
            is_correct = bool(grade_answer_verl(response, answer))
            has_boxed = "boxed" in response
            length = len(response)
            correct.append(is_correct)
            lengths.append(length)
            rollout_correct += int(is_correct)
            format_errors += int(not has_boxed)
            output_lengths.append(length)
            per_rollout.append(
                {
                    "seed": candidate.get("seed"),
                    "correct": is_correct,
                    "has_boxed": has_boxed,
                    "output_length_chars": length,
                }
            )
        question_rows.append(
            {
                "example_id": example_id,
                "correct": correct,
                "score": sum(correct) / len(correct) if correct else 0.0,
                "output_lengths_chars": lengths,
                "rollouts": per_rollout,
            }
        )

    if len(question_rows) == 0:
        raise ValueError(f"No rollout rows found in {output_path}")
    if any(len(row["correct"]) != n for row in question_rows):
        raise ValueError(f"Unexpected per-question rollout count in {output_path}")

    result = {
        "schema_version": 1,
        "task": task,
        "step": step,
        "n": n,
        "num_questions": len(question_rows),
        "num_rollouts": len(rows),
        "mean_score": rollout_correct / len(rows),
        "correct_rollouts": rollout_correct,
        "format_error_rollouts": format_errors,
        "avg_output_length_chars": (
            sum(output_lengths) / len(output_lengths) if output_lengths else 0.0
        ),
        "questions": question_rows,
        "source_rollouts": str(output_path),
        "created_at": now(),
    }
    atomic_write_json(analysis_path, result)
    return result


def paired_bootstrap(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    seed: int,
    samples: int,
) -> dict[str, float | int | str]:
    left_questions = {int(row["example_id"]): row for row in left["questions"]}
    right_questions = {int(row["example_id"]): row for row in right["questions"]}
    ids = sorted(set(left_questions) & set(right_questions))
    if len(ids) < 2:
        raise ValueError("Need at least two common questions for paired bootstrap")
    differences = np.asarray(
        [
            float(left_questions[example_id]["score"])
            - float(right_questions[example_id]["score"])
            for example_id in ids
        ],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(differences), size=(samples, len(differences)))
    bootstrap_means = differences[indices].mean(axis=1)
    low, high = np.quantile(bootstrap_means, [0.025, 0.975])
    return {
        "left_step": int(left["step"]),
        "right_step": int(right["step"]),
        "delta_left_minus_right": float(differences.mean()),
        "delta_left_minus_right_pp": float(differences.mean() * 100.0),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "ci95_low_pp": float(low * 100.0),
        "ci95_high_pp": float(high * 100.0),
        "questions": len(ids),
        "bootstrap_samples": samples,
        "bootstrap_seed": seed,
    }


def threshold_pp(stage: str) -> float:
    sd = REFERENCE_SD_PP["avg4" if stage == "avg4" else "avg16"]
    return 2.0 * math.sqrt(2.0) * sd


def conclude_stage(
    stage: str,
    analyses: dict[int, dict[str, Any]],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    if not analyses:
        raise ValueError(f"No completed analyses for stage {stage}")
    rows = []
    for step, analysis in analyses.items():
        rows.append(
            {
                "step": int(step),
                "mean_score": float(analysis["mean_score"]),
                "score_pp": float(analysis["mean_score"] * 100.0),
                "correct_rollouts": int(analysis["correct_rollouts"]),
                "num_rollouts": int(analysis["num_rollouts"]),
                "format_error_rollouts": int(analysis["format_error_rollouts"]),
                "avg_output_length_chars": float(
                    analysis["avg_output_length_chars"]
                ),
                "analysis_path": analysis["analysis_path"],
            }
        )
    rows.sort(key=lambda row: (-row["mean_score"], row["step"]))
    best_step = rows[0]["step"]
    best_analysis = analyses[best_step]
    margin_threshold = threshold_pp(stage)
    comparisons = []
    survivors = []
    for row in sorted(rows, key=lambda item: item["step"]):
        step = row["step"]
        if step == best_step:
            comparison = {
                "step": step,
                "vs_best": True,
                "delta_to_best_pp": 0.0,
                "ci95_low_pp": 0.0,
                "ci95_high_pp": 0.0,
                "clearly_worse": False,
            }
        else:
            bootstrap = paired_bootstrap(
                analyses[step],
                best_analysis,
                seed=bootstrap_seed + step,
                samples=bootstrap_samples,
            )
            # left= candidate, right=best.  A candidate is rejected only when
            # both the raw gap exceeds the calibrated noise threshold and the
            # paired bootstrap CI excludes zero in the worse direction.
            gap_pp = float(best_analysis["mean_score"] - analyses[step]["mean_score"]) * 100.0
            clearly_worse = (
                gap_pp > margin_threshold
                and float(bootstrap["ci95_high_pp"]) < 0.0
            )
            comparison = {
                "step": step,
                "vs_best": False,
                "delta_to_best_pp": -gap_pp,
                "gap_to_best_pp": gap_pp,
                "ci95_low_pp": bootstrap["ci95_low_pp"],
                "ci95_high_pp": bootstrap["ci95_high_pp"],
                "clearly_worse": clearly_worse,
                "bootstrap": bootstrap,
            }
        comparisons.append(comparison)
        if not comparison["clearly_worse"]:
            survivors.append(step)

    return {
        "schema_version": 1,
        "stage": stage,
        "completed_steps": [row["step"] for row in sorted(rows, key=lambda x: x["step"])],
        "raw_ranked": rows,
        "raw_best_step": best_step,
        "raw_best_score_pp": rows[0]["score_pp"],
        "reference_sd_pp": REFERENCE_SD_PP[stage],
        "reference_sd_status": (
            "provisional: avg@4 SD was estimated by slicing prior avg@16 rows"
            if stage == "avg4"
            else "measured from repeated avg@16 evaluations"
        ),
        "two_sigma_independent_difference_threshold_pp": margin_threshold,
        "comparisons_to_raw_best": comparisons,
        "survivors": sorted(survivors),
        "policy": (
            "Reject only if raw gap exceeds the calibrated two-sigma "
            "independent-evaluation noise estimate and paired-bootstrap "
            "95% CI excludes zero; unresolved candidates survive."
        ),
        "created_at": now(),
    }


def format_tsv(results: dict[str, dict[str, Any]], stage: str) -> str:
    header = (
        "stage\tstep\tmean_score\tscore_pp\tcorrect_rollouts\tnum_rollouts\t"
        "format_error_rollouts\tavg_output_length_chars\tanalysis_path\n"
    )
    lines = [header]
    for step in sorted(results, key=lambda value: int(value)):
        item = results[step]
        lines.append(
            "\t".join(
                [
                    stage,
                    str(step),
                    f"{float(item['mean_score']):.12g}",
                    f"{float(item['mean_score']) * 100.0:.12g}",
                    str(item["correct_rollouts"]),
                    str(item["num_rollouts"]),
                    str(item["format_error_rollouts"]),
                    f"{float(item['avg_output_length_chars']):.12g}",
                    str(item["analysis_path"]),
                ]
            )
            + "\n"
        )
    return "".join(lines)


def render_progress(state: dict[str, Any]) -> str:
    config = state["config"]
    lines = [
        f"# Checkpoint selection progress: `{state['run_name']}`",
        "",
        f"- status: **{state['status']}**",
        f"- updated: `{state['updated_at']}`",
        f"- task: `{config['task']}`",
        f"- checkpoint root: `{config['checkpoint_root']}`",
        f"- selection: `avg@4 -> avg@16`",
        f"- max tokens: `{config['max_tokens']}`",
        f"- max model len: `{config['max_model_len']}`",
        f"- max batched tokens: `{config['max_num_batched_tokens']}`",
        f"- continuous batching: `{config['continuous_batching']}`",
        "",
        "## Completed evaluations",
        "",
        "| Stage | Completed steps |",
        "|---|---|",
    ]
    for stage in ("avg4", "avg16"):
        completed = sorted(
            int(step) for step in state["results"][stage].keys()
        )
        lines.append(
            f"| {stage} | "
            + (", ".join(map(str, completed)) if completed else "none")
            + " |"
        )
    confirm_stages = sorted(
        stage for stage in state["results"] if stage.startswith("confirm_")
    )
    for stage in confirm_stages:
        completed = sorted(
            int(step) for step in state["results"][stage].keys()
        )
        lines.append(
            f"| {stage} | "
            + (", ".join(map(str, completed)) if completed else "none")
            + " |"
        )

    for stage in ("avg4", "avg16"):
        conclusion = (
            state.get("conclusions", {}).get(stage)
            or state.get("provisional_conclusions", {}).get(stage)
        )
        if not conclusion:
            continue
        lines.extend(
            [
                "",
                f"## Intermediate conclusion: {stage}"
                + (
                    ""
                    if stage in state.get("conclusions", {})
                    else " (provisional)"
                ),
                "",
                f"- raw best: step `{conclusion['raw_best_step']}` "
                f"({conclusion['raw_best_score_pp']:.3f}pp)",
                f"- calibrated 2σ difference threshold: "
                f"`{conclusion['two_sigma_independent_difference_threshold_pp']:.3f}pp`",
                f"- survivors: `{conclusion['survivors']}`",
                f"- note: {conclusion['reference_sd_status']}",
            ]
        )

    final = state.get("final_report")
    selection = state.get("selection_report")
    if selection:
        lines.extend(
            [
                "",
                "## Selection conclusion",
                "",
                f"- selected checkpoint: **step {selection['selected_step']}**",
                f"- selection report: `{selection['report_path']}`",
            ]
        )
    if final:
        lines.extend(
            [
                "",
                "## Final conclusion",
                "",
                f"- selected checkpoint: **step {final['selected_step']}**",
                f"- avg@16 raw best: step {final['avg16_raw_best_step']}",
                f"- final report: `{final['report_path']}`",
                f"- confirm results: `{final['confirm_results']}`",
            ]
        )
    lines.append("")
    return "\n".join(lines)


class Runner:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.work_dir = REPO_ROOT / "logs" / args.run_name
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.work_dir / "state.json"
        self.events_path = self.work_dir / "events.jsonl"
        self.progress_path = self.work_dir / "progress.md"
        self.config = self.build_config(args)
        self.state = self.load_or_initialize_state()
        self.stop_requested = False

        signal.signal(signal.SIGTERM, self.handle_signal)
        signal.signal(signal.SIGINT, self.handle_signal)

    def build_config(self, args: argparse.Namespace) -> dict[str, Any]:
        steps = sorted(set(args.steps))
        if not steps:
            raise ValueError("At least one checkpoint step is required")
        return {
            "task": args.task,
            "checkpoint_root": str(Path(args.checkpoint_root).resolve()),
            "steps": steps,
            "data_path": str(Path(args.data_path).resolve()),
            "gpus": args.gpus,
            "tokenizer": args.tokenizer,
            "max_tokens": args.max_tokens,
            "max_model_len": args.max_model_len,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "max_num_seqs": args.max_num_seqs,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "continuous_batching": True,
            "bootstrap_samples": args.bootstrap_samples,
            "bootstrap_seed": args.bootstrap_seed,
            "sd_reference_pp": copy.deepcopy(REFERENCE_SD_PP),
            "confirm_tasks": [
                {
                    "name": "AIME25",
                    "data_path": str(
                        (
                            REPO_ROOT
                            / "datasets"
                            / "test_data"
                            / "AIME25"
                            / "test.parquet"
                        ).resolve()
                    ),
                },
                {
                    "name": "AMC23",
                    "data_path": str(
                        (
                            REPO_ROOT
                            / "datasets"
                            / "test_data"
                            / "AMC23"
                            / "test.parquet"
                        ).resolve()
                    ),
                },
            ],
        }

    def load_or_initialize_state(self) -> dict[str, Any]:
        if self.state_path.exists():
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
            if state.get("config") != self.config:
                raise RuntimeError(
                    f"Existing run config differs from requested config: {self.state_path}"
                )
            state.setdefault("results", {"avg4": {}, "avg16": {}})
            state["results"].setdefault("avg4", {})
            state["results"].setdefault("avg16", {})
            state.setdefault("conclusions", {})
            state.setdefault("provisional_conclusions", {})
            state.setdefault("selection_report", None)
            state.setdefault("final_report", None)
            state.pop("error", None)
            state["status"] = "resumed"
            state["updated_at"] = now()
            self.persist_state(state)
            append_event(self.events_path, "run_resumed", state_path=str(self.state_path))
            return state

        state = {
            "schema_version": 1,
            "run_name": self.args.run_name,
            "status": "initialized",
            "created_at": now(),
            "updated_at": now(),
            "config": self.config,
            "results": {"avg4": {}, "avg16": {}},
            "conclusions": {},
            "provisional_conclusions": {},
            "selection_report": None,
            "final_report": None,
        }
        self.persist_state(state)
        append_event(
            self.events_path,
            "run_initialized",
            config=self.config,
            state_path=str(self.state_path),
        )
        return state

    def persist_state(self, state: dict[str, Any] | None = None) -> None:
        if state is not None:
            self.state = state
        self.state["updated_at"] = now()
        atomic_write_json(self.state_path, self.state)
        atomic_write_text(self.progress_path, render_progress(self.state))

    def handle_signal(self, signum: int, _frame: Any) -> None:
        self.stop_requested = True
        self.state["status"] = f"interrupted_by_signal_{signum}"
        self.persist_state()
        append_event(
            self.events_path,
            "run_interrupted",
            signal=signum,
            state_path=str(self.state_path),
        )
        raise KeyboardInterrupt

    def checkpoint_paths(
        self,
        task: str,
        step: int,
        n: int,
        *,
        phase: str,
    ) -> dict[str, Path]:
        ckpt_root = Path(self.config["checkpoint_root"])
        merged = ckpt_root / f"hf_step{step}"
        name = (
            f"{self.args.run_name}_cb32k_{phase}_{task.lower()}"
            f"_step{step}_avg{n}"
        )
        out_dir = EVAL_OUTPUT_ROOT / name
        output = out_dir / (
            f"{task.lower()}_t{self.config['temperature']}_"
            f"p{self.config['top_p']}_n{n}-MNT{self.config['max_tokens']}.jsonl"
        )
        return {
            "merged": merged,
            "name": Path(name),
            "out_dir": out_dir,
            "output": output,
            "grading": out_dir / "grading_results.json",
            "analysis": self.work_dir / f"{phase}_{task.lower()}_step{step}_avg{n}_question_scores.json",
            "eval_log": self.work_dir / f"{phase}_{task.lower()}_step{step}_avg{n}.eval.log",
            "merge_log": self.work_dir / f"step{step}.merge.log",
        }

    def ensure_merged(self, step: int, paths: dict[str, Path]) -> None:
        if paths["merged"].is_dir() and (paths["merged"] / "config.json").is_file():
            return
        if paths["merged"].exists():
            quarantine = paths["merged"].with_name(
                f"{paths['merged'].name}.partial.{dt.datetime.now().strftime('%Y%m%dT%H%M%S')}"
            )
            paths["merged"].rename(quarantine)
            append_event(
                self.events_path,
                "partial_merge_quarantined",
                step=step,
                old_path=str(paths["merged"]),
                quarantine_path=str(quarantine),
            )
        actor = Path(self.config["checkpoint_root"]) / f"global_step_{step}" / "actor"
        command = [
            sys.executable,
            "-u",
            "-m",
            "verl.model_merger",
            "merge",
            "--backend",
            "fsdp",
            "--local_dir",
            str(actor),
            "--target_dir",
            str(paths["merged"]),
            "--use_cpu_initialization",
        ]
        run_logged(
            command,
            cwd=REPO_ROOT,
            env=os.environ.copy(),
            log_path=paths["merge_log"],
            events_path=self.events_path,
            label=f"merge_step_{step}",
        )
        if not (paths["merged"] / "config.json").is_file():
            raise RuntimeError(f"merged checkpoint is incomplete: {paths['merged']}")

    def eval_env(
        self,
        model_path: Path,
        model_name: str,
        task: str,
        data_path: str,
        n: int,
    ) -> dict[str, str]:
        env = os.environ.copy()
        env.update(
            {
                "EVAL_TASK": task,
                "EVAL_DATA_PATH": data_path,
                "EVAL_N": str(n),
                "EVAL_MODEL_PATH": str(model_path),
                "EVAL_MODEL_NAME": model_name,
                # grade.py uses EVAL_NAME (not EVAL_MODEL_NAME) to locate the
                # JSONL directory and write grading_results.json.
                "EVAL_NAME": model_name,
                "EVAL_GPUS": self.config["gpus"],
                "EVAL_MAX_TOKENS": str(self.config["max_tokens"]),
                "EVAL_MAX_MODEL_LEN": str(self.config["max_model_len"]),
                "EVAL_MAX_NUM_SEQS": str(self.config["max_num_seqs"]),
                "EVAL_MAX_NUM_BATCHED_TOKENS": str(
                    self.config["max_num_batched_tokens"]
                ),
                "EVAL_GPU_MEMORY_UTILIZATION": str(
                    self.config["gpu_memory_utilization"]
                ),
                "EVAL_LENGTH_TOKENIZER": self.config["tokenizer"],
                "EVAL_REPLACE": "false",
            }
        )
        return env

    def evaluate_one(
        self,
        stage: str,
        step: int,
        n: int,
        *,
        task: str | None = None,
        data_path: str | None = None,
        phase: str | None = None,
    ) -> dict[str, Any]:
        key = str(step)
        task = task or self.config["task"]
        data_path = data_path or self.config["data_path"]
        phase = phase or stage
        self.state["results"].setdefault(stage, {})
        existing = self.state["results"][stage].get(key)
        paths = self.checkpoint_paths(task, step, n, phase=phase)
        output = paths["output"]
        result = paths["grading"]
        analysis_path = paths["analysis"]
        expected_questions = len(
            __import__("pandas").read_parquet(data_path)
        )

        # Recover a completed checkpoint from files even if the process was
        # killed after generation/grading but before state.json was updated.
        # In particular, grading may have been interrupted or misconfigured
        # while the rollout JSONL is already complete; retain that JSONL and
        # retry only the cheap grading step.
        existing_output_valid = False
        if output.is_file():
            try:
                validation = validate_rollout_file(output, expected_questions, n)
                existing_output_valid = True
                if not result.is_file():
                    env = self.eval_env(
                        paths["merged"], str(paths["name"]), task, data_path, n
                    )
                    run_logged(
                        [sys.executable, "-u", "grade.py"],
                        cwd=EVAL_DIR,
                        env=env,
                        log_path=paths["eval_log"],
                        events_path=self.events_path,
                        label=f"grade_existing_{phase}_step_{step}",
                    )
                if not result.is_file():
                    raise RuntimeError(
                        f"grading result was not produced after retry: {result}"
                    )
                if not analysis_path.is_file():
                    analysis = analyze_rollouts(
                        output,
                        analysis_path,
                        task=task,
                        step=step,
                        n=n,
                    )
                else:
                    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
                if int(analysis.get("n", -1)) != n or int(
                    analysis.get("num_questions", -1)
                ) != expected_questions:
                    raise ValueError(
                        f"question analysis metadata mismatch: "
                        f"analysis_n={analysis.get('n')} expected_n={n}, "
                        f"analysis_questions={analysis.get('num_questions')} "
                        f"expected_questions={expected_questions}"
                    )
                analysis["analysis_path"] = str(analysis_path)
                self.state["results"][stage][key] = {
                    **analysis,
                    "validation": validation,
                    "output_path": str(output),
                    "grading_path": str(result),
                }
                self.persist_state()
                append_event(
                    self.events_path,
                    "checkpoint_recovered_or_verified",
                    stage=stage,
                    step=step,
                    n=n,
                    score_pp=analysis["mean_score"] * 100.0,
                )
                return self.state["results"][stage][key]
            except Exception as exc:
                append_event(
                    self.events_path,
                    "checkpoint_existing_files_invalid",
                    stage=stage,
                    step=step,
                    n=n,
                    error=str(exc),
                )

        # A killed vLLM/grade process can leave an incomplete JSONL or an
        # incomplete output directory.  Preserve it for diagnosis, then let
        # this resumable runner start a clean attempt.
        if (
            not existing_output_valid
            and paths["out_dir"].exists()
            and any(paths["out_dir"].iterdir())
        ):
            quarantine = paths["out_dir"].with_name(
                f"{paths['out_dir'].name}.partial."
                f"{dt.datetime.now().strftime('%Y%m%dT%H%M%S')}"
            )
            paths["out_dir"].rename(quarantine)
            append_event(
                self.events_path,
                "partial_eval_quarantined",
                stage=stage,
                step=step,
                n=n,
                old_path=str(paths["out_dir"]),
                quarantine_path=str(quarantine),
            )

        self.state["status"] = f"evaluating_{stage}_step_{step}"
        self.persist_state()
        append_event(self.events_path, "checkpoint_start", stage=stage, step=step, n=n)

        paths["out_dir"].mkdir(parents=True, exist_ok=True)
        self.ensure_merged(step, paths)
        model_name = str(paths["name"])
        env = self.eval_env(paths["merged"], model_name, task, data_path, n)
        generate_command = [
            sys.executable,
            "-u",
            "gen_vllm.py",
            "--enable-thinking",
            "--gpus",
            self.config["gpus"],
            "--n",
            str(n),
            "--max-tokens",
            str(self.config["max_tokens"]),
            "--max-model-len",
            str(self.config["max_model_len"]),
            "--max-num-seqs",
            str(self.config["max_num_seqs"]),
            "--max-num-batched-tokens",
            str(self.config["max_num_batched_tokens"]),
            "--gpu-memory-utilization",
            str(self.config["gpu_memory_utilization"]),
        ]
        run_logged(
            generate_command,
            cwd=EVAL_DIR,
            env=env,
            log_path=paths["eval_log"],
            events_path=self.events_path,
            label=f"generate_{stage}_step_{step}",
        )
        validation = validate_rollout_file(output, expected_questions, n)

        grade_command = [sys.executable, "-u", "grade.py"]
        run_logged(
            grade_command,
            cwd=EVAL_DIR,
            env=env,
            log_path=paths["eval_log"],
            events_path=self.events_path,
            label=f"grade_{stage}_step_{step}",
        )
        if not result.is_file():
            raise RuntimeError(f"grading result was not produced: {result}")
        analysis = analyze_rollouts(
            output,
            analysis_path,
            task=task,
            step=step,
            n=n,
        )
        analysis["analysis_path"] = str(analysis_path)
        record = {
            **analysis,
            "validation": validation,
            "output_path": str(output),
            "grading_path": str(result),
        }
        self.state["results"][stage][key] = record
        self.persist_state()
        append_event(
            self.events_path,
            "checkpoint_complete",
            stage=stage,
            step=step,
            n=n,
            score_pp=analysis["mean_score"] * 100.0,
            correct_rollouts=analysis["correct_rollouts"],
            output_path=str(output),
            analysis_path=str(analysis_path),
        )
        return record

    def write_stage_files(self, stage: str, conclusion: dict[str, Any]) -> None:
        atomic_write_json(self.work_dir / f"{stage}_conclusion.json", conclusion)
        atomic_write_text(
            self.work_dir / f"{stage}_results.tsv",
            format_tsv(self.state["results"][stage], stage),
        )
        self.state["conclusions"][stage] = conclusion
        self.persist_state()
        append_jsonl(
            self.work_dir / "conclusions.jsonl",
            {
                "time": now(),
                "kind": "final_stage_conclusion",
                "stage": stage,
                "conclusion": conclusion,
            },
        )
        append_event(
            self.events_path,
            "stage_conclusion_persisted",
            stage=stage,
            raw_best_step=conclusion["raw_best_step"],
            survivors=conclusion["survivors"],
            conclusion_path=str(self.work_dir / f"{stage}_conclusion.json"),
        )

    def write_provisional_stage_files(self, stage: str) -> None:
        """Persist the best currently-known conclusion after every checkpoint."""
        records = self.state["results"][stage]
        if not records:
            return
        analyses = {int(step): record for step, record in records.items()}
        conclusion = conclude_stage(
            stage,
            analyses,
            bootstrap_samples=self.config["bootstrap_samples"],
            bootstrap_seed=self.config["bootstrap_seed"]
            + (0 if stage == "avg4" else 100000),
        )
        self.state["provisional_conclusions"][stage] = conclusion
        atomic_write_json(
            self.work_dir / f"{stage}_latest_conclusion.json",
            conclusion,
        )
        atomic_write_text(
            self.work_dir / f"{stage}_results.tsv",
            format_tsv(records, stage),
        )
        self.persist_state()
        append_jsonl(
            self.work_dir / "conclusions.jsonl",
            {
                "time": now(),
                "kind": "provisional_stage_conclusion",
                "stage": stage,
                "conclusion": conclusion,
            },
        )
        append_event(
            self.events_path,
            "provisional_conclusion_persisted",
            stage=stage,
            completed_steps=conclusion["completed_steps"],
            raw_best_step=conclusion["raw_best_step"],
            survivors=conclusion["survivors"],
            conclusion_path=str(
                self.work_dir / f"{stage}_latest_conclusion.json"
            ),
        )

    def write_selection_files(
        self,
        *,
        avg4_conclusion: dict[str, Any],
        avg16_conclusion: dict[str, Any],
    ) -> dict[str, Any]:
        """Persist the checkpoint choice before any confirm-set evaluation."""
        existing = self.state.get("selection_report")
        if existing:
            return existing

        noninferior = sorted(avg16_conclusion["survivors"])
        selected_step = (
            min(noninferior)
            if noninferior
            else int(avg16_conclusion["raw_best_step"])
        )
        selection = {
            "schema_version": 1,
            "run_name": self.args.run_name,
            "status": "selection_complete_confirm_pending",
            "selection_method": "AIME24 avg@4 -> avg@16",
            "selected_step": selected_step,
            "selected_checkpoint": str(
                Path(self.config["checkpoint_root"])
                / f"global_step_{selected_step}"
            ),
            "avg4_raw_best_step": int(avg4_conclusion["raw_best_step"]),
            "avg4_survivors": list(avg4_conclusion["survivors"]),
            "avg16_raw_best_step": int(avg16_conclusion["raw_best_step"]),
            "avg16_noninferior_steps": noninferior,
            "rule": (
                "At each stage, reject a checkpoint only when its absolute "
                "gap to the raw stage winner exceeds the calibrated two-sigma "
                "independent-evaluation noise estimate and the "
                "question-level paired-bootstrap 95% CI excludes zero. "
                "Among unresolved avg@16 candidates, select the earliest."
            ),
            "stage_conclusions": {
                "avg4": str(self.work_dir / "avg4_conclusion.json"),
                "avg16": str(self.work_dir / "avg16_conclusion.json"),
            },
            "report_path": str(self.work_dir / "selection_report.json"),
            "created_at": now(),
        }
        atomic_write_json(self.work_dir / "selection_report.json", selection)
        self.state["selection_report"] = selection
        self.state["status"] = "selection_complete_confirm_pending"
        self.persist_state()
        append_event(
            self.events_path,
            "selection_conclusion_persisted",
            selected_step=selected_step,
            report_path=str(self.work_dir / "selection_report.json"),
        )
        return selection

    def write_confirm_results(
        self,
        selection: dict[str, Any],
    ) -> dict[str, Any]:
        """Persist confirm-set results independently from selection logic."""
        confirm_results: dict[str, Any] = {
            "schema_version": 1,
            "run_name": self.args.run_name,
            "selected_step": int(selection["selected_step"]),
            "selected_checkpoint": selection["selected_checkpoint"],
            "selection_report": selection["report_path"],
            "selection_data_is_not_used_for_confirm_choice": True,
            "results": {},
            "created_at": now(),
        }
        selected_step = int(selection["selected_step"])
        for task_cfg in self.config["confirm_tasks"]:
            task = str(task_cfg["name"])
            stage = f"confirm_{task.lower()}"
            record = self.state["results"].get(stage, {}).get(str(selected_step))
            if record is None:
                record = self.evaluate_one(
                    stage,
                    selected_step,
                    16,
                    task=task,
                    data_path=str(task_cfg["data_path"]),
                    phase=stage,
                )
            result = {
                "task": task,
                "n": 16,
                "mean_score": float(record["mean_score"]),
                "score_pp": float(record["mean_score"] * 100.0),
                "correct_rollouts": int(record["correct_rollouts"]),
                "num_rollouts": int(record["num_rollouts"]),
                "format_error_rollouts": int(record["format_error_rollouts"]),
                "avg_output_length_chars": float(
                    record["avg_output_length_chars"]
                ),
                "output_path": record["output_path"],
                "grading_path": record["grading_path"],
                "question_analysis_path": record["analysis_path"],
            }
            confirm_results["results"][task.lower()] = result
            atomic_write_json(
                self.work_dir / f"{stage}_conclusion.json",
                {
                    "kind": "confirm_result",
                    **result,
                    "selection_report": selection["report_path"],
                    "created_at": now(),
                },
            )
            self.persist_state()
            append_event(
                self.events_path,
                "confirm_result_persisted",
                task=task,
                selected_step=selected_step,
                score_pp=result["score_pp"],
                report_path=str(self.work_dir / f"{stage}_conclusion.json"),
            )

        atomic_write_json(self.work_dir / "confirm_results.json", confirm_results)
        return confirm_results

    def write_final_report(
        self,
        selection: dict[str, Any],
        confirm_results: dict[str, Any],
    ) -> dict[str, Any]:
        selected_step = int(selection["selected_step"])
        selected_aime24 = self.state["results"]["avg16"].get(str(selected_step))
        if selected_aime24 is None:
            raise RuntimeError(
                f"selected AIME24 avg@16 result is missing for step {selected_step}"
            )
        selection_result = {
            "task": "AIME24",
            "n": 16,
            "mean_score": float(selected_aime24["mean_score"]),
            "score_pp": float(selected_aime24["mean_score"] * 100.0),
            "correct_rollouts": int(selected_aime24["correct_rollouts"]),
            "num_rollouts": int(selected_aime24["num_rollouts"]),
            "format_error_rollouts": int(
                selected_aime24["format_error_rollouts"]
            ),
            "avg_output_length_chars": float(
                selected_aime24["avg_output_length_chars"]
            ),
            "output_path": selected_aime24["output_path"],
            "grading_path": selected_aime24["grading_path"],
            "question_analysis_path": selected_aime24["analysis_path"],
        }
        final = {
            "schema_version": 1,
            "run_name": self.args.run_name,
            "status": "completed",
            "selected_step": int(selection["selected_step"]),
            "selected_checkpoint": selection["selected_checkpoint"],
            "selection_report": selection["report_path"],
            "confirm_results": str(self.work_dir / "confirm_results.json"),
            "selection_method": selection["selection_method"],
            "avg4_survivors": selection["avg4_survivors"],
            "avg16_raw_best_step": selection["avg16_raw_best_step"],
            "avg16_noninferior_steps": selection["avg16_noninferior_steps"],
            "rule": selection["rule"],
            "evaluation": {
                "selection_task": self.config["task"],
                "confirm_tasks": [
                    task["name"] for task in self.config["confirm_tasks"]
                ],
                "avg4_n": 4,
                "avg16_n": 16,
                "max_tokens": self.config["max_tokens"],
                "max_model_len": self.config["max_model_len"],
                "max_num_batched_tokens": self.config["max_num_batched_tokens"],
                "continuous_batching": True,
            },
            "results": {
                "aime24_selection": selection_result,
                **confirm_results["results"],
            },
            "stage_conclusions": {
                "avg4": str(self.work_dir / "avg4_conclusion.json"),
                "avg16": str(self.work_dir / "avg16_conclusion.json"),
            },
            "report_path": str(self.work_dir / "final_report.json"),
            "created_at": now(),
        }
        atomic_write_json(self.work_dir / "final_report.json", final)
        lines = [
            f"# Final checkpoint-selection report: `{self.args.run_name}`",
            "",
            f"- selected checkpoint: **step {final['selected_step']}**",
            f"- checkpoint: `{final['selected_checkpoint']}`",
            f"- selection method: `{final['selection_method']}`",
            f"- avg@4 survivors: `{final['avg4_survivors']}`",
            f"- avg@16 raw best: `step {final['avg16_raw_best_step']}`",
            f"- avg@16 non-inferior candidates: "
            f"`{final['avg16_noninferior_steps']}`",
            "",
            "## Confirm sets",
            "",
            "| Task | avg@16 | Correct rollouts | Format errors |",
            "|---|---:|---:|---:|",
        ]
        lines.append(
            f"| aime24 (selection) | {selection_result['score_pp']:.3f}pp | "
            f"{selection_result['correct_rollouts']}/"
            f"{selection_result['num_rollouts']} | "
            f"{selection_result['format_error_rollouts']} |"
        )
        for task, result in confirm_results["results"].items():
            lines.append(
                f"| {task} | {result['score_pp']:.3f}pp | "
                f"{result['correct_rollouts']}/{result['num_rollouts']} | "
                f"{result['format_error_rollouts']} |"
            )
        lines.extend(
            [
                "",
                "> Confirm-set results are reported after selection and are not "
                "used to choose the checkpoint.",
                "",
            ]
        )
        atomic_write_text(self.work_dir / "final_report.md", "\n".join(lines))
        return final

    def run(self) -> None:
        append_event(self.events_path, "run_started", run_name=self.args.run_name)
        self.state["status"] = "running"
        self.state.pop("error", None)
        self.persist_state()

        try:
            avg4_steps = self.config["steps"]
            if self.state.get("conclusions", {}).get("avg4"):
                avg4_steps = []
            for step in avg4_steps:
                if self.stop_requested:
                    raise KeyboardInterrupt
                self.evaluate_one("avg4", int(step), 4)
                self.write_provisional_stage_files("avg4")

            avg4_conclusion = self.state.get("conclusions", {}).get("avg4")
            if avg4_conclusion is None:
                avg4_analyses = {
                    int(step): record
                    for step, record in self.state["results"]["avg4"].items()
                }
                avg4_conclusion = conclude_stage(
                    "avg4",
                    avg4_analyses,
                    bootstrap_samples=self.config["bootstrap_samples"],
                    bootstrap_seed=self.config["bootstrap_seed"],
                )
                self.write_stage_files("avg4", avg4_conclusion)

            survivors = avg4_conclusion["survivors"]
            append_event(
                self.events_path,
                "next_stage_planned",
                stage="avg16",
                n=16,
                steps=survivors,
            )
            avg16_steps = [
                step for step in survivors
                if str(step) not in self.state["results"]["avg16"]
            ]
            if self.state.get("conclusions", {}).get("avg16"):
                avg16_steps = []
            for step in avg16_steps:
                if self.stop_requested:
                    raise KeyboardInterrupt
                self.evaluate_one("avg16", int(step), 16)
                self.write_provisional_stage_files("avg16")

            avg16_conclusion = self.state.get("conclusions", {}).get("avg16")
            if avg16_conclusion is None:
                avg16_analyses = {
                    int(step): record
                    for step, record in self.state["results"]["avg16"].items()
                }
                avg16_conclusion = conclude_stage(
                    "avg16",
                    avg16_analyses,
                    bootstrap_samples=self.config["bootstrap_samples"],
                    bootstrap_seed=self.config["bootstrap_seed"] + 100000,
                )
                self.write_stage_files("avg16", avg16_conclusion)

            selection = self.write_selection_files(
                avg4_conclusion=avg4_conclusion,
                avg16_conclusion=avg16_conclusion,
            )
            confirm_results = self.write_confirm_results(selection)
            final = self.write_final_report(selection, confirm_results)
            self.state["final_report"] = final
            self.state["status"] = "completed"
            self.persist_state()
            append_event(
                self.events_path,
                "final_conclusion_persisted",
                selected_step=selection["selected_step"],
                report_path=str(self.work_dir / "final_report.json"),
            )
        except KeyboardInterrupt:
            if not self.state["status"].startswith("interrupted"):
                self.state["status"] = "interrupted"
                self.persist_state()
                append_event(self.events_path, "run_interrupted")
            raise
        except Exception as exc:
            self.state["status"] = "failed"
            self.state["error"] = str(exc)
            self.persist_state()
            append_event(self.events_path, "run_failed", error=str(exc))
            raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument(
        "--checkpoint-root",
        default=str(REPO_ROOT / "checkpoint" / "opd_sky7b_sampled_token"),
    )
    parser.add_argument("--task", default="AIME24")
    parser.add_argument(
        "--data-path",
        default=str(REPO_ROOT / "datasets" / "test_data" / "AIME24" / "test.parquet"),
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        type=int,
        default=DEFAULT_STEPS,
    )
    parser.add_argument("--gpus", default=DEFAULT_GPUS)
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--max-tokens", type=int, default=31744)
    parser.add_argument("--max-model-len", type=int, default=33792)
    parser.add_argument("--max-num-batched-tokens", type=int, default=32768)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260823)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runner = Runner(args)
    runner.run()


if __name__ == "__main__":
    main()
