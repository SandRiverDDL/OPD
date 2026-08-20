#!/usr/bin/env python3
"""Generic YAML-driven launcher for the synchronous OPD trainer."""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_yaml_with_extends(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text()) or {}
    parent = raw.pop("extends", None)
    if parent is None:
        return raw
    parent_path = (path.parent / parent).resolve()
    if not parent_path.is_file():
        raise FileNotFoundError(f"YAML parent does not exist: {parent_path}")
    return deep_merge(load_yaml_with_extends(parent_path), raw)


def expand(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(os.path.expanduser(value))
    if isinstance(value, list):
        return [expand(item) for item in value]
    if isinstance(value, dict):
        return {key: expand(item) for key, item in value.items()}
    return value


def repo_path(repo_root: Path, value: str) -> Path:
    path = Path(os.path.expandvars(os.path.expanduser(value)))
    return path if path.is_absolute() else repo_root / path


def find_hf_snapshot(repo_id: str, kind: str, candidates: list[str]) -> Path | None:
    repo_dir_name = f"{kind}--{repo_id.replace('/', '--')}"
    for candidate in candidates:
        if not candidate:
            continue
        candidate_path = Path(candidate).expanduser()
        hub_roots = [candidate_path / "hub"]
        if candidate_path.name == "hub":
            hub_roots.append(candidate_path)
        for hub in hub_roots:
            repo_dir = hub / repo_dir_name
            refs_main = repo_dir / "refs" / "main"
            if refs_main.is_file():
                revision = refs_main.read_text().strip()
                snapshot = repo_dir / "snapshots" / revision
                if snapshot.is_dir():
                    return snapshot
    return None


def resolve_model_path(
    repo_root: Path,
    configured: str | None,
    env_name: str,
    repo_id: str,
    candidates: list[str],
) -> Path:
    if configured:
        path = repo_path(repo_root, configured)
        if path.exists():
            return path
        raise FileNotFoundError(f"{env_name} configured path does not exist: {path}")
    env_value = os.environ.get(env_name)
    if env_value:
        path = Path(env_value).expanduser()
        if path.exists():
            return path
        raise FileNotFoundError(f"{env_name} points to a missing path: {path}")
    snapshot = find_hf_snapshot(repo_id, "models", candidates)
    if snapshot is None:
        raise FileNotFoundError(
            f"Cannot resolve model {repo_id}; set {env_name} or provide an explicit YAML path."
        )
    return snapshot


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")


def as_bool(value: Any) -> str:
    return "True" if bool(value) else "False"


def hydra_arg(key: str, value: Any) -> str:
    if isinstance(value, bool):
        value = as_bool(value)
    return f"{key}={value}"


def build_effective_config(repo_root: Path, config_path: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    config = expand(load_yaml_with_extends(config_path))
    env_cfg = config["environment"]
    data_cfg = config["data"]
    model_cfg = config["model"]
    length_cfg = config["length"]
    rollout_cfg = config["rollout"]
    train_cfg = config["training"]
    system_cfg = config["system"]

    hf_home = os.environ.get("HF_HOME")
    candidates = []
    if hf_home:
        candidates.append(hf_home)
    hf_hub_cache = os.environ.get("HF_HUB_CACHE")
    if hf_hub_cache:
        candidates.append(hf_hub_cache)
    candidates.extend(env_cfg.get("hf_home_candidates", []))

    # Some instances put the repository below an extra namespace directory,
    # e.g. <share>/hunyuan/<user>/dev/lab/OPD, while the persistent cache is
    # at <share>/<user>/.cache/huggingface.  Generate only the corresponding
    # shallow candidates instead of scanning the shared filesystem.
    for source in (os.environ.get("TAIJI_IDE_DEV_DIR"), str(repo_root)):
        if not source:
            continue
        source_path = Path(source).expanduser()
        parts = source_path.parts
        share_markers = [index for index, part in enumerate(parts) if part.startswith("share_")]
        if not share_markers:
            continue
        share_index = share_markers[-1]
        share_root = Path(*parts[: share_index + 1])
        for component in parts[share_index + 1 :]:
            if component in {".", ".."}:
                continue
            candidates.append(str(share_root / component / ".cache" / "huggingface"))

    candidates.append(str(Path.home() / ".cache" / "huggingface"))
    # Deduplicate after expansion.
    candidates = list(dict.fromkeys(str(Path(item).expanduser()) for item in candidates if item))

    student = resolve_model_path(
        repo_root,
        model_cfg.get("student_path"),
        "HF_STUDENT_MODEL_PATH",
        env_cfg["hf_student_repo"],
        candidates,
    )
    teacher = resolve_model_path(
        repo_root,
        model_cfg.get("teacher_path"),
        "HF_TEACHER_MODEL_PATH",
        env_cfg["hf_teacher_repo"],
        candidates,
    )
    train_dataset = repo_path(repo_root, data_cfg["train_dataset"])
    test_data_dir = repo_path(repo_root, data_cfg["test_data_dir"])
    test_dataset = test_data_dir / "MATH-500" / "test.parquet"
    require_file(train_dataset, "training dataset")
    require_file(test_dataset, "validation dataset")

    prompt_batch = int(train_cfg["prompt_batch_size"])
    mini_batch = int(train_cfg["ppo_mini_batch_size"])
    n_responses = int(rollout_cfg["n_responses"])
    if rollout_cfg.get("eopd_enabled", False):
        eopd_top_k = int(rollout_cfg.get("eopd_top_k", 16))
        if eopd_top_k <= 0:
            raise ValueError("EOPD requires rollout.eopd_top_k to be positive")
        # EOPD uses the existing top-k collection path to obtain the
        # teacher distribution used by its forward-KL auxiliary loss.
        rollout_cfg["log_prob_top_k"] = eopd_top_k
    world_size = int(system_cfg["n_gpus_per_node"]) * int(system_cfg["nnodes"])
    if prompt_batch <= 0 or mini_batch <= 0 or n_responses <= 0:
        raise ValueError("prompt_batch_size, ppo_mini_batch_size and n_responses must be positive")
    if prompt_batch % int(system_cfg["parallel_size"]) != 0:
        raise ValueError("prompt_batch_size must be divisible by parallel_size")
    if mini_batch % int(system_cfg["parallel_size"]) != 0:
        raise ValueError("ppo_mini_batch_size must be divisible by parallel_size")

    ckpt_path = repo_root / "checkpoint" / config["experiment"]["name"]
    effective_paths = {
        "student": student,
        "teacher": teacher,
        "train_dataset": train_dataset,
        "test_dataset": test_dataset,
        "ckpt_path": ckpt_path,
    }
    return config, effective_paths


def build_command(config: dict[str, Any], paths: dict[str, Path]) -> list[str]:
    data_cfg = config["data"]
    model_cfg = config["model"]
    length_cfg = config["length"]
    rollout_cfg = config["rollout"]
    train_cfg = config["training"]
    system_cfg = config["system"]
    experiment_cfg = config["experiment"]

    test_files = json.dumps([str(paths["test_dataset"])], separators=(",", ":"))
    max_model_len = max(
        int(length_cfg["max_prompt"]) + int(length_cfg["max_response"]),
        int(length_cfg["max_prompt"]) + int(length_cfg["max_val_response"]),
    )
    args = [
        "python3",
        "-m",
        "verl.trainer.main_ppo",
        hydra_arg("algorithm.adv_estimator", train_cfg["adv_estimator"]),
        hydra_arg("algorithm.grpo_outcome_weight", train_cfg["grpo_outcome_weight"]),
        hydra_arg("data.shuffle", data_cfg["shuffle"]),
        hydra_arg("data.train_files", str(paths["train_dataset"])),
        hydra_arg("data.val_files", test_files),
        hydra_arg("data.train_max_samples", data_cfg["train_max_samples"]),
        hydra_arg("data.val_max_samples", data_cfg["val_max_samples"]),
        hydra_arg("data.train_batch_size", train_cfg["prompt_batch_size"]),
        hydra_arg("data.max_prompt_length", length_cfg["max_prompt"]),
        hydra_arg("data.max_response_length", length_cfg["max_response"]),
        hydra_arg("data.filter_overlong_prompts", data_cfg["filter_overlong_prompts"]),
        "data.truncation=error",
        "data.return_raw_chat=True",
        hydra_arg("actor_rollout_ref.model.path", str(paths["student"])),
        "actor_rollout_ref.model.use_remove_padding=True",
        "actor_rollout_ref.model.enable_activation_offload=True",
        "actor_rollout_ref.model.enable_gradient_checkpointing=True",
        hydra_arg("actor_rollout_ref.actor.optim.lr", train_cfg["learning_rate"]),
        hydra_arg("actor_rollout_ref.actor.ppo_mini_batch_size", train_cfg["ppo_mini_batch_size"]),
        "actor_rollout_ref.actor.use_dynamic_bsz=True",
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1",
        hydra_arg(
            "actor_rollout_ref.actor.ppo_max_token_len_per_gpu",
            train_cfg["model_max_token_len_per_gpu"],
        ),
        hydra_arg("actor_rollout_ref.actor.ulysses_sequence_parallel_size", system_cfg["parallel_size"]),
        hydra_arg("actor_rollout_ref.actor.use_kl_loss", train_cfg["use_kl"]),
        hydra_arg("actor_rollout_ref.actor.loss_agg_mode", train_cfg["loss_agg_mode"]),
        "actor_rollout_ref.actor.fsdp_config.param_offload=False",
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload=False",
        "actor_rollout_ref.actor.fsdp_config.forward_prefetch=True",
        hydra_arg("actor_rollout_ref.actor.fsdp_config.model_dtype", model_cfg["dtype"]),
        hydra_arg("actor_rollout_ref.actor.grad_clip", train_cfg["grad_clip"]),
        hydra_arg("actor_rollout_ref.rollout.max_num_batched_tokens", rollout_cfg["vllm_max_num_batched_tokens"]),
        hydra_arg("actor_rollout_ref.rollout.max_num_seqs", rollout_cfg["vllm_max_num_seqs"]),
        "actor_rollout_ref.ref.fsdp_config.param_offload=True",
        hydra_arg("actor_rollout_ref.ref.fsdp_config.model_dtype", model_cfg["dtype"]),
        "actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True",
        hydra_arg(
            "actor_rollout_ref.ref.log_prob_max_token_len_per_gpu",
            train_cfg["ref_log_prob_max_token_len_per_gpu"],
        ),
        "actor_rollout_ref.rollout.name=vllm",
        hydra_arg("actor_rollout_ref.rollout.temperature", rollout_cfg["temperature"]),
        "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True",
        hydra_arg(
            "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu",
            train_cfg["rollout_log_prob_max_token_len_per_gpu"],
        ),
        hydra_arg("+actor_rollout_ref.rollout.log_prob_top_k", rollout_cfg["log_prob_top_k"]),
        hydra_arg("+actor_rollout_ref.rollout.top_k_strategy", rollout_cfg["top_k_strategy"]),
        hydra_arg("+actor_rollout_ref.rollout.reward_weight_mode", rollout_cfg["reward_weight_mode"]),
        hydra_arg("+actor_rollout_ref.rollout.teacher_temperature", rollout_cfg["teacher_temperature"]),
        hydra_arg("actor_rollout_ref.rollout.eopd_enabled", rollout_cfg.get("eopd_enabled", False)),
        hydra_arg("actor_rollout_ref.rollout.eopd_top_k", rollout_cfg.get("eopd_top_k", 16)),
        hydra_arg(
            "actor_rollout_ref.actor.policy_loss.loss_mode",
            "eopd" if rollout_cfg.get("eopd_enabled", False) else "vanilla",
        ),
        hydra_arg(
            "actor_rollout_ref.actor.policy_loss.eopd_entropy_threshold",
            rollout_cfg.get("eopd_entropy_threshold", 0.8),
        ),
        hydra_arg(
            "actor_rollout_ref.actor.policy_loss.eopd_forward_kl_coef",
            rollout_cfg.get("eopd_forward_kl_coef", 1.0),
        ),
        hydra_arg(
            "actor_rollout_ref.rollout.tensor_model_parallel_size",
            system_cfg["parallel_size"],
        ),
        hydra_arg(
            "actor_rollout_ref.rollout.gpu_memory_utilization",
            rollout_cfg["vllm_gpu_memory_utilization"],
        ),
        hydra_arg("actor_rollout_ref.rollout.max_model_len", max_model_len),
        hydra_arg("actor_rollout_ref.rollout.n", rollout_cfg["n_responses"]),
        "actor_rollout_ref.rollout.val_kwargs.do_sample=True",
        hydra_arg("+actor_rollout_ref.rollout.val_kwargs.max_tokens", length_cfg["max_val_response"]),
        hydra_arg("actor_rollout_ref.rollout.val_kwargs.n", rollout_cfg["val_n"]),
        "actor_rollout_ref.rollout.val_kwargs.temperature=0.7",
        "actor_rollout_ref.rollout.val_kwargs.top_p=0.95",
        hydra_arg("actor_rollout_ref.rollout.repetition_penalty", rollout_cfg["repetition_penalty"]),
        "actor_rollout_ref.rollout.calculate_log_probs=True",
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1",
        "reward_model.enable=True",
        hydra_arg("+reward_model.reward_kwargs.enable_format_reward", train_cfg["enable_format_reward"]),
        hydra_arg("reward_model.model.path", str(paths["teacher"])),
        "reward_model.model.input_tokenizer=null",
        "reward_model.model.use_remove_padding=True",
        "reward_model.model.fsdp_config.param_offload=False",
        hydra_arg("+reward_model.model.dtype", model_cfg["dtype"]),
        hydra_arg("reward_model.forward_max_token_len_per_gpu", train_cfg["teacher_max_token_len_per_gpu"]),
        "reward_model.micro_batch_size_per_gpu=24",
        'custom_reward_function.path=verl/verl/utils/reward_score/ttrl_math/__init__.py',
        "custom_reward_function.name=reward_func",
        "trainer.val_before_train=False",
        "trainer.log_val_generations=2",
        hydra_arg("trainer.project_name", experiment_cfg["project_name"]),
        hydra_arg("trainer.experiment_name", experiment_cfg["name"]),
        hydra_arg("trainer.validation_data_dir", f"validation_log/{experiment_cfg['name']}"),
        hydra_arg("trainer.logger", system_cfg["logger"]),
        hydra_arg("trainer.n_gpus_per_node", system_cfg["n_gpus_per_node"]),
        hydra_arg("trainer.nnodes", system_cfg["nnodes"]),
        hydra_arg("trainer.save_freq", system_cfg["save_freq"]),
        hydra_arg("trainer.test_freq", system_cfg["test_freq"]),
        hydra_arg("trainer.total_epochs", train_cfg["total_epochs"]),
        hydra_arg("trainer.total_training_steps", train_cfg["total_training_steps"]),
        hydra_arg("trainer.default_local_dir", str(paths["ckpt_path"])),
        hydra_arg("trainer.is_plot", system_cfg["is_plot"]),
    ]
    return args


def print_summary(config: dict[str, Any], paths: dict[str, Path], command: list[str]) -> None:
    train_cfg = config["training"]
    rollout_cfg = config["rollout"]
    system_cfg = config["system"]
    prompt_batch = int(train_cfg["prompt_batch_size"])
    n_responses = int(rollout_cfg["n_responses"])
    world_size = int(system_cfg["n_gpus_per_node"]) * int(system_cfg["nnodes"])
    trajectories = prompt_batch * n_responses
    local_trajectories = trajectories / world_size
    print("Resolved OPD configuration:")
    print(f"  experiment={config['experiment']['name']}")
    print(f"  prompt_batch_size={prompt_batch}")
    print(f"  n_responses={n_responses}")
    print(f"  trajectories_per_step={trajectories}")
    print(f"  trajectories_per_gpu={local_trajectories:g}")
    print(f"  ppo_mini_batch_size={train_cfg['ppo_mini_batch_size']}")
    print(f"  total_training_steps={train_cfg['total_training_steps']}")
    print(f"  learning_rate={train_cfg['learning_rate']}")
    print(f"  vllm_max_num_seqs={rollout_cfg['vllm_max_num_seqs']}")
    print(f"  vllm_max_num_batched_tokens={rollout_cfg['vllm_max_num_batched_tokens']}")
    print(f"  student={paths['student']}")
    print(f"  teacher={paths['teacher']}")
    print(f"  train_dataset={paths['train_dataset']}")
    print(f"  ckpt_path={paths['ckpt_path']}")
    print("  command:")
    print("    " + " ".join(command))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    config_path = args.config if args.config.is_absolute() else repo_root / args.config
    config_path = config_path.resolve()
    config, paths = build_effective_config(repo_root, config_path)
    command = build_command(config, paths)
    print_summary(config, paths, command)
    if args.dry_run:
        return 0

    if paths["ckpt_path"].exists():
        raise FileExistsError(
            f"Checkpoint directory already exists; refusing to overwrite: {paths['ckpt_path']}"
        )

    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ.setdefault("PYTHONPATH", str(repo_root / "verl"))
    if config["system"].get("cuda_launch_blocking", True):
        os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")
    os.environ.setdefault(
        "TOKENIZERS_PARALLELISM",
        "true" if config["system"].get("tokenizers_parallelism", True) else "false",
    )
    os.environ.setdefault("HYDRA_FULL_ERROR", "1")
    os.environ.setdefault("NCCL_TIMEOUT", "7200")
    os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")

    subprocess.run(["ray", "stop", "--force"], check=False)
    subprocess.run(["ray", "start", "--head"], check=True)
    training_log_file = os.environ.get("OPD_TRAIN_LOG_FILE")
    if training_log_file:
        training_log_path = Path(os.path.expandvars(os.path.expanduser(training_log_file)))
        if not training_log_path.is_absolute():
            training_log_path = repo_root / training_log_path
        training_log_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"training_log={training_log_path}", flush=True)
        with training_log_path.open("a", encoding="utf-8") as stream:
            subprocess.run(
                ["python3", "-m", "verl.trainer.main_ppo", *command[3:]],
                cwd=repo_root,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=True,
            )
    else:
        subprocess.run(
            ["python3", "-m", "verl.trainer.main_ppo", *command[3:]],
            cwd=repo_root,
            check=True,
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
