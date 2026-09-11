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


def normalize_cli_distillation_method(method: str) -> str:
    """Normalize method names without importing the runtime framework.

    The launcher intentionally stays usable for YAML validation and dry-runs
    before Ray/PyTorch are installed.  Runtime workers use the shared method
    registry in ``verl.utils.distillation``.
    """

    aliases = {
        "prune-opd": "pruneopd",
        "prune_opd": "pruneopd",
        "pruned-opd": "pruneopd",
        "pruned_opd": "pruneopd",
    }
    normalized = aliases.get(str(method).strip().lower(), str(method).strip().lower())
    supported = {"vanilla", "eopd", "poweropd", "aopd", "pruneopd"}
    if normalized not in supported:
        raise ValueError(
            f"Unsupported distillation.method={method!r}; expected one of: "
            f"{', '.join(sorted(supported))}"
        )
    return normalized


def resolve_distillation_config(config: dict[str, Any]) -> str:
    """Resolve the user-facing distillation method into internal rollout fields."""
    rollout_cfg = config["rollout"]
    distillation_cfg = config.get("distillation", {})
    method = str(distillation_cfg.get("method", "vanilla")).lower()

    # Preserve compatibility with the pre-selector EOPD/Prune-OPD configs.
    if method == "vanilla" and rollout_cfg.get("eopd_enabled", False):
        method = "eopd"
    prune_cfg = rollout_cfg.get("prune_opd", {})
    if method == "vanilla" and isinstance(prune_cfg, dict) and prune_cfg.get("enable", False):
        method = "pruneopd"

    method = normalize_cli_distillation_method(method)

    params = distillation_cfg.get("params") or {}
    if not isinstance(params, dict):
        raise ValueError("distillation.params must be a mapping")

    rollout_cfg["distillation_method"] = method
    rollout_cfg["eopd_enabled"] = method == "eopd"
    rollout_cfg["poweropd_reward_alpha"] = 5.0
    rollout_cfg["aopd_threshold"] = float(rollout_cfg.get("aopd_threshold", 0.1))
    rollout_cfg["aopd_opd_weight"] = float(rollout_cfg.get("aopd_opd_weight", 1.0))
    rollout_cfg["aopd_gkd_weight"] = float(rollout_cfg.get("aopd_gkd_weight", 1.0))
    rollout_cfg["aopd_jsd_beta"] = float(rollout_cfg.get("aopd_jsd_beta", 1.0))

    if method == "eopd":
        eopd_top_k = int(params.get("top_k", rollout_cfg.get("eopd_top_k", 16)))
        entropy_threshold = float(
            params.get("entropy_threshold", rollout_cfg.get("eopd_entropy_threshold", 0.8))
        )
        forward_kl_coef = float(
            params.get("forward_kl_coef", rollout_cfg.get("eopd_forward_kl_coef", 1.0))
        )
        if eopd_top_k <= 0:
            raise ValueError("EOPD requires distillation.params.top_k to be positive")
        if entropy_threshold < 0:
            raise ValueError("EOPD entropy_threshold must be non-negative")
        if forward_kl_coef < 0:
            raise ValueError("EOPD forward_kl_coef must be non-negative")
        rollout_cfg["eopd_top_k"] = eopd_top_k
        rollout_cfg["eopd_entropy_threshold"] = entropy_threshold
        rollout_cfg["eopd_forward_kl_coef"] = forward_kl_coef
        # EOPD reuses the top-k collection path for its auxiliary forward KL.
        rollout_cfg["log_prob_top_k"] = eopd_top_k
    elif method == "aopd":
        aopd_top_k = int(params.get("top_k", rollout_cfg.get("log_prob_top_k", 16)))
        threshold = float(params.get("threshold", rollout_cfg.get("aopd_threshold", 0.1)))
        opd_weight = float(params.get("opd_weight", rollout_cfg.get("aopd_opd_weight", 1.0)))
        gkd_weight = float(params.get("gkd_weight", rollout_cfg.get("aopd_gkd_weight", 1.0)))
        jsd_beta = float(params.get("jsd_beta", rollout_cfg.get("aopd_jsd_beta", 1.0)))
        if aopd_top_k <= 0:
            raise ValueError("AOPD requires distillation.params.top_k to be positive")
        if threshold < 0:
            raise ValueError("AOPD threshold must be non-negative")
        if opd_weight < 0 or gkd_weight < 0:
            raise ValueError("AOPD opd_weight and gkd_weight must be non-negative")
        if not 0.0 <= jsd_beta <= 1.0:
            raise ValueError("AOPD jsd_beta must be in [0, 1]")
        rollout_cfg["log_prob_top_k"] = aopd_top_k
        rollout_cfg["aopd_threshold"] = threshold
        rollout_cfg["aopd_opd_weight"] = opd_weight
        rollout_cfg["aopd_gkd_weight"] = gkd_weight
        rollout_cfg["aopd_jsd_beta"] = jsd_beta
    elif method == "pruneopd":
        prune_top_k = int(params.get("top_k", rollout_cfg.get("log_prob_top_k", 16)))
        existing_prune_cfg = rollout_cfg.get("prune_opd", {})
        metric = str(params.get("metric", existing_prune_cfg.get("metric", "overlap_ratio")))
        threshold = float(params.get("threshold", existing_prune_cfg.get("threshold", 0.7)))
        w_drop = float(params.get("w_drop", existing_prune_cfg.get("w_drop", 0.01)))
        w_base = float(params.get("w_base", existing_prune_cfg.get("w_base", 0.5)))
        if prune_top_k <= 0:
            raise ValueError("Prune-OPD requires distillation.params.top_k to be positive")
        if metric != "overlap_ratio":
            raise ValueError("This implementation supports only Prune-OPD metric=overlap_ratio")
        if not 0.0 < threshold <= 1.0:
            raise ValueError("Prune-OPD threshold must be in (0, 1]")
        if w_drop < 0 or w_base < 0:
            raise ValueError("Prune-OPD w_drop and w_base must be non-negative")
        rollout_cfg["log_prob_top_k"] = prune_top_k
        rollout_cfg["prune_opd"] = {
            "enable": True,
            "metric": metric,
            "threshold": threshold,
            "w_drop": w_drop,
            "w_base": w_base,
        }
    elif method == "poweropd":
        alpha = float(params.get("alpha", 5.0))
        if alpha <= 0:
            raise ValueError("PowerOPD requires distillation.params.alpha to be positive")
        rollout_cfg["poweropd_reward_alpha"] = alpha
        # PowerOPD only needs the sampled token's teacher/student log-probs.
        rollout_cfg["log_prob_top_k"] = 0

    config.setdefault("distillation", {})["method"] = method
    config["distillation"]["params"] = dict(params)
    return method


def build_effective_config(repo_root: Path, config_path: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    config = expand(load_yaml_with_extends(config_path))
    resume_requested = "resume" in config
    env_cfg = config["environment"]
    data_cfg = config["data"]
    model_cfg = config["model"]
    length_cfg = config["length"]
    rollout_cfg = config["rollout"]
    train_cfg = config["training"]
    system_cfg = config["system"]
    resolve_distillation_config(config)

    resume_cfg = config.get("resume", {})
    resume_mode = str(resume_cfg.get("mode", "auto"))
    resume_from_path = resume_cfg.get("path")
    if resume_mode not in {"auto", "disable", "resume_path"}:
        raise ValueError(
            f"Unsupported resume.mode={resume_mode!r}; "
            "expected auto, disable, or resume_path"
        )
    if resume_mode == "resume_path":
        if not resume_from_path:
            raise ValueError("resume.path is required when resume.mode=resume_path")
        resume_path = repo_path(repo_root, str(resume_from_path))
        if not resume_path.is_dir():
            raise FileNotFoundError(f"resume checkpoint does not exist: {resume_path}")
        if "global_step_" not in resume_path.name:
            raise ValueError(
                f"resume checkpoint must be a global_step_* directory: {resume_path}"
            )
        resume_cfg["path"] = str(resume_path)
    config["resume"] = {
        "mode": resume_mode,
        "path": resume_cfg.get("path"),
    }
    config["_resume_requested"] = resume_requested

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
    distillation_method = rollout_cfg.get("distillation_method", "vanilla")

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
        hydra_arg("actor_rollout_ref.rollout.distillation_method", distillation_method),
        hydra_arg(
            "actor_rollout_ref.rollout.poweropd_reward_alpha",
            rollout_cfg.get("poweropd_reward_alpha", 5.0),
        ),
        hydra_arg("actor_rollout_ref.rollout.eopd_enabled", rollout_cfg.get("eopd_enabled", False)),
        hydra_arg("actor_rollout_ref.rollout.eopd_top_k", rollout_cfg.get("eopd_top_k", 16)),
        hydra_arg("actor_rollout_ref.rollout.aopd_threshold", rollout_cfg.get("aopd_threshold", 0.1)),
        hydra_arg("actor_rollout_ref.rollout.aopd_opd_weight", rollout_cfg.get("aopd_opd_weight", 1.0)),
        hydra_arg("actor_rollout_ref.rollout.aopd_gkd_weight", rollout_cfg.get("aopd_gkd_weight", 1.0)),
        hydra_arg("actor_rollout_ref.rollout.aopd_jsd_beta", rollout_cfg.get("aopd_jsd_beta", 1.0)),
        hydra_arg(
            "actor_rollout_ref.rollout.prune_opd.enable",
            rollout_cfg.get("prune_opd", {}).get("enable", False),
        ),
        hydra_arg(
            "actor_rollout_ref.rollout.prune_opd.metric",
            rollout_cfg.get("prune_opd", {}).get("metric", "overlap_ratio"),
        ),
        hydra_arg(
            "actor_rollout_ref.rollout.prune_opd.threshold",
            rollout_cfg.get("prune_opd", {}).get("threshold", 0.7),
        ),
        hydra_arg(
            "actor_rollout_ref.rollout.prune_opd.w_drop",
            rollout_cfg.get("prune_opd", {}).get("w_drop", 0.01),
        ),
        hydra_arg(
            "actor_rollout_ref.rollout.prune_opd.w_base",
            rollout_cfg.get("prune_opd", {}).get("w_base", 0.5),
        ),
        hydra_arg(
            "actor_rollout_ref.actor.policy_loss.loss_mode",
            distillation_method if distillation_method in {"eopd", "aopd"} else "vanilla",
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
            "actor_rollout_ref.actor.policy_loss.aopd_threshold",
            rollout_cfg.get("aopd_threshold", 0.1),
        ),
        hydra_arg(
            "actor_rollout_ref.actor.policy_loss.aopd_gkd_weight",
            rollout_cfg.get("aopd_gkd_weight", 1.0),
        ),
        hydra_arg(
            "actor_rollout_ref.actor.policy_loss.aopd_jsd_beta",
            rollout_cfg.get("aopd_jsd_beta", 1.0),
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
        hydra_arg("trainer.resume_mode", config["resume"]["mode"]),
    ]
    if config["resume"]["path"] is not None:
        args.append(hydra_arg("trainer.resume_from_path", config["resume"]["path"]))
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
    print(f"  distillation_method={rollout_cfg.get('distillation_method', 'vanilla')}")
    if rollout_cfg.get("distillation_method") == "poweropd":
        print(f"  poweropd_reward_alpha={rollout_cfg['poweropd_reward_alpha']}")
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

    if paths["ckpt_path"].exists() and not config["_resume_requested"]:
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
