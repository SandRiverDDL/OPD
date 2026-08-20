#!/usr/bin/env bash
set -euo pipefail

# One-click setup for the OPD torch-base environment.
#
# This script intentionally does not start Ray, launch training, install
# project-specific worker services, or download models. It installs the local
# OPD/verl dependencies, verifies the cached models and data, and applies only
# the tcodex context-window patch required by this project.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

DRY_RUN=0
SKIP_INSTALL=0
SKIP_TCODEX=0

usage() {
    cat <<'EOF'
Usage:
  bash scripts/setup_opd_env.sh [options]

Options:
  --dry-run       Print planned changes without installing or patching files.
  --skip-install  Skip Python package installation and only patch/verify.
  --skip-tcodex   Skip the tcodex context-window patch.
  -h, --help      Show this help.

Environment overrides:
  PYTHON                  Python executable; default: python
  HF_HOME                Hugging Face cache root
  HF_HUB_CACHE           Hugging Face Hub cache directory
  OPD_SETUP_LOG_DIR      Log directory; default: <repo>/logs/env_setup
EOF
}

while (($# > 0)); do
    case "$1" in
        --dry-run) DRY_RUN=1 ;;
        --skip-install) SKIP_INSTALL=1 ;;
        --skip-tcodex) SKIP_TCODEX=1 ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "error: unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
    shift
done

PYTHON="${PYTHON:-python}"
command -v "$PYTHON" >/dev/null 2>&1 || {
    echo "error: Python executable not found: $PYTHON" >&2
    exit 1
}

LOG_DIR="${OPD_SETUP_LOG_DIR:-$PROJECT_ROOT/logs/env_setup}"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/setup_$(date +%Y%m%d_%H%M%S).log"

# Keep a persistent log while still showing progress in the terminal.
exec > >(tee -a "$LOG_FILE") 2>&1

echo "OPD environment setup"
echo "project: $PROJECT_ROOT"
echo "python:  $("$PYTHON" -c 'import sys; print(sys.executable)')"
echo "log:     $LOG_FILE"

if [[ -z "${HF_HOME:-}" ]]; then
    # Prefer the cache used by the current TAIJI workspace, then fall back to
    # the conventional repository-relative and user-local locations.
    HF_CANDIDATES=()
    if [[ -n "${TAIJI_IDE_DEV_DIR:-}" ]]; then
        HF_CANDIDATES+=("${TAIJI_IDE_DEV_DIR%/dev}/.cache/huggingface")
        if [[ "${TAIJI_IDE_DEV_DIR}" == */hunyuan/*/dev ]]; then
            TAIJI_ROOT="${TAIJI_IDE_DEV_DIR%%/hunyuan/*}"
            TAIJI_USER="$(basename "${TAIJI_IDE_DEV_DIR%/dev}")"
            HF_CANDIDATES+=("$TAIJI_ROOT/$TAIJI_USER/.cache/huggingface")
        fi
    fi
    HF_CANDIDATES+=(
        "${PROJECT_ROOT%/dev/lab/OPD}/.cache/huggingface"
        "$HOME/.cache/huggingface"
    )
    for HF_CANDIDATE in "${HF_CANDIDATES[@]}"; do
        if [[ -d "$HF_CANDIDATE/hub" ]]; then
            export HF_HOME="$HF_CANDIDATE"
            break
        fi
    done
    export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
fi
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"

echo "HF_HOME:      $HF_HOME"
echo "HF_HUB_CACHE: $HF_HUB_CACHE"

run_cmd() {
    printf '+'
    printf ' %q' "$@"
    printf '\n'
    if (( ! DRY_RUN )); then
        "$@"
    fi
}

if (( ! SKIP_INSTALL )); then
    echo
    echo "== Install local verl package =="
    run_cmd "$PYTHON" -m pip install -e "$PROJECT_ROOT/verl" --no-deps

    echo
    echo "== Install OPD runtime dependencies =="
    run_cmd "$PYTHON" -m pip install --no-cache-dir \
        'tensordict>=0.8.0,<=0.10.0,!=0.9.0' \
        math-verify \
        latex2sympy2-extended
else
    echo
    echo "== Skip Python package installation =="
fi

resolve_hf_snapshot() {
    local repo_kind="$1"
    local repo_id="$2"
    local repo_dir="$HF_HUB_CACHE/${repo_kind}--${repo_id//\//--}"
    local revision

    [[ -f "$repo_dir/refs/main" ]] || return 1
    revision="$(cat "$repo_dir/refs/main")"
    [[ -d "$repo_dir/snapshots/$revision" ]] || return 1
    printf '%s\n' "$repo_dir/snapshots/$revision"
}

STUDENT_MODEL_PATH="${HF_STUDENT_MODEL_PATH:-$(resolve_hf_snapshot models deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B || true)}"
TEACHER_MODEL_PATH="${HF_TEACHER_MODEL_PATH:-$(resolve_hf_snapshot models hbx/JustRL-DeepSeek-1.5B || true)}"
DAPO_SNAPSHOT_PATH="${HF_DAPO_SNAPSHOT_PATH:-$(resolve_hf_snapshot datasets BytedTsinghua-SIA/DAPO-Math-17k || true)}"
DAPO_DATASET_PATH="${HF_DAPO_DATASET_PATH:-${DAPO_SNAPSHOT_PATH:+$DAPO_SNAPSHOT_PATH/data/dapo-math-17k.parquet}}"

echo
echo "== Verify cached OPD assets =="
for required_path in \
    "$STUDENT_MODEL_PATH/config.json" \
    "$STUDENT_MODEL_PATH/tokenizer_config.json" \
    "$STUDENT_MODEL_PATH/model.safetensors" \
    "$TEACHER_MODEL_PATH/config.json" \
    "$TEACHER_MODEL_PATH/tokenizer_config.json" \
    "$TEACHER_MODEL_PATH/model.safetensors" \
    "$DAPO_DATASET_PATH"; do
    if [[ ! -f "$required_path" ]]; then
        echo "error: required cached asset is missing: $required_path" >&2
        exit 1
    fi
done
echo "student: $STUDENT_MODEL_PATH"
echo "teacher: $TEACHER_MODEL_PATH"
echo "dataset: $DAPO_DATASET_PATH"

for validation_file in \
    "$PROJECT_ROOT/datasets/test_data/AIME24/test.parquet" \
    "$PROJECT_ROOT/datasets/test_data/AIME25/test.parquet" \
    "$PROJECT_ROOT/datasets/test_data/AMC23/test.parquet"; do
    [[ -f "$validation_file" ]] || {
        echo "error: validation file is missing: $validation_file" >&2
        exit 1
    }
done

if (( ! SKIP_TCODEX )); then
    echo
    echo "== Configure tcodex context window =="
    if ! command -v tcodex >/dev/null 2>&1; then
        echo "error: tcodex is not on PATH; use --skip-tcodex to omit this step" >&2
        exit 1
    fi

    TCODEX_BIN="$(readlink -f "$(command -v tcodex)")"
    TCODEX_ROOT="$(cd -- "$(dirname -- "$TCODEX_BIN")/.." && pwd)"
    PRODUCT_JSON="$TCODEX_ROOT/product.json"
    TCODEX_DIST="$TCODEX_ROOT/dist/tcodex.js"
    SKILL_CREATOR_METADATA=""
    for candidate in \
        "${TCODEX_HOME:-}/skills/.system/skill-creator/agents/openai.yaml" \
        "${CODEX_HOME:-}/skills/.system/skill-creator/agents/openai.yaml" \
        "$HOME/.tcodex/skills/.system/skill-creator/agents/openai.yaml" \
        "$HOME/.codex/skills/.system/skill-creator/agents/openai.yaml"; do
        if [[ -n "$candidate" && -f "$candidate" ]]; then
            SKILL_CREATOR_METADATA="$candidate"
            break
        fi
    done

    [[ -f "$PRODUCT_JSON" ]] || {
        echo "error: tcodex product catalog not found: $PRODUCT_JSON" >&2
        exit 1
    }
    [[ -f "$TCODEX_DIST" ]] || {
        echo "error: tcodex distribution file not found: $TCODEX_DIST" >&2
        exit 1
    }
    [[ -n "$SKILL_CREATOR_METADATA" ]] || {
        echo "error: skill-creator metadata not found under tcodex/codex home" >&2
        exit 1
    }

    if (( DRY_RUN )); then
        echo "would cap gpt-5.6-* maxInputTokens at 272000 in $PRODUCT_JSON"
        echo "would patch model catalog generation in $TCODEX_DIST"
        echo "would disable skill-creator implicit invocation in $SKILL_CREATOR_METADATA"
    else
        "$PYTHON" - "$PRODUCT_JSON" "$TCODEX_DIST" "$SKILL_CREATOR_METADATA" <<'PY'
from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

INPUT_TOKEN_CAP = 272_000
DIST_OLD = "eg=eu.maxInputTokens;return{slug:ed"
DIST_MARKER = 'eu.id.startsWith("gpt-5.6-")?Math.min(eu.maxInputTokens,272000):'
DIST_NEW = (
    'eg=eu.id.startsWith("gpt-5.6-")?Math.min(eu.maxInputTokens,272000):'
    "eu.maxInputTokens;eu.maxInputTokens=eg;return{slug:ed"
)


def backup_once(path: Path) -> None:
    backup = path.with_name(path.name + ".bak-opd-set-up")
    if not backup.exists():
        shutil.copy2(path, backup)


product_path = Path(sys.argv[1])
dist_path = Path(sys.argv[2])
skill_creator_path = Path(sys.argv[3])

product = json.loads(product_path.read_text())
changed_models = []
for model in product.get("models", []):
    model_id = str(model.get("id", ""))
    if model_id.startswith("gpt-5.6-") and model.get("maxInputTokens") != INPUT_TOKEN_CAP:
        model["maxInputTokens"] = INPUT_TOKEN_CAP
        changed_models.append(model_id)

if changed_models:
    backup_once(product_path)
    product_path.write_text(json.dumps(product, ensure_ascii=False, indent=2) + "\n")

source = dist_path.read_text()
if DIST_MARKER in source:
    catalog_status = "already patched"
elif source.count(DIST_OLD) == 1:
    backup_once(dist_path)
    dist_path.write_text(source.replace(DIST_OLD, DIST_NEW, 1))
    catalog_status = "patched"
else:
    raise RuntimeError(
        "unsupported tcodex version: model catalog pattern was not found exactly once"
    )

old_text = skill_creator_path.read_text()
new_text = re.sub(
    r"(?m)^(\s*allow_implicit_invocation:\s*)(true|false)\s*$",
    r"\1false",
    old_text,
)
if new_text == old_text and not re.search(
    r"(?m)^\s*allow_implicit_invocation:\s*(true|false)\s*$", old_text
):
    new_text = old_text.rstrip("\n") + "\npolicy:\n  allow_implicit_invocation: false\n"
if new_text != old_text:
    backup_once(skill_creator_path)
    skill_creator_path.write_text(new_text)
    skill_creator_status = "patched"
else:
    skill_creator_status = "already disabled"

print("product catalog:", ", ".join(changed_models) if changed_models else "already capped")
print("catalog generator:", catalog_status)
print("skill-creator implicit invocation:", skill_creator_status)
print("input token cap:", INPUT_TOKEN_CAP)
print("effective tcodex window after its 95% factor: about 258400")
PY
    fi
else
    echo
    echo "== Skip tcodex configuration =="
fi

echo
echo "== Verify Python runtime =="
if (( DRY_RUN )); then
    echo "would import torch, verl, vllm, ray, tensordict and math reward dependencies"
else
    PYTHONPATH="$PROJECT_ROOT/verl${PYTHONPATH:+:$PYTHONPATH}" \
        HF_HOME="$HF_HOME" \
        TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}" \
        "$PYTHON" - <<'PY'
from importlib import import_module, metadata

import torch

required_distributions = [
    "verl",
    "tensordict",
    "math-verify",
    "latex2sympy2-extended",
    "vllm",
    "ray",
]
for distribution in required_distributions:
    print(f"{distribution}: {metadata.version(distribution)}")

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available")
print("CUDA devices:", torch.cuda.device_count())

for module_name in [
    "verl.trainer.main_ppo",
    "verl.utils.reward_score.ttrl_math",
    "vllm",
    "ray",
    "tensordict",
    "math_verify",
    "latex2sympy2_extended",
]:
    import_module(module_name)
    print("import OK:", module_name)

from verl.utils.reward_score.ttrl_math import reward_func

result = reward_func("math", r"The answer is \boxed{2}", "2")
if not isinstance(result, dict) or not result.get("acc"):
    raise RuntimeError(f"math reward smoke test failed: {result!r}")
print("math reward smoke test: OK")
PY
fi

if [[ -f "$PROJECT_ROOT/on_policy_distillation.sh" ]]; then
    echo
    echo "== Verify OPD launcher path resolution =="
    if (( DRY_RUN )); then
        echo "would run: DRY_RUN=True bash on_policy_distillation.sh"
    else
        HF_HOME="$HF_HOME" \
            HF_HUB_CACHE="$HF_HUB_CACHE" \
            DRY_RUN=True \
            HF_STUDENT_MODEL_PATH="$STUDENT_MODEL_PATH" \
            HF_TEACHER_MODEL_PATH="$TEACHER_MODEL_PATH" \
            HF_DAPO_DATASET_PATH="$DAPO_DATASET_PATH" \
            bash "$PROJECT_ROOT/on_policy_distillation.sh"
    fi
fi

echo
echo "Setup checks completed."
echo "Restart tcodex, then verify its models.json reports:"
echo "  context_window: 272000"
echo "  auto_compact_token_limit: 244800"
echo "No Ray job or training task was started."
