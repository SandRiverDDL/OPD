import os
import json
import re
import argparse
import concurrent.futures
import multiprocessing  # Added for spawn-based worker management
import gc  # Added for explicit resource cleanup
import time
import copy
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm
import yaml

# --------------------------------------------------------------------------- #
#                   Global constants / variables                              #
# --------------------------------------------------------------------------- #
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_DIR = REPO_ROOT / "scripts" / "val" / "data"
PROMPT_TEMPLATE = """{problem} Please reason step by step, and put your final answer within \\boxed{{}}."""
PROMPT_SUFFIX = r" Please reason step by step, and put your final answer within \boxed{}."

# These globals are populated by configure_evaluation() before worker
# processes are created. Keeping them global avoids passing the complete
# evaluation configuration through every multiprocessing task.
MODEL_NAMES: list[str] = []
MODEL_DISPLAY_NAMES: dict[str, str] = {}
EVAL_TASK = ""
EVAL_TASK_N = 1
TASKS: list[dict[str, Any]] = []
MAX_TOKENS = 31744
EVAL_MAX_MODEL_LEN = MAX_TOKENS + 2048
TEMPERATURE = 0.7
TOP_P = 0.95
EVAL_SEED: int | None = None
EVAL_MAX_NUM_SEQS = 32
EVAL_MAX_NUM_BATCHED_TOKENS = 32768
EVAL_GPU_MEMORY_UTILIZATION = 0.90
AVAILABLE_GPUS: list[int] = list(range(8))
REPLACE = False
VLLM_PORT_BASE: int | None = None


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
    if not isinstance(raw, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    parent = raw.pop("extends", None)
    if parent is None:
        return raw
    parent_path = (path.parent / str(parent)).resolve()
    if not parent_path.is_file():
        raise FileNotFoundError(f"YAML parent does not exist: {parent_path}")
    return deep_merge(load_yaml_with_extends(parent_path), raw)


def resolve_config_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    candidates = [(Path.cwd() / path).resolve(), (REPO_ROOT / path).resolve()]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def parse_gpu_ids(value: Any) -> list[int]:
    if isinstance(value, str):
        values = value.split(",")
    elif isinstance(value, (list, tuple)):
        values = value
    else:
        raise ValueError("evaluation.gpus/EVAL_GPUS must be a list or comma-separated string")

    gpu_ids = []
    for raw_value in values:
        try:
            gpu_id = int(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid GPU ID: {raw_value!r}") from exc
        if gpu_id < 0:
            raise ValueError(f"GPU ID must be non-negative: {gpu_id}")
        gpu_ids.append(gpu_id)
    if not gpu_ids:
        raise ValueError("At least one evaluation GPU is required")
    return gpu_ids


def parse_bool(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"{name} must be a boolean, got {value!r}")


def resolve_data_path(value: str | Path, config_path: Path | None) -> str:
    path = Path(value).expanduser()
    if path.is_absolute():
        return str(path)
    if config_path is not None:
        return str((REPO_ROOT / path).resolve())
    return str(path)


def configure_evaluation(args: argparse.Namespace) -> None:
    """Resolve YAML, legacy environment variables, and CLI overrides."""
    global AVAILABLE_GPUS
    global EVAL_GPU_MEMORY_UTILIZATION, EVAL_MAX_MODEL_LEN
    global EVAL_MAX_NUM_BATCHED_TOKENS
    global EVAL_MAX_NUM_SEQS, EVAL_SEED, EVAL_TASK, EVAL_TASK_N
    global MODEL_DISPLAY_NAMES, MODEL_NAMES, MAX_TOKENS, REPLACE
    global TASKS, TEMPERATURE, TOP_P, VLLM_PORT_BASE

    config_path_value = args.config or os.environ.get("EVAL_CONFIG")
    config_path = resolve_config_path(config_path_value) if config_path_value else None
    evaluation_cfg: dict[str, Any] = {}
    if config_path is not None:
        if not config_path.is_file():
            raise FileNotFoundError(f"Evaluation config does not exist: {config_path}")
        config = load_yaml_with_extends(config_path)
        evaluation_cfg = config.get("evaluation", {}) or {}
        if not isinstance(evaluation_cfg, dict):
            raise ValueError(f"evaluation must be a mapping in {config_path}")

    def value(
        key: str,
        env_name: str | None,
        default: Any,
        cli_value: Any = None,
    ) -> Any:
        if cli_value is not None:
            return cli_value
        if key in evaluation_cfg:
            return evaluation_cfg[key]
        if env_name is not None and env_name in os.environ:
            return os.environ[env_name]
        return default

    model_path_from_cli_or_config = (
        args.model_path is not None or "model_path" in evaluation_cfg
    )
    model_path = value(
        "model_path",
        "EVAL_MODEL_PATH",
        "../../model/Qwen3-4B",
        args.model_path,
    )
    if model_path_from_cli_or_config and not Path(str(model_path)).is_absolute():
        model_path = str((REPO_ROOT / Path(str(model_path))).resolve())
    model_name = value(
        "model_name",
        "EVAL_MODEL_NAME",
        Path(str(model_path)).name,
        args.model_name,
    )
    MODEL_NAMES = [str(model_path)]
    MODEL_DISPLAY_NAMES = {str(model_path): str(model_name)}

    EVAL_TASK = str(value("task", "EVAL_TASK", "MATH-500", args.task))
    EVAL_TASK_N = int(value("n", "EVAL_N", 1, args.num_samples))
    if EVAL_TASK_N <= 0:
        raise ValueError("evaluation.n/EVAL_N must be positive")

    configured_data_path = value("data_path", "EVAL_DATA_PATH", None, args.data_path)
    if configured_data_path is None:
        configured_data_path = DEFAULT_DATA_DIR / EVAL_TASK / "test.parquet"
    EVAL_DATA_PATH = resolve_data_path(configured_data_path, config_path)
    TASKS = [{"name": EVAL_TASK, "path": EVAL_DATA_PATH, "N": EVAL_TASK_N}]

    MAX_TOKENS = int(value("max_tokens", "EVAL_MAX_TOKENS", 31744, args.max_tokens))
    max_model_len = value(
        "max_model_len",
        "EVAL_MAX_MODEL_LEN",
        MAX_TOKENS + 2048,
        args.max_model_len,
    )
    EVAL_MAX_MODEL_LEN = int(max_model_len)

    TEMPERATURE = float(value("temperature", "EVAL_TEMPERATURE", 0.7, args.temperature))
    TOP_P = float(value("top_p", "EVAL_TOP_P", 0.95, args.top_p))
    seed_value = value("seed", "EVAL_SEED", None, args.seed)
    EVAL_SEED = None if seed_value in {None, "", "null", "None"} else int(seed_value)

    legacy_batch_size = int(os.environ.get("EVAL_BATCH_SIZE", "32"))
    EVAL_MAX_NUM_SEQS = int(
        value(
            "max_num_seqs",
            "EVAL_MAX_NUM_SEQS",
            legacy_batch_size * EVAL_TASK_N,
            args.max_num_seqs,
        )
    )
    EVAL_MAX_NUM_BATCHED_TOKENS = int(
        value(
            "max_num_batched_tokens",
            "EVAL_MAX_NUM_BATCHED_TOKENS",
            32768,
            args.max_num_batched_tokens,
        )
    )
    EVAL_GPU_MEMORY_UTILIZATION = float(
        value(
            "gpu_memory_utilization",
            "EVAL_GPU_MEMORY_UTILIZATION",
            0.90,
            args.gpu_memory_utilization,
        )
    )
    AVAILABLE_GPUS = parse_gpu_ids(
        value("gpus", "EVAL_GPUS", list(range(8)), args.gpus)
    )
    requested_port_base = os.environ.get("EVAL_VLLM_PORT_BASE")
    if requested_port_base is not None:
        VLLM_PORT_BASE = int(requested_port_base)
    else:
        # Reserve one disjoint base port per worker.  vLLM's default
        # find-then-bind selection can race when multiple TP=1 workers
        # initialize simultaneously.
        VLLM_PORT_BASE = 20000 + (os.getpid() % 1000) * 32
    REPLACE = parse_bool(
        value("replace", "EVAL_REPLACE", False, args.replace),
        "evaluation.replace",
    )

    if EVAL_MAX_NUM_SEQS <= 0 or EVAL_MAX_NUM_BATCHED_TOKENS <= 0:
        raise ValueError("max_num_seqs and max_num_batched_tokens must be positive")
    if not 0 < EVAL_GPU_MEMORY_UTILIZATION <= 1:
        raise ValueError("gpu_memory_utilization must be in (0, 1]")

    enable_thinking = value(
        "enable_thinking",
        "EVAL_ENABLE_THINKING",
        False,
        args.enable_thinking,
    )
    args.enable_thinking = parse_bool(enable_thinking, "evaluation.enable_thinking")

# --------------------------------------------------------------------------- #
#                               Helper functions                              #
# --------------------------------------------------------------------------- #
def load_samples(filepath: str):
    """Read parquet file and return a list of prompts (no duplication)."""
    df = pd.read_parquet(filepath)
    if "BRUMO25" in filepath or "CMIMC25" in filepath or "HMMT25" in filepath:
        samples = [
            {
                "example_id": i,
                "prompt": df.at[i, "problem"].strip(),
                "answer": df.at[i, "answer"].strip(),
            }
            for i in range(len(df))
        ]
    else:
        samples = [
            {
                "example_id": i,
                "prompt": df.at[i, "prompt"][0]["content"].strip(),
                "answer": df.at[i, "reward_model"]["ground_truth"].strip(),
            }
            for i in range(len(df))
        ]
    print(f"Total unique samples: {len(samples)}")
    return samples


def split_samples(samples: list[dict], num_workers: int) -> list[list[dict]]:
    """Split samples into contiguous, near-equal data-parallel shards."""
    num_workers = min(num_workers, len(samples)) if samples else num_workers
    floor, remainder = divmod(len(samples), num_workers)
    shards = []
    start = 0
    for rank in range(num_workers):
        size = floor + (rank < remainder)
        shards.append(samples[start : start + size])
        start += size
    return shards


# --------------------------------------------------------------------------- #
#              Worker process (one model instance per GPU worker)              #
# --------------------------------------------------------------------------- #
def worker_process(args_tuple):
    """
    Each worker runs on a single GPU:
    args_tuple = (
        model_name,
        samples,
        gpu_id,
        enable_thinking,
        num_samples,
        worker_config,
    )
    gpu_id: values such as "0" or "3", used for CUDA_VISIBLE_DEVICES
    """
    (
        model_name,
        samples,
        gpu_id,
        enable_thinking,
        num_samples,
        worker_config,
    ) = args_tuple
    max_tokens = worker_config["max_tokens"]
    max_model_len = worker_config["max_model_len"]
    temperature = worker_config["temperature"]
    top_p = worker_config["top_p"]
    seed = worker_config["seed"]
    max_num_seqs = worker_config["max_num_seqs"]
    max_num_batched_tokens = worker_config["max_num_batched_tokens"]
    gpu_memory_utilization = worker_config["gpu_memory_utilization"]
    vllm_port_base = worker_config["vllm_port_base"]
    worker_index = worker_config["worker_index"]
    
    # CUDA_VISIBLE_DEVICES must be set inside the spawned process.
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    # Leave a gap between workers because vLLM may probe subsequent ports
    # when its requested port is already occupied.  Consecutive assignments
    # let one worker's fallback collide with the next worker's reserved port.
    os.environ["VLLM_PORT"] = str(vllm_port_base + worker_index * 16)
    # Import CUDA/vLLM only after constraining visibility to this worker.
    import torch
    from vllm import LLM, SamplingParams
    try:
        from vllm.distributed.parallel_state import destroy_model_parallel
    except ImportError:
        destroy_model_parallel = None
    try:
        from vllm.distributed.parallel_state import destroy_distributed_environment
    except ImportError:
        destroy_distributed_environment = None
    
    results = []
    llm = None
    stop_token_ids = []
    worker_start = time.perf_counter()
    generation_seconds = 0.0
    input_tokens = 0
    output_tokens = 0
    generated_sequences = 0
    
    try:
        print(
            f"[GPU {gpu_id}] | Model: {model_name} | samples={len(samples)} "
            f"| loading model (TP=1, max_num_seqs={max_num_seqs}, "
            f"max_num_batched_tokens={max_num_batched_tokens}, "
            f"enable_thinking={enable_thinking})...",
            flush=True,
        )
        
        # Initialize a single-GPU, single-instance LLM.
        llm = LLM(
            model=model_name,
            trust_remote_code=True,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=1,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
        )
        
        # Get the tokenizer.
        try:
            tokenizer = llm.get_tokenizer()
            
            # Encode stop tokens.
            for stop_token in ["<|im_end|>", "<|endoftext|>"]:
                try:
                    if hasattr(tokenizer, "encode"):
                        encoded = tokenizer.encode(stop_token, add_special_tokens=False)
                        if encoded:
                            stop_token_ids.append(encoded[0])
                except Exception:
                    pass
        except Exception as e:
            tokenizer = None
            print(f"[GPU {gpu_id}] Warning: Could not get tokenizer for stop tokens: {e}", flush=True)
        
        sampling = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            n=num_samples,
            seed=seed,
            stop_token_ids=stop_token_ids if stop_token_ids else None,
        )

        if tokenizer is None:
            raise RuntimeError("Tokenizer is required for apply_chat_template, but it could not be loaded.")

        formatted_prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": s["prompt"]}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
            for s in samples
        ]
        generation_start = time.perf_counter()
        outputs = llm.generate(formatted_prompts, sampling, use_tqdm=False)
        generation_seconds = time.perf_counter() - generation_start

        for sample, out in zip(samples, outputs):
            prompt_token_ids = getattr(out, "prompt_token_ids", None)
            if prompt_token_ids is not None:
                input_tokens += len(prompt_token_ids)

            for sample_output in out.outputs[:num_samples]:
                token_ids = getattr(sample_output, "token_ids", None)
                if token_ids is not None:
                    output_tokens += len(token_ids)
                generated_sequences += 1
                results.append(
                    {
                        "example_id": sample["example_id"],
                        "prompt": sample["prompt"],
                        "answer": sample["answer"],
                        "seed": len(results),
                        "response": sample_output.text,
                    }
                )

        elapsed = time.perf_counter() - worker_start
        total_tokens = input_tokens + output_tokens
        generation_tps = (
            total_tokens / generation_seconds if generation_seconds > 0 else 0.0
        )
        print(
            f"[GPU {gpu_id}] completed samples={len(samples)} "
            f"sequences={generated_sequences} "
            f"input_tokens={input_tokens} output_tokens={output_tokens} "
            f"generation_seconds={generation_seconds:.2f} "
            f"generation_tokens_per_second={generation_tps:.2f} "
            f"worker_elapsed_seconds={elapsed:.2f}",
            flush=True,
        )

    except Exception as e:
        print(f"[GPU {gpu_id}] Critical Error: {e}", flush=True)
        raise
    
    finally:
        # Explicitly release vLLM resources.
        # This helps prevent CUDA context deadlocks and zombie processes.
        print(f"[GPU {gpu_id}] Cleaning up resources...", flush=True)
        if llm is not None:
            del llm
        
        if destroy_model_parallel is not None:
            try:
                destroy_model_parallel()
            except Exception:
                pass

        if destroy_distributed_environment is not None:
            try:
                destroy_distributed_environment()
            except Exception:
                pass
        
        gc.collect()
        torch.cuda.empty_cache()
        print(f"[GPU {gpu_id}] Cleanup done.", flush=True)

    return results


# --------------------------------------------------------------------------- #
#                                   main                                      #
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="Generate evaluation rollouts with vLLM.")
    parser.add_argument(
        "--config",
        help="OPD YAML config containing an evaluation section.",
    )
    parser.add_argument("--model-path", help="Model/checkpoint path to evaluate.")
    parser.add_argument("--model-name", help="Display name for the output directory.")
    parser.add_argument("--task", help="Evaluation task name.")
    parser.add_argument("--data-path", help="Evaluation parquet path.")
    parser.add_argument("--n", "--num-samples", dest="num_samples", type=int)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--max-model-len", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--gpus", help="Comma-separated GPU IDs, for example 0,1,2,3.")
    parser.add_argument("--max-num-seqs", type=int)
    parser.add_argument("--max-num-batched-tokens", type=int)
    parser.add_argument("--gpu-memory-utilization", type=float)
    parser.add_argument(
        "--replace",
        dest="replace",
        action="store_true",
        default=None,
        help="Overwrite an existing result file.",
    )
    thinking_group = parser.add_mutually_exclusive_group()
    thinking_group.add_argument(
        "--enable-thinking",
        dest="enable_thinking",
        action="store_true",
        help="Enable thinking when applying the chat template.",
    )
    thinking_group.add_argument(
        "--disable-thinking",
        dest="enable_thinking",
        action="store_false",
        help="Disable thinking when applying the chat template.",
    )
    parser.set_defaults(enable_thinking=None)
    args = parser.parse_args()
    configure_evaluation(args)

    # One independent TP=1 vLLM replica per GPU.
    available_gpus = AVAILABLE_GPUS
    gpu_workers = [str(gpu_id) for gpu_id in available_gpus]
    num_workers = len(gpu_workers)

    print(f"GPU workers (one model per GPU): {gpu_workers}")
    print(f"apply_chat_template enable_thinking={args.enable_thinking}")
    print(
        f"max_num_seqs={EVAL_MAX_NUM_SEQS}, "
        f"max_num_batched_tokens={EVAL_MAX_NUM_BATCHED_TOKENS}, "
        f"gpu_memory_utilization={EVAL_GPU_MEMORY_UTILIZATION}, "
        f"max_model_len={EVAL_MAX_MODEL_LEN}"
    )

    for model_name in MODEL_NAMES:
        print(f"\n{'='*50}\nStarting evaluation for model: {model_name}\n{'='*50}")
        
        out_name = MODEL_DISPLAY_NAMES.get(model_name, Path(model_name).name)
        OUT_DIR = REPO_ROOT / "scripts" / "val" / "eval" / "justrl_eval_outputs" / out_name
        OUT_DIR.mkdir(parents=True, exist_ok=True)

        for task in TASKS:
            task_name = task["name"]
            task_path = task["path"]
            N = task["N"]

            print(f"Starting evaluation for task: {task_name} (N={N})")
            
            out_path = OUT_DIR / f"{task_name.lower()}_t{TEMPERATURE}_p{TOP_P}_n{N}-MNT{MAX_TOKENS}.jsonl"

            # --- Repetition Check ---
            if not REPLACE and out_path.exists():
                print(f"Result file already exists at '{out_path}'. Skipping.")
                continue  # Skip to the next task

            # 1. Load original prompts
            samples = load_samples(task_path)

            # Append suffix prompt to each sample
            for sample in samples:
                # Holdout and benchmark parquet files may already contain the
                # instruction suffix. Do not append it a second time.
                prompt = sample["prompt"].rstrip()
                if not prompt.endswith(PROMPT_SUFFIX):
                    prompt = PROMPT_TEMPLATE.format(problem=prompt)
                sample["prompt"] = prompt

            if len(samples) > 0:
                print("Example prompt after formatting:")
                print(samples[0]["prompt"])
            
            # 2. Shard prompts across independent vLLM replicas.
            sample_shards = split_samples(samples, num_workers)

            # 3. Launch workers, with each worker using one GPU.
            all_results = []
            args_list = [
                (
                    model_name,
                    sample_shards[i],
                    gpu_workers[i],
                    args.enable_thinking,
                    N,
                    {
                        "max_tokens": MAX_TOKENS,
                        "max_model_len": EVAL_MAX_MODEL_LEN,
                        "temperature": TEMPERATURE,
                        "top_p": TOP_P,
                        "seed": EVAL_SEED,
                        "max_num_seqs": EVAL_MAX_NUM_SEQS,
                        "max_num_batched_tokens": EVAL_MAX_NUM_BATCHED_TOKENS,
                        "gpu_memory_utilization": EVAL_GPU_MEMORY_UTILIZATION,
                        "vllm_port_base": VLLM_PORT_BASE,
                        "worker_index": i,
                    },
                )
                for i in range(len(sample_shards))
            ]
            
            # Use the spawn start method for worker processes.
            ctx = multiprocessing.get_context("spawn")
            
            with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as ex:
                futures = [ex.submit(worker_process, tup) for tup in args_list]
                worker_errors = []
                
                # Track overall progress with tqdm.
                for fut in tqdm(concurrent.futures.as_completed(futures),
                                total=len(futures), desc=f"GPU workers ({task_name})"):
                    try:
                        res = fut.result()
                        all_results.extend(res)
                    except Exception as e:
                        print(f"A worker process failed with error: {e}")
                        worker_errors.append(e)

            if worker_errors:
                raise RuntimeError(
                    f"{len(worker_errors)} evaluation worker(s) failed for task "
                    f"{task_name}; refusing to save an incomplete result"
                ) from worker_errors[0]

            all_results.sort(key=lambda item: (item["example_id"], item["seed"]))
            print(f"Total generations collected for {task_name}: {len(all_results)}")

            # 4. Save to disk
            if all_results:
                with out_path.open("w", encoding="utf-8") as f:
                    for item in all_results:
                        f.write(json.dumps(item, ensure_ascii=False) + "\n")
                print(f"Saved results for {task_name} to {out_path}")
            else:
                print(f"No results collected for {task_name} (Check for errors).")


if __name__ == "__main__":
    # Set the start method to avoid multiprocessing issues in some environments.
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    main()
