import os
import json
import re
import argparse
import concurrent.futures
import multiprocessing  # Added for spawn-based worker management
import gc  # Added for explicit resource cleanup
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# --------------------------------------------------------------------------- #
#                   Global constants / variables                              #
# --------------------------------------------------------------------------- #
DATA_DIR = "../data"
# Override these from the shell for a checkpoint without editing this file.
MODEL_NAMES = [os.environ.get("EVAL_MODEL_PATH", "../../model/Qwen3-4B")]
MODEL_DISPLAY_NAMES = {
    MODEL_NAMES[0]: os.environ.get("EVAL_MODEL_NAME", Path(MODEL_NAMES[0]).name)
}

EVAL_TASK = os.environ.get("EVAL_TASK", "MATH-500")
EVAL_TASK_N = int(os.environ.get("EVAL_N", "1"))
EVAL_DATA_PATH = os.environ.get("EVAL_DATA_PATH")
TASKS = [
    {
        "name": EVAL_TASK,
        "path": EVAL_DATA_PATH or f"{DATA_DIR}/{EVAL_TASK}/test.parquet",
        "N": EVAL_TASK_N,
    },
]

PROMPT_TEMPLATE = """{problem} Please reason step by step, and put your final answer within \\boxed{{}}."""
MAX_TOKENS  = int(os.environ.get("EVAL_MAX_TOKENS", "31744"))
TEMPERATURE = 0.7
TOP_P       = 0.95
EVAL_SEED = os.environ.get("EVAL_SEED")
EVAL_BATCH_SIZE = int(os.environ.get("EVAL_BATCH_SIZE", "32"))
EVAL_MAX_NUM_SEQS = int(
    os.environ.get("EVAL_MAX_NUM_SEQS", str(EVAL_BATCH_SIZE * EVAL_TASK_N))
)
AVAILABLE_GPUS = [
    int(value)
    for value in os.environ.get("EVAL_GPUS", "0,1,2,3,4,5,6,7").split(",")
    if value.strip()
]
REPLACE     = False

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
    args_tuple = (model_name, samples, gpu_id, enable_thinking, eval_batch_size, num_samples)
    gpu_id: values such as "0" or "3", used for CUDA_VISIBLE_DEVICES
    """
    model_name, samples, gpu_id, enable_thinking, eval_batch_size, num_samples = args_tuple
    
    # CUDA_VISIBLE_DEVICES must be set inside the spawned process.
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
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
    
    try:
        print(
            f"[GPU {gpu_id}] | Model: {model_name} | samples={len(samples)} "
            f"| loading model (TP=1, batch_size={eval_batch_size}, "
            f"enable_thinking={enable_thinking})...",
            flush=True,
        )
        
        # Initialize a single-GPU, single-instance LLM.
        llm = LLM(
            model=model_name,
            trust_remote_code=True,
            gpu_memory_utilization=0.9,
            tensor_parallel_size=1,
            max_model_len=int(
                os.environ.get("EVAL_MAX_MODEL_LEN", str(MAX_TOKENS + 2048))
            ),
            max_num_seqs=EVAL_MAX_NUM_SEQS,
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
            temperature=TEMPERATURE,
            top_p=TOP_P,
            max_tokens=MAX_TOKENS,
            n=num_samples,
            seed=int(EVAL_SEED) if EVAL_SEED is not None else None,
            stop_token_ids=stop_token_ids if stop_token_ids else None,
        )

        if tokenizer is None:
            raise RuntimeError("Tokenizer is required for apply_chat_template, but it could not be loaded.")

        for batch_start in range(0, len(samples), eval_batch_size):
            batch = samples[batch_start : batch_start + eval_batch_size]
            formatted_prompts = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": s["prompt"]}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=enable_thinking,
                )
                for s in batch
            ]
            outputs = llm.generate(formatted_prompts, sampling, use_tqdm=False)
            for sample, out in zip(batch, outputs):
                for sample_output in out.outputs[:num_samples]:
                    results.append(
                        {
                            "example_id": sample["example_id"],
                            "prompt": sample["prompt"],
                            "answer": sample["answer"],
                            "seed": len(results),
                            "response": sample_output.text,
                        }
                    )
    
    except Exception as e:
        print(f"[GPU {gpu_id}] Critical Error: {e}", flush=True)
        # For debugging, error details could be stored in results or logged directly.
    
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
    parser.set_defaults(enable_thinking=False)
    args = parser.parse_args()

    # One independent TP=1 vLLM replica per GPU.
    available_gpus = AVAILABLE_GPUS
    gpu_workers = [str(gpu_id) for gpu_id in available_gpus]
    num_workers = len(gpu_workers)

    print(f"GPU workers (one model per GPU): {gpu_workers}")
    print(f"apply_chat_template enable_thinking={args.enable_thinking}")
    print(
        f"prompt_batch_size={EVAL_BATCH_SIZE}, "
        f"max_num_seqs={EVAL_MAX_NUM_SEQS}"
    )

    for model_name in MODEL_NAMES:
        print(f"\n{'='*50}\nStarting evaluation for model: {model_name}\n{'='*50}")
        
        out_name = MODEL_DISPLAY_NAMES.get(model_name, Path(model_name).name)
        OUT_DIR = Path(f"justrl_eval_outputs/{out_name}")
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
                # Ensure the prompt format is correct.
                sample["prompt"] = PROMPT_TEMPLATE.format(problem=sample["prompt"])

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
                    EVAL_BATCH_SIZE,
                    N,
                )
                for i in range(len(sample_shards))
            ]
            
            # Use the spawn start method for worker processes.
            ctx = multiprocessing.get_context("spawn")
            
            with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as ex:
                futures = [ex.submit(worker_process, tup) for tup in args_list]
                
                # Track overall progress with tqdm.
                for fut in tqdm(concurrent.futures.as_completed(futures),
                                total=len(futures), desc=f"GPU workers ({task_name})"):
                    try:
                        res = fut.result()
                        all_results.extend(res)
                    except Exception as e:
                        print(f"A worker process failed with error: {e}")

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
