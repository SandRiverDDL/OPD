# OPD 项目常用训练与评估命令

本文档只整理本仓库作者自己提供的训练、推理和评估入口，不重复
`verl/` 或 `LlamaFactory/` 自带的 cookbook。

默认从项目根目录执行。长任务建议使用 `tmux`、`nohup` 或集群调度系统，
日志分别查看对应脚本创建的 `logs/`、`checkpoint/` 和
`justrl_eval_outputs/` 目录。

## 1. OPD 训练

### 直接启动作者维护的 OPD 入口

```bash
bash on_policy_distillation.sh
```

当前脚本默认使用：

```text
student:  DeepSeek-R1-Distill-Qwen-1.5B
teacher:  JustRL-DeepSeek-1.5B
dataset:  DAPO-Math-17k
```

如果模型、数据不在脚本自动解析的位置，可以显式覆盖路径：

```bash
HF_STUDENT_MODEL_PATH=/path/to/student \
HF_TEACHER_MODEL_PATH=/path/to/teacher \
HF_DAPO_DATASET_PATH=/path/to/dapo-math-17k.parquet \
bash on_policy_distillation.sh
```

常见 OPD 参数通过环境变量覆盖。例如：

```bash
TRAIN_MAX_SAMPLES=8 \
N_RESPONSES=1 \
TOTAL_EPOCHS=1 \
MINI_BATCH_SIZE=1 \
bash on_policy_distillation.sh
```

正式训练前可以只检查路径解析，不启动 Ray：

```bash
DRY_RUN=True bash on_policy_distillation.sh
```

项目当前使用 `verl v0.7.0` 时，建议保持：

```bash
MAX_VAL_RESP_LENGTH="$MAX_RESP_LENGTH" \
TEST_FREQ=-1 \
bash on_policy_distillation.sh
```

原因是仓库作者在 `README.md` 中说明，`verl v0.7.0` 内置 validation
可能低估结果，正式结果应使用后面的独立评估流程。

## 2. GRPO 训练

仓库提供了 GRPO 参考入口：

```bash
bash grpo.sh
```

GRPO 入口的关键设置是：

```bash
ADV_ESTIMATOR=grpo \
LOG_PROB_TOP_K=0 \
bash grpo.sh
```

运行前应根据实际实验显式确认 `ACTOR_MODEL_PATH`、
`REWARD_MODEL_PATH`、`TRAIN_DATASET`、GPU 数量和 checkpoint 输出路径。

## 3. Teacher rollout 与 SFT

作者使用 `scripts/infer/vllm_rollout.py` 生成 teacher responses，再交给
LlamaFactory 做 SFT。

示例：

```bash
python scripts/infer/vllm_rollout.py \
  --input-parquet datasets/OpenThoughts3-1.2M-math.parquet \
  --model-path model/Qwen3-4B \
  --gpu-ids 0,1,2,3,4,5,6,7 \
  --enable-thinking false \
  --enable-rejection-sampling true \
  --max-attempts-per-rollout 3
```

生成数据后，使用对应的 LlamaFactory 配置训练：

```bash
llamafactory-cli train LlamaFactory/examples/train_full/qwen3_base_full_sft.yaml
```

上面的数据集和模型路径是 README 中的作者示例，当前仓库环境中不保证
这些路径已经存在，正式执行前需要替换成实际路径。

## 4. 独立评估流程

### 4.1 评估脚本位置

项目自己的评估脚本在：

```text
scripts/val/eval/gen_vllm.py
scripts/val/eval/grade.py
scripts/val/eval/utils.py
```

评估数据在：

```text
scripts/val/data/
```

当前包含：

```text
MATH-500
BRUMO25
CMIMC25
HMMT25
Minerva
Olympiad-Bench
```

评估流程复用了 [JustRL](https://github.com/thunlp/JustRL) 的 pipeline，
但生成和打分脚本已经放在本仓库中。

### 4.2 生成 vLLM 评估结果

可以通过环境变量指定 checkpoint：

```bash
EVAL_MODEL_PATH=/path/to/checkpoint \
python gen_vllm.py
```

脚本当前是按每张 GPU 启动一个 vLLM worker，默认使用 GPU `0` 到 `7`。
如果 GPU 数量不同，需要同步修改 `available_gpus`。不过当前
`MATH-500` 的 `N=1` 配置下，题目会均分到各个 GPU worker，每个 worker
加载一个 TP=1 vLLM 副本并处理自己的 shard。

然后执行：

```bash
cd scripts/val/eval
python gen_vllm.py
```

也可以控制是否启用模型的 thinking template：

```bash
python gen_vllm.py --config configs/opd/experiments/opd_sky7b_sampled_token.yaml
```

```bash
python gen_vllm.py --config configs/opd/base.yaml
```

生成结果写入：

```text
scripts/val/eval/justrl_eval_outputs/<model-name>/
```

当前默认评测任务为 `MATH-500`，每道题生成 `N=1` 个 response。当前实现按
题目做数据并行分片，每个 GPU worker 将自己的 shard 一次提交给
`llm.generate(...)`，由 vLLM 内部 scheduler 按 token budget 和并发序列数进行
continuous batching，避免 Python 层固定 batch 的长尾等待。默认采样参数和最大
生成长度写在 OPD YAML 的 `evaluation` 段中：

```python
evaluation:
  task: MATH-500
  data_path: scripts/val/data/MATH-500/test.parquet
  n: 1
  enable_thinking: false
  max_tokens: 31744
  max_model_len: 33792
  temperature: 0.7
  top_p: 0.95
```

当前实现已按题目做 8 卡数据并行分片。可以通过 `EVAL_GPUS`、
`EVAL_MAX_NUM_SEQS`、`EVAL_MAX_NUM_BATCHED_TOKENS` 和
`EVAL_GPU_MEMORY_UTILIZATION` 调整每个 vLLM worker 的调度上限和显存使用率：

```bash
EVAL_GPUS=0,1,2,3,4,5,6,7 \
EVAL_MAX_NUM_SEQS=32 \
EVAL_MAX_NUM_BATCHED_TOKENS=32768 \
EVAL_GPU_MEMORY_UTILIZATION=0.90 \
EVAL_MODEL_PATH=/path/to/checkpoint \
python gen_vllm.py
```

使用实验 YAML 时，评估配置按 `extends` 合并读取：

```bash
python gen_vllm.py \
  --config configs/opd/experiments/opd_sky7b_poweropd_step120_20260821.yaml \
  --model-path checkpoint/opd_sky7b_poweropd_step120_20260821/hf_step120 \
  --model-name opd_sky7b_poweropd_step120_20260821_step120
```

命令行参数优先级为：CLI > YAML `evaluation` > 旧版环境变量 > 默认值。
模型 checkpoint 和输出名称通常通过 CLI 指定；评测协议则保存在 YAML 中。

本次已验证的 DeepSeek 1.5B MATH-500 命令：

```bash
EVAL_MODEL_PATH=/path/to/DeepSeek-R1-Distill-Qwen-1.5B \
EVAL_MODEL_NAME=DeepSeek-R1-Distill-Qwen-1.5B \
EVAL_GPUS=0,1,2,3,4,5,6,7 \
EVAL_MAX_NUM_SEQS=32 \
EVAL_MAX_NUM_BATCHED_TOKENS=32768 \
python gen_vllm.py --config configs/opd/base.yaml
```

worker 完成后会打印输入/输出 token 数、生成耗时和
`generation_tokens_per_second`，可据此调节上述参数。`EVAL_BATCH_SIZE` 仍可作为
旧命令的兼容 fallback，但不再控制 Python 层的外部 batch；新命令应优先使用
`EVAL_MAX_NUM_SEQS`。

### 4.3 规则打分

`grade.py` 可以通过环境变量指定生成结果目录：

```bash
cd scripts/val/eval
EVAL_NAME=DeepSeek-R1-Distill-Qwen-1.5B \
EVAL_MODEL_PATH=/path/to/DeepSeek-R1-Distill-Qwen-1.5B \
python grade.py
```

默认只使用规则匹配，不加载 verifier 模型。结果写入：

```text
scripts/val/eval/justrl_eval_outputs/<NAME>/grading_results.json
```

### 4.4 启用 LLM verifier 打分

如果规则匹配失败的答案需要额外使用 verifier 判断：

```bash
cd scripts/val/eval
python grade.py --enable_model_verifier
```

启用前需要确认 `grade.py` 中的：

```python
MODEL_NAME = "../../model/CompassVerifier-3B"
```

路径指向可用的 verifier 模型，并且当前机器有足够 GPU 显存。

## 5. 评估时的注意事项

- 使用 `verl v0.7.0` 时，不建议把训练过程中的 validation 结果当作最终
  结果；应单独运行 `gen_vllm.py` 和 `grade.py`。
- 当前 MATH-500 是 500 道题，代码实际会以 500 条 prompt 作为一次
  vLLM 请求，不能通过命令行设置成独立的“512 batch”。500 小于 512，
  但 `MAX_TOKENS=31744` 时请求的潜在 KV cache 很大，不建议把“512”
  理解为 512 道题都同时完整生成；vLLM 会受显存和调度配置限制。
- 如果要稳定控制显存，应优先调节 `EVAL_MAX_NUM_SEQS` 和
  `EVAL_MAX_NUM_BATCHED_TOKENS`，而不是恢复 Python 层固定 batch。
- `gen_vllm.py` 和 `grade.py` 含有实验作者的固定路径、模型名和 GPU 数量，
  执行前需要先检查顶部配置。
- `gen_vllm.py` 会跳过已经存在的结果文件；如果要重新生成，需要修改
  `REPLACE = True` 或删除对应输出文件。
- 评估生成和 grading 都应在 `scripts/val/eval` 目录执行，因为脚本内部
  使用相对路径。
- 不要把 `verl_example/opd.sh` 当作当前项目的主 OPD 入口；它是独立的
  `verl` 示例，配置组和路径与当前仓库的 legacy OPD 入口不同。
