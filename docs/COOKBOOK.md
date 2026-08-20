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
`MATH-500` 的 `N=1` 配置下，只有第一个 worker 实际承担 rollout，
其余 worker 仍会加载模型但没有 rollout ID，存在显存和进程开销。

然后执行：

```bash
cd scripts/val/eval
python gen_vllm.py
```

也可以控制是否启用模型的 thinking template：

```bash
python gen_vllm.py --enable-thinking
```

```bash
python gen_vllm.py --disable-thinking
```

生成结果写入：

```text
scripts/val/eval/justrl_eval_outputs/<model-name>/
```

当前默认评测任务为 `MATH-500`，每道题生成 `N=1` 个 response。当前实现按
题目做数据并行分片，每个 GPU worker 再按 batch 调用
`llm.generate(...)`，不会把全部 500 道题塞进单次调用。默认采样参数和最大生成长度在
`gen_vllm.py` 顶部修改：

```python
MAX_TOKENS = 31744
TEMPERATURE = 0.7
TOP_P = 0.95
```

当前实现已按题目做 8 卡数据并行分片，并按每批 32 道题调用 vLLM；
可以通过 `EVAL_BATCH_SIZE` 和 `EVAL_GPUS` 覆盖默认值：

```bash
EVAL_BATCH_SIZE=16 \
EVAL_GPUS=0,1,2,3,4,5,6,7 \
EVAL_MODEL_PATH=/path/to/checkpoint \
python gen_vllm.py
```

本次已验证的 DeepSeek 1.5B MATH-500 命令：

```bash
EVAL_MODEL_PATH=/path/to/DeepSeek-R1-Distill-Qwen-1.5B \
EVAL_MODEL_NAME=DeepSeek-R1-Distill-Qwen-1.5B \
EVAL_BATCH_SIZE=16 \
EVAL_GPUS=0,1,2,3,4,5,6,7 \
python gen_vllm.py --disable-thinking
```

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
- 如果要稳定控制显存，应给 `gen_vllm.py` 增加分块 batch 逻辑，例如每批
  32/64/128 道题，而不是简单把 500 道题全部塞进一次 `generate`。
- `gen_vllm.py` 和 `grade.py` 含有实验作者的固定路径、模型名和 GPU 数量，
  执行前需要先检查顶部配置。
- `gen_vllm.py` 会跳过已经存在的结果文件；如果要重新生成，需要修改
  `REPLACE = True` 或删除对应输出文件。
- 评估生成和 grading 都应在 `scripts/val/eval` 目录执行，因为脚本内部
  使用相对路径。
- 不要把 `verl_example/opd.sh` 当作当前项目的主 OPD 入口；它是独立的
  `verl` 示例，配置组和路径与当前仓库的 legacy OPD 入口不同。
