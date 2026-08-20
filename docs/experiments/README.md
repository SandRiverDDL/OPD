# OPD 实验索引

> 本目录按 run 保存 OPD 训练实验报告。  
> `docs/OPD_BENCHMARK_SUMMARY.md` 负责维护跨 run 的 benchmark 分数账本；
> 本目录负责记录每轮实验的目的、实际生效配置、可比性和解释边界。

## 1. Reference definitions

### Base model

```text
DeepSeek-R1-Distill-Qwen-1.5B
```

### Teacher

```text
JustRL-DeepSeek-1.5B
```

### Canonical OPD baseline

当前建议以后续实验优先对比：

```text
opd_strict_repro_20260818_013200
```

该 run 使用了：

- `N_RESPONSES=4`；
- actor/model/Adam state 为 `fp32`；
- 原始 DAPO-Math-17k 数据，按 prompt 长度过滤；
- `train_batch_size=64`；
- `MAX_RESP_LENGTH=7168`；
- `LOG_PROB_TOP_K=16`；
- `TOP_K_STRATEGY=only_stu`；
- `REWARD_WEIGHT_MODE=student_p`；
- `USE_KL=False`。

注意：当前已有 checkpoint 和独立 AIME25 评测覆盖到 `global_step_220`。
训练 dataloader 的理论总 step 数为 279，但本索引只把实际存在并核对过的
checkpoint 写入结果表。

### Canonical AIME25 evaluation

除非报告另有说明，AIME25 结果统一采用：

```text
题目数：30
N=16
temperature=0.7
top_p=0.95
max_tokens=31744
max_model_len=33792
enable_thinking=True
rule-based grading
```

主要指标：

- `avg@16`：480 条 response 的平均正确率；
- `best@16`：每道题 16 次采样中至少一次正确的题目比例，作为
  `pass@16` 风格辅助指标；
- `solve_all` / `solve_none`：16 次采样全部正确 / 全部错误的题目数。

## 2. Experiment registry

| Run | Status | Parent/reference | Main change | Training/evaluated steps | AIME25 avg@16 | Report |
|---|---|---|---|---:|---:|---|
| Base student | reference | — | 未进行 OPD | — | **25.00%** | [Benchmark summary](../OPD_BENCHMARK_SUMMARY.md) |
| Teacher | reference | — | teacher 模型 | — | **36.04%** | [Benchmark summary](../OPD_BENCHMARK_SUMMARY.md) |
| `opd_100steps_20260817_173758` | diagnostic | Base student | `N=1`、bf16 参数/state、持久化短数据集 | 100 / 100 | 24.38% | [Run report](opd_100steps_20260817_173758.md) |
| `opd_strict_repro_20260818_013200` | baseline | Base student | `N=4`、fp32 参数/state、原始 DAPO 配置 | 279 / 220 evaluated | **35.83%** | [Run report](opd_strict_repro_20260818_013200.md) |

## 3. Comparability notes

### 3.1 旧 run 不是干净的 `N=1` baseline

`opd_100steps_20260817_173758` 与严格复现 run 同时改变了多个因素：

- `N_RESPONSES: 1 → 4`；
- actor 参数 dtype：`bf16 → fp32`；
- Adam `exp_avg` / `exp_avg_sq`：`bf16 → fp32`；
- 训练数据：持久化短数据集 → 原始 DAPO-Math-17k；
- 过长 prompt 处理：关闭 → 开启；
- 训练时长：100 step → 预计 279 step，当前评测到 step220。

因此不能把两轮结果写成：

```text
N=4 比 N=1 提升了多少
```

更准确的表述是：

```text
严格复现配置明显优于旧配置；旧 run 的 bf16 参数/optimizer state
导致有效更新极度稀疏，很可能是性能差异的主要原因之一。
```

如果要研究 response 数量本身，后续应以严格复现配置为 parent，只改变：

```text
N_RESPONSES=4 → 1
```

并保持 dtype、数据集、长度、学习率、shuffle 和评测配置不变。

### 3.2 benchmark 结果必须固定生成口径

跨 run 比较时必须同时固定：

- `enable_thinking`；
- `max_tokens`；
- `temperature`；
- `top_p`；
- `N`；
- grading method；
- prompt/template；
- 是否存在 format-error 过滤。

特别是 MATH-500：历史结果已经显示，`max_tokens=7168` 与
`max_tokens=31744` 会产生很大差异，不能混成一条曲线。

## 4. 新增实验的最小流程

以后每一轮实验只需：

1. 为实际 run ID 新建 `docs/experiments/<run_id>.md`；
2. 在本文件 registry 增加一行；
3. 在 run 报告中写清 parent 和实际变更字段；
4. 将独立评测结果补充到 `docs/OPD_BENCHMARK_SUMMARY.md`；
5. 不把原始大日志、checkpoint 或完整 JSONL 复制到 docs。

推荐的状态标签：

- `planned`：尚未启动；
- `running`：正在运行；
- `baseline`：当前标准参考配置；
- `completed`：正常完成，可用于比较；
- `diagnostic`：用于排查问题，不是严格对照；
- `interrupted`：训练中断，但已有产物可分析；
- `invalid`：配置或运行错误，不应作为实验结论。

