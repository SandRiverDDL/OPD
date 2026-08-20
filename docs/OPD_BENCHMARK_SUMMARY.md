# OPD Benchmark Summary

> 更新时间：2026-08-18  
> 本文只汇总仓库中**已经实际生成并核对过**的评测结果，不把未运行的
> benchmark 或未验证的推测写成结果。

## 1. 评测口径

### AIME25

- 数据集：AIME25，30 道题。
- 采样：`N=16`。
- 主要指标：`avg@16`，即 16 次 rollout 的平均单次正确率。
- 辅助指标：当前 `grade.py` 输出的 `best_score`，本文记为
  `best@16`；它表示每道题 16 次中至少有一次成功的题目比例，
  可作为 `pass@16` 风格指标参考。
- 生成参数：

  ```text
  temperature=0.7
  top_p=0.95
  max_tokens=31744
  max_model_len=33792
  enable_thinking=True
  ```

- 每个模型输出 `30 × 16 = 480` 条 response；本文表格中的结果均已核对
  输出文件为 480 行，并存在 `grading_results.json`。
- 评分为 `grade.py` 的 rule-based grading，未启用模型 verifier。

### MATH-500

- 数据集：MATH-500，500 道题。
- 主要结果为 `N=1` 的 `mean_score`，可直接理解为单次 pass rate。
- 长输出评测使用：

  ```text
  temperature=0.7
  top_p=0.95
  max_tokens=31744
  ```

- 不同历史实验的 `enable_thinking` 设置可能不同，见各节说明；因此不能
  把所有历史 MATH-500 数值无条件放在同一条曲线上比较。

## 2. Base student 与 teacher

模型：

```text
student: DeepSeek-R1-Distill-Qwen-1.5B
teacher: JustRL-DeepSeek-1.5B
```

### 2.1 AIME25：avg@16

这两项使用同一套 AIME25 长输出、`N=16` 评测配置，均启用 thinking。

| 模型 | avg@16 | best@16 / pass@16 风格 | solve_all | solve_none | format errors |
|---|---:|---:|---:|---:|---:|
| Base student | **25.00%** | 46.67% | 2/30 | 16/30 | 56/480 |
| Teacher | **36.04%** | 56.67% | 3/30 | 13/30 | 57/480 |

相对 base student：

- teacher 的 `avg@16` 高 **11.04 个百分点**；
- teacher 的 `best@16` 高 **10.00 个百分点**。

结果文件：

```text
student:
scripts/val/eval/justrl_eval_outputs/
  aime25_avg16_fixed_20260817_230714_student/grading_results.json

teacher:
scripts/val/eval/justrl_eval_outputs/
  aime25_avg16_fixed_20260817_230714_teacher/grading_results.json
```

### 2.2 MATH-500：N=1

以下是独立的长输出 MATH-500 评测。两者的生成 template 不完全相同：
student 结果来自 `enable_thinking=False`，teacher 结果来自
`enable_thinking=True`，所以这张表可用于量级参考，但不应视为完全
严格的同口径对照。

| 模型 | MATH-500 mean score | 正确题数 | format errors | 平均输出长度 |
|---|---:|---:|---:|---:|
| Base student | **82.80%** | 414/500 | 12 | 5328.69 |
| Teacher | **86.00%** | 430/500 | 19 | 4167.15 |

结果文件：

```text
student:
scripts/val/eval/justrl_eval_outputs/
  DeepSeek-R1-Distill-Qwen-1.5B/grading_results.json

teacher:
scripts/val/eval/justrl_eval_outputs/
  teacher_JustRL-DeepSeek-1.5B_math500_long_20260817_221824/
  grading_results.json
```

## 3. 严格复现 run 的 checkpoint：AIME25

训练 run：

```text
checkpoint/opd_strict_repro_20260818_013200
```

本 run 的主要配置：

```text
student=DeepSeek-R1-Distill-Qwen-1.5B
teacher=JustRL-DeepSeek-1.5B
dataset=datasets/dapo-math-17k.parquet
filtered training samples=17909
N_RESPONSES=4
train_batch_size=64
ppo_mini_batch_size=64
MAX_PROMPT_LENGTH=1024
MAX_RESP_LENGTH=7168
MODEL_DTYPE=fp32
LOG_PROB_TOP_K=16
TOP_K_STRATEGY=only_stu
REWARD_WEIGHT_MODE=student_p
USE_KL=False
SAVE_FREQ=20
```

所有 checkpoint 使用同一套 AIME25 `avg@16` 配置。step20 和 step60 是
后续补评；step40、80、120、160、200、220 是每 40 steps 的评测点。

| Checkpoint | avg@16 | best@16 | solve_all | solve_none | format errors | 平均输出长度 |
|---:|---:|---:|---:|---:|---:|---:|
| step20 | **25.00%** | 43.33% | 2/30 | 17/30 | 44/480 | 12869.79 |
| step40 | **25.42%** | 53.33% | 1/30 | 14/30 | 34/480 | 10748.87 |
| step60 | **28.75%** | 50.00% | 2/30 | 15/30 | 19/480 | 9835.20 |
| step80 | **32.08%** | 53.33% | 3/30 | 14/30 | 16/480 | 8560.35 |
| step120 | **32.08%** | 56.67% | 2/30 | 13/30 | 21/480 | 8150.37 |
| step160 | **33.96%** | 53.33% | 3/30 | 14/30 | 35/480 | 8302.29 |
| step200 | **34.17%** | 50.00% | 4/30 | 15/30 | 39/480 | 8508.62 |
| step220 | **35.83%** | 56.67% | 4/30 | 13/30 | 37/480 | 8340.54 |

### 3.1 相对 base student 的变化

以 AIME25 base student 的 `25.00%` 为基线：

| Checkpoint | 相对 base student |
|---:|---:|
| step20 | +0.00 pp |
| step40 | +0.42 pp |
| step60 | +3.75 pp |
| step80 | +7.08 pp |
| step120 | +7.08 pp |
| step160 | +8.96 pp |
| step200 | +9.17 pp |
| step220 | **+10.83 pp** |

严格复现 run 当前按 AIME25 `avg@16` 的最佳 checkpoint 是：

```text
step220: 35.83%
```

step220 与 teacher 的 AIME25 结果 `36.04%` 只差 **0.21 个百分点**。
由于 AIME25 只有 30 道题，且每个 checkpoint 使用一次 `N=16` 采样，
这个差距不应解读为统计上显著的 teacher 等价。

结果根目录：

```text
scripts/val/eval/justrl_eval_outputs/
```

严格复现 run 的评测日志和汇总：

```text
logs/opd_strict_repro_20260818_013200_aime25_every40_20260818_1339.orchestrator.log
logs/opd_strict_repro_20260818_013200_aime25_every40_20260818_1339.summary.tsv
```

## 4. 已有的其他 OPD 实验结果

下面这些结果来自其他 run，不能与上面的严格复现曲线直接拼接。
它们保留在这里，方便查阅历史 benchmark。

### 4.1 另一轮 100-step run：AIME25 avg@16

run 标识：

```text
opd_100steps_20260817_173758
```

| Checkpoint | AIME25 avg@16 |
|---:|---:|
| step20 | 21.88% |
| step40 | 26.88% |
| step60 | 24.58% |
| step80 | 24.79% |
| step100 | 24.38% |

结果目录前缀：

```text
scripts/val/eval/justrl_eval_outputs/
  opd_100steps_aime25_avg16_20260817_232641_step*
```

### 4.2 另一轮 100-step run：MATH-500

这里同时存在两种 `max_tokens`，必须分开看。

#### 长输出配置：`max_tokens=31744`

| Checkpoint | MATH-500 mean score |
|---:|---:|
| step20 | 82.20% |
| step40 | 81.00% |
| step60 | 84.20% |
| step80 | 84.00% |
| step100 | 82.80% |

#### 短输出配置：`max_tokens=7168`

| Checkpoint | MATH-500 mean score |
|---:|---:|
| step20 | 75.00% |
| step40 | 73.60% |
| step60 | 75.40% |
| step80 | 74.80% |
| step100 | 74.80% |

短输出结果不能和 `max_tokens=31744` 的结果直接比较；差异主要可能
来自生成预算和长度截断，而不应直接归因于训练效果。

### 4.3 早期 dedup run：MATH-500

早期 run：

```text
opd_dedup_math500_20260814_202302_dedup
```

| 评测设置 | mean score |
|---|---:|
| `max_tokens=7168` | 74.40% |
| `max_tokens=31744` | **83.80%** |

同一 checkpoint 的这两个结果再次说明：MATH-500 的生成长度配置会
显著影响分数，必须在表格中显式区分。

## 5. 当前结论

1. **Base student 与 teacher 存在明显能力差距。**  
   AIME25 `avg@16` 为 `25.00% vs 36.04%`；MATH-500 长输出参考结果为
   `82.80% vs 86.00%`。

2. **严格复现 run 的 AIME25 曲线整体上升。**  
   `step20=25.00%`，到 `step220=35.83%`，提升
   **10.83 个百分点**。

3. **当前最佳 checkpoint 是 step220。**  
   但由于 AIME25 只有 30 题，step80/120 的相同分数以及
   step220 与 teacher 的 0.21 个百分点差距，都不能脱离采样误差过度解读。

4. **MATH-500 历史结果说明评测配置非常重要。**  
   `max_tokens=7168` 与 `31744` 的结果差异很大；后续比较 checkpoint
   时必须固定 `enable_thinking`、`max_tokens`、`temperature`、`top_p`、
   `N` 和 grading 方式。

5. **目前没有发现已完成的 AMC25/AMC23 结果文件。**  
   当前可直接纳入本总结的 benchmark 主要是 AIME25 和 MATH-500。

