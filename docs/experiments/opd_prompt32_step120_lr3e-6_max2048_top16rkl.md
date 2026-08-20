# OPD Prompt32 / Step120：Max-2048 Top-16 Reverse-KL

## 1. Summary

| Field | Value |
|---|---|
| Run ID | `opd_prompt32_step120_lr3e-6_max2048_top16rkl` |
| Status | **completed** |
| Parent/reference | `opd_prompt32_step80_lr3e-6_20260819` |
| Main change | `max_prompt=1024`、`max_response=1024`，总长度上限 2048 |
| Config | `configs/opd/experiments/opd_prompt32_step120_lr3e-6_max2048_top16rkl.yaml` |
| Best AIME25 checkpoint | `step60` |
| Best AIME25 avg@16 | **32.29%** |
| Best training top-16 overlap | `step120: 0.9422` |

这里的 “2048” 指训练时最大 prompt 长度与最大 response 长度之和：

```text
1024 + 1024 = 2048 tokens
```

该 run 仍使用 JustRL teacher、top-16 reward、`only_stu` 和
`student_p`，主要用于观察短 response budget 的影响。

## 2. Effective configuration

| Field | Value |
|---|---|
| Student | `DeepSeek-R1-Distill-Qwen-1.5B` |
| Teacher | `JustRL-DeepSeek-1.5B` |
| Train dataset | `datasets/dapo-math-17k.parquet` |
| Prompt batch size | 32 |
| `ppo_mini_batch_size` | 32 |
| `N_RESPONSES` | 4 |
| Total training steps | 120 |
| Learning rate | `3e-6` |
| `max_prompt` | 1024 |
| `max_response` | 1024 |
| `max_val_response` | 1024 |
| `LOG_PROB_TOP_K` | 16 |
| `TOP_K_STRATEGY` | `only_stu` |
| `REWARD_WEIGHT_MODE` | `student_p` |
| `USE_KL` | `False` |

## 3. Training diagnostics

| Step | Top-16 overlap | Training true reward mean | Response length mean | Response clip ratio |
|---:|---:|---:|---:|---:|
| 20 | 0.7707 | 0.0000 | 1024 | 100% |
| 40 | 0.8778 | 0.0000 | 1024 | 100% |
| 60 | 0.9126 | 0.0078 | 1024 | 100% |
| 80 | 0.9316 | 0.0078 | 1024 | 100% |
| 100 | 0.9369 | 0.0078 | 1024 | 100% |
| 120 | **0.9422** | 0.0000 | 1024 | 100% |

训练于 2026-08-20 05:34:10 +08:00 以 `rc=0` 结束，保存了
`global_step_20/40/60/80/100/120`。由于训练 response 被 1024 token
硬截断，训练阶段的长度行为与普通 `MAX_RESP_LENGTH=7168` 的 run 不同。

## 4. AIME25 evaluation

独立 AIME25 评测仍使用统一的长输出设置，而不是训练时的 1024 上限：

```text
30 题，N=16，temperature=0.7，top_p=0.95，
max_tokens=31744，max_model_len=33792，enable_thinking=True，
rule-based grading
```

| Checkpoint | avg@16 | best@16 | solve_all | solve_none | format errors | 平均输出长度 |
|---:|---:|---:|---:|---:|---:|---:|
| step20 | 26.46% | 53.33% | 3/30 | 14/30 | 55/480 | 12003.19 |
| step40 | 30.63% | 46.67% | 5/30 | 16/30 | 39/480 | 10696.89 |
| step60 | **32.29%** | 50.00% | 3/30 | 15/30 | 40/480 | 11995.43 |
| step80 | 28.33% | 60.00% | 2/30 | 12/30 | 55/480 | 13420.39 |
| step100 | 30.21% | 53.33% | 1/30 | 14/30 | 60/480 | 13221.23 |
| step120 | 29.58% | 60.00% | 2/30 | 12/30 | 55/480 | 12819.44 |

## 5. Interpretation and limitations

- 训练中的 top-16 overlap 持续升高到 0.9422，但独立 AIME25 的最高
  `avg@16` 出现在 step60，说明训练内部 overlap 不能直接当作 benchmark
  选择指标。
- 该 run 的训练 response 在所有记录 step 都达到 1024 上限；因此它测试
  的不仅是 distillation step 数，还包括极短 response budget 带来的截断
  行为。
- 独立评测使用 `max_tokens=31744`，所以表中的平均输出长度大于训练上限
  并不矛盾；两者必须分开解释。
- 当前结果支持“2048 训练预算可以正常跑通并产生可分析 checkpoint”，
  但不足以证明它优于 7168 response budget。

## 6. Artifacts

```text
logs/opd_prompt32_step120_lr3e-6_max2048_top16rkl_20260820.train.log
logs/opd_prompt32_step120_lr3e-6_max2048_top16rkl_20260820.status.log
logs/opd_prompt32_step120_lr3e-6_max2048_top16rkl_20260820.checkpoint_ranking.tsv
checkpoint/opd_prompt32_step120_lr3e-6_max2048_top16rkl/
scripts/val/eval/justrl_eval_outputs/
  opd_prompt32_step120_lr3e-6_max2048_top16rkl_aime25_avg16_step{20,40,60,80,100,120}/
```
