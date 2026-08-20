# OPD Sampled-Token with `Skywork-OR1-Math-7B` Teacher

## 1. Summary

| Field | Value |
|---|---|
| Run ID | `opd_sky7b_sampled_token` |
| Status | **completed** |
| Parent/reference | `opd_prompt32_step80_lr3e-6_sampled_token_20260819` |
| Main change | teacher 改为 `Skywork/Skywork-OR1-Math-7B` |
| Config | `configs/opd/experiments/opd_sky7b_sampled_token.yaml` |
| Training steps | 120 |
| Best AIME25 checkpoint | `step120` |
| Best AIME25 avg@16 | **29.58%** |

本 run 同时使用 sampled-token distillation（`LOG_PROB_TOP_K=0`）和
Skywork 7B teacher。由于 teacher、reward scale 和训练长度都不同，
它是 teacher 替换实验，不应与 JustRL teacher 的 top-16 曲线直接拼接。

## 2. Effective configuration

| Field | Value |
|---|---|
| Student | `DeepSeek-R1-Distill-Qwen-1.5B` |
| Teacher | `Skywork/Skywork-OR1-Math-7B` |
| Train dataset | `datasets/dapo-math-17k.parquet` |
| Prompt batch size | 32 |
| `ppo_mini_batch_size` | 32 |
| `N_RESPONSES` | 4 |
| Total training steps | 120 |
| Learning rate | `3e-6` |
| `MAX_PROMPT_LENGTH` | 1024 |
| `MAX_RESP_LENGTH` | 7168 |
| `LOG_PROB_TOP_K` | 0 |
| Distillation mode | sampled-token |
| `USE_KL` | `False` |

## 3. Training diagnostics

该 sampled-token run 没有 top-k overlap 指标；checkpoint ranking 使用训练
batch 中的 `true_reward_mean`：

| Step | Training true reward mean |
|---:|---:|
| 20 | 0.1875 |
| 40 | 0.09375 |
| 60 | 0.08594 |
| 80 | 0.15625 |
| 100 | 0.25000 |
| 120 | **0.265625** |

训练于 2026-08-20 04:54:20 +08:00 以 `rc=0` 结束，保存了
`global_step_20/40/60/80/100/120`。

## 4. AIME25 evaluation

所有 checkpoint 使用统一的 AIME25 长输出评测口径：

```text
30 题，N=16，temperature=0.7，top_p=0.95，
max_tokens=31744，max_model_len=33792，enable_thinking=True，
rule-based grading
```

| Checkpoint | avg@16 | best@16 | solve_all | solve_none | format errors | 平均输出长度 |
|---:|---:|---:|---:|---:|---:|---:|
| step20 | 25.63% | 40.00% | 3/30 | 18/30 | 128/480 | 19060.10 |
| step40 | 27.50% | 50.00% | 2/30 | 15/30 | 80/480 | 16361.91 |
| step60 | 28.33% | 53.33% | 2/30 | 14/30 | 66/480 | 15814.22 |
| step80 | 28.33% | 46.67% | 2/30 | 16/30 | 66/480 | 15582.58 |
| step100 | 29.17% | 46.67% | 4/30 | 16/30 | 51/480 | 15498.44 |
| step120 | **29.58%** | 43.33% | 3/30 | 17/30 | 44/480 | 14431.22 |

## 5. Interpretation and limitations

- AIME25 `avg@16` 随训练从 25.63% 增至 29.58%，同时平均输出长度和
  format errors 整体下降。
- step120 同时是本 run 的最佳 AIME25 checkpoint 和最高训练
  `true_reward_mean`，但 AIME25 只有 30 道题，不能据此证明稳定泛化。
- 该结果主要回答“替换 teacher 后是否能跑通并产生可分析曲线”，不构成
  Skywork teacher 与 JustRL teacher 的严格能力对照。

## 6. Artifacts

```text
logs/opd_sky7b_sampled_token_20260820.train.log
logs/opd_sky7b_sampled_token_20260820.status.log
logs/opd_sky7b_sampled_token_20260820.checkpoint_ranking.tsv
checkpoint/opd_sky7b_sampled_token/
scripts/val/eval/justrl_eval_outputs/
  opd_sky7b_sampled_token_aime25_avg16_step{20,40,60,80,100,120}/
```
