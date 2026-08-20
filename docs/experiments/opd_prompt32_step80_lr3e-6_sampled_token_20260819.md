# OPD Prompt32 / Step80：Sampled-Token `lr=3e-6`

## 1. Summary

| Field | Value |
|---|---|
| Run ID | `opd_prompt32_step80_lr3e-6_sampled_token_20260819` |
| Status | **completed** |
| Parent/reference | `opd_prompt32_step80_lr3e-6_20260819` |
| Main change | `LOG_PROB_TOP_K=0`，改用 sampled-token distillation |
| Config | `configs/opd/experiments/opd_prompt32_step80_lr3e-6_sampled_token_20260819.yaml` |
| Best AIME25 checkpoint | `step80` |
| Best AIME25 avg@16 | **31.88%** |

这是 sampled-token OPD，不是 top-16 版本。训练日志明确记录：
top-k log probabilities 不存在，distillation reward 使用
`student_logp - teacher_logp`。因此不能把它的内部 reward 数值与
top-16 run 的 overlap 直接混用。

## 2. Effective configuration

| Field | Value |
|---|---|
| Student | `DeepSeek-R1-Distill-Qwen-1.5B` |
| Teacher | `JustRL-DeepSeek-1.5B` |
| Train dataset | `datasets/dapo-math-17k.parquet` |
| Prompt batch size | 32 |
| `ppo_mini_batch_size` | 32 |
| `N_RESPONSES` | 4 |
| Total training steps | 80 |
| Learning rate | `3e-6` |
| `MAX_PROMPT_LENGTH` | 1024 |
| `MAX_RESP_LENGTH` | 7168 |
| Actor/model and optimizer state | fp32 |
| `LOG_PROB_TOP_K` | 0 |
| Distillation mode | sampled-token |
| `USE_KL` | `False` |

## 3. Training status

训练到达 `step80` 并保存了：

```text
checkpoint/opd_prompt32_step80_lr3e-6_sampled_token_20260819/
  global_step_20/
  global_step_40/
  global_step_60/
  global_step_80/
```

训练日志尾部虽有 Ray worker 退出提示，但 checkpoint 已保存，后续
AIME25 评测也完整生成了 480 条 response 并完成 rule-based grading；
因此本报告将训练标为 completed，而不是根据清理阶段的 worker 提示
误判为失败。

## 4. AIME25 evaluation

评测口径：

```text
30 题，N=16，temperature=0.7，top_p=0.95，
max_tokens=31744，max_model_len=33792，enable_thinking=True，
rule-based grading
```

| Checkpoint | avg@16 | best@16 | solve_all | solve_none | format errors | 平均输出长度 |
|---:|---:|---:|---:|---:|---:|---:|
| step20 | 27.71% | 50.00% | 4/30 | 15/30 | 28/480 | 10578.07 |
| step40 | 31.04% | 53.33% | 3/30 | 14/30 | 21/480 | 8140.96 |
| step60 | 29.79% | 43.33% | 1/30 | 17/30 | 32/480 | 8192.84 |
| step80 | **31.88%** | 56.67% | 3/30 | 13/30 | 51/480 | 8227.88 |

## 5. Interpretation and limitations

- step80 的 `avg@16` 为 31.88%，与同 learning rate 的 top-16 run
  step80 数值相同，但 `best@16` 和 format-error 数不同；不能据此
  认为两种 reward 已经等价。
- sampled-token 与 top-16 的训练目标不同，后续比较应同时查看 reward
  定义、format errors、输出长度和独立评测结果。
- 本 run 的 AIME25 结果是独立 checkpoint 评测，不是训练过程中的固定
  validation 曲线。

## 6. Artifacts

```text
logs/opd_prompt32_step80_lr3e-6_sampled_token_20260819.launcher.log
logs/opd_prompt32_step80_lr3e-6_sampled_token_20260819.post_train_eval.log
logs/opd_prompt32_step80_lr3e-6_sampled_token_20260819.aime25_eval_summary.tsv
checkpoint/opd_prompt32_step80_lr3e-6_sampled_token_20260819/
scripts/val/eval/justrl_eval_outputs/
  opd_prompt32_step80_lr3e-6_sampled_token_20260819_aime25_step{20,40,60,80}/
```
