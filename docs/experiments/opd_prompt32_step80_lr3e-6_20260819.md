# OPD Prompt32 / Step80：`lr=3e-6` Top-16

## 1. Summary

| Field | Value |
|---|---|
| Run ID | `opd_prompt32_step80_lr3e-6_20260819` |
| Status | **completed** |
| Parent/reference | `opd_prompt32_step80_20260818_181805` |
| Main change | learning rate 改为 `3e-6` |
| Config | `configs/opd/experiments/opd_prompt32_step80_lr3e-6.yaml` |
| Best AIME25 checkpoint | `step60` |
| Best AIME25 avg@16 | **33.33%** |
| Best training top-16 overlap | `step80: 0.9242` |

该 run 与 prompt32/step80 的前一轮保持相同的 batch、响应数量、长度限制
和 top-16 reward，主要用于观察更大学习率在 80 step 内的行为。它仍然
不是只相对 strict baseline 改一个字段，因为 prompt batch size 已从
canonical baseline 的 64 改为 32。

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
| `LOG_PROB_TOP_K` | 16 |
| `TOP_K_STRATEGY` | `only_stu` |
| `REWARD_WEIGHT_MODE` | `student_p` |
| `USE_KL` | `False` |

## 3. Training diagnostics

| Step | Top-16 overlap | Advantage abs mean | Grad norm | Response length mean | Clip ratio |
|---:|---:|---:|---:|---:|---:|
| 20 | 0.7719 | 0.01704 | 1.0014 | 5643.23 | 46.88% |
| 40 | 0.8740 | 0.00924 | 0.2943 | 5772.55 | 42.97% |
| 60 | 0.9136 | 0.00596 | 0.1559 | 5508.23 | 30.47% |
| 80 | **0.9242** | 0.00538 | 0.1077 | 5824.55 | 40.63% |

训练进程正常到达 `step80`，并保存了 `global_step_20/40/60/80`。

## 4. AIME25 evaluation

所有 checkpoint 使用统一的 AIME25 长输出评测口径：

```text
30 题，N=16，temperature=0.7，top_p=0.95，
max_tokens=31744，max_model_len=33792，enable_thinking=True，
rule-based grading
```

| Checkpoint | avg@16 | best@16 | solve_all | solve_none | format errors | 平均输出长度 |
|---:|---:|---:|---:|---:|---:|---:|
| step20 | 26.67% | 50.00% | 2/30 | 15/30 | 25/480 | 10037.28 |
| step40 | 31.67% | 50.00% | 2/30 | 15/30 | 28/480 | 7946.27 |
| step60 | **33.33%** | 50.00% | 3/30 | 15/30 | 28/480 | 7888.09 |
| step80 | 31.88% | 46.67% | 3/30 | 16/30 | 37/480 | 7924.95 |

## 5. Interpretation and limitations

- 在本 run 的 80 step 范围内，step60 的 AIME25 `avg@16=33.33%` 最高；
  step80 的训练 overlap 最高，但独立 benchmark 没有继续提升。
- 相对 `lr=1e-6` 的 prompt32/step80 run，结果只能作描述性比较；
  AIME 采样、checkpoint 选择和短训练长度都会带来波动。
- AIME25 只有 30 道题，不能根据 step60 与 step80 的小幅差异判断稳定
  的最优训练时长。

## 6. Artifacts

```text
logs/opd_prompt32_step80_lr3e-6_20260819.train.log
logs/opd_prompt32_step80_lr3e-6_20260819.checkpoint_overlap_ranking.tsv
logs/opd_prompt32_step80_lr3e-6_20260819.aime25_eval_summary.tsv
checkpoint/opd_prompt32_step80_lr3e-6_20260819/
scripts/val/eval/justrl_eval_outputs/
  opd_prompt32_step80_lr3e-6_20260819_aime25_by_overlap_rank{1,2,3,4}_step*/
```
