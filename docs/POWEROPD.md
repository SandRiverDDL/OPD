# PowerOPD sampled-token implementation

本仓库的 PowerOPD 实现在现有 sampled-token OPD 路径上替换
token-level distillation reward，不需要 teacher top-k 或完整词表分布。

对于 student rollout 中实际采样的 token，vanilla OPD 使用：

```text
r_t = log p_teacher(x_t) - log p_student(x_t)
```

PowerOPD 使用论文中的 bounded power reward：

```text
r_t^(alpha) = p_teacher(x_t)^alpha - p_student(x_t)^alpha
```

其中 `alpha > 0`。由于两个概率都在 `[0, 1]`，reward 落在 `[-1, 1]`。
当前训练配置使用 `token_reward_direct`，因此该 reward 会直接作为
token-level advantage 使用；实现修改的是 reward/advantage 的来源，而不是
另外增加一套 advantage estimator。

## 配置

推荐使用统一的方法选择字段：

```yaml
distillation:
  method: poweropd
  params:
    alpha: 5.0
```

可直接运行的实验配置：

```text
configs/opd/experiments/opd_prompt32_step80_lr3e-6_poweropd_20260821.yaml
```

`scripts/opd/run_opd.py` 会自动转换为内部配置：

```text
actor_rollout_ref.rollout.distillation_method=poweropd
actor_rollout_ref.rollout.poweropd_reward_alpha=5.0
actor_rollout_ref.rollout.log_prob_top_k=0
actor_rollout_ref.actor.policy_loss.loss_mode=vanilla
```

PowerOPD 的 `alpha` 不是 EOPD 的 forward-KL 系数。它是概率幂变换的指数：

- 较小的 `alpha` 更关注低概率 token，行为更接近 log-ratio；
- 较大的 `alpha` 更强地压低低概率差异，reward 仍保持有界。

当前默认 `alpha=5.0` 对齐官方 PowerOPD 仓库的启动脚本；正式实验仍应
根据当前 teacher/student、数据和训练预算单独记录结果。

## 当前边界

- 已实现论文的 sampled-token PowerOPD reward；
- 没有启用 EOPD 的 teacher top-k forward KL；
- 没有引入 PowerOPD 官方仓库中的额外 reward normalization、terminal-stop
  处理或 reward-position 绘图，这些不是 PowerOPD 核心 reward 公式所必需的；
- 尚未运行 GPU PowerOPD 训练，因此当前没有 benchmark 结果。
