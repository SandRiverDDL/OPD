# Prune-OPD overlap implementation

本仓库实现了 Prune-OPD 的 **overlap variant**。它使用 student top-k 与
teacher top-k 的重叠比例作为 teacher 信号可靠性的 proxy：

```text
overlap_ratio_t = |TopK_student(t) ∩ TopK_teacher(t)| / K
```

当某个 response position 的 overlap ratio 低于阈值时，视为一次不可靠
事件。之后的位置累计降低 OPD top-k reward 的 loss weight：

```text
w_t = clamp(1 - w_drop * cumulative_bad_events_t, 0, 1)
loss_weight_t = w_t + w_base
```

该实现只在 GPU actor worker 上处理 reward，不引入动态 response length、
额外 trainer 或异步队列，优先保持当前训练路径简单。

配置示例：

```yaml
distillation:
  method: pruneopd
  params:
    top_k: 16
    metric: overlap_ratio
    threshold: 0.7
    w_drop: 0.01
    w_base: 0.5
```

可直接使用：

```text
configs/opd/experiments/opd_prompt32_step80_lr3e-6_pruneopd_overlap_20260821.yaml
```

当前实现边界：

- 只实现 overlap ratio metric；
- 只实现 causal cumulative weighting；
- 未实现官方仓库中的 dynamic response length、teacher top-p accept metric
  和可视化；
- 尚未运行 GPU Prune-OPD 训练或 benchmark。
