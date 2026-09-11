# AOPD sampled-token implementation

本仓库的 AOPD（Asymmetric On-Policy Distillation）复用现有 sampled-token
OPD 流程，在 token level 做 exploitation / imitation 分流：

1. 计算 sampled token 的 `p_student - p_teacher`；
2. 当差值不超过 `threshold` 时，使用 sampled-token OPD reward；
3. 当 student 概率明显高于 teacher 时，不再用 OPD reward，而是在
   teacher top-k 集合上加入 GKD loss；
4. `jsd_beta=1` 是 teacher-to-student forward-KL，当前默认配置采用这一
   形式；`0 <= jsd_beta < 1` 可混入 top-k reverse-KL。

配置示例：

```yaml
distillation:
  method: aopd
  params:
    top_k: 16
    threshold: 0.1
    opd_weight: 1.0
    gkd_weight: 1.0
    jsd_beta: 1.0
```

可直接使用：

```text
configs/opd/experiments/opd_prompt32_step80_lr3e-6_aopd_20260821.yaml
```

AOPD 需要 teacher top-k，因此相较于 sampled-token vanilla/PowerOPD 会增加
top-k teacher 数据和一次 actor top-k log-prob forward；它的目标是减少在
student 已经优于 teacher 的 token 上进行错误模仿，而不是减少 teacher
forward 次数。

当前只完成代码和 CPU 单元验证，尚未运行 GPU AOPD 训练或 benchmark。
