# EOPD sampled-token implementation

本仓库的 EOPD（Entropy-Aware On-Policy Distillation）实现复用现有
sampled-token OPD 的训练流程，不引入另一套 trainer：

1. 低熵 token 保持原有 sampled-token reverse-KL reward：
   `log p_teacher(x_t) - log p_student_old(x_t)`。
2. teacher 熵满足 `H_teacher > tau` 的 response 位置，额外加入
   teacher top-k 到 student 的 forward-KL。
3. forward-KL 使用 teacher top-k 概率在 top-k 集合内重新归一化的近似，
   student 概率使用同一批 teacher top-k ids 的 full-vocabulary log-prob。

默认参数与论文实验一致：

```yaml
rollout:
  eopd_enabled: true
  eopd_top_k: 16
  eopd_entropy_threshold: 0.8
  eopd_forward_kl_coef: 1.0
```

可直接使用的配置文件：

```text
configs/opd/experiments/opd_prompt32_step80_lr3e-6_eopd_20260821.yaml
```

当前只完成代码和 CPU 单元验证，尚未运行该 EOPD 配置，因此文档不包含
EOPD 的训练曲线或 benchmark 结果。运行后应单独记录到
`docs/experiments/`，不要把未验证的结果写入本文件。

PowerOPD、AOPD 和 TrOPD 暂未实现：这三个名称对应的具体论文公式/
约束位置尚未在当前仓库中确认，不能根据名称猜测实现。
