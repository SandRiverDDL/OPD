# EOPD sampled-token implementation

本仓库的 EOPD（Entropy-Aware On-Policy Distillation）实现复用现有
sampled-token OPD 的训练流程，不引入另一套 trainer：

1. 低熵 token 保持原有 sampled-token reverse-KL reward：
   `log p_teacher(x_t) - log p_student_old(x_t)`。
2. teacher 熵满足 `H_teacher > tau` 的 response 位置，额外加入
   teacher top-k 到 student 的 forward-KL。
3. forward-KL 使用 teacher top-k 概率在 top-k 集合内重新归一化的近似，
   student 概率使用同一批 teacher top-k ids 的 full-vocabulary log-prob。

推荐通过统一的方法选择字段配置：

```yaml
distillation:
  method: eopd
  params:
    top_k: 16
    entropy_threshold: 0.8
    forward_kl_coef: 1.0
```

launcher 会把这组用户配置转换为内部的 rollout / policy-loss 字段。

可直接使用的配置文件：

```text
configs/opd/experiments/opd_prompt32_step80_lr3e-6_eopd_20260821.yaml
```

当前只完成代码和 CPU 单元验证，尚未运行该 EOPD 配置，因此文档不包含
EOPD 的训练曲线或 benchmark 结果。运行后应单独记录到
`docs/experiments/`，不要把未验证的结果写入本文件。

PowerOPD 已实现 sampled-token bounded power reward，详见
`docs/POWEROPD.md`。AOPD 和 TrOPD 暂未实现。
