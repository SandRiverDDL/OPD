---
name: opd-set-up
description: 配置 OPD 的 torch-base/verl 环境，检查 HF 缓存中的模型和数据，并将 tcodex 的 gpt-5.6 上下文上限配置为 272000（tcodex 实际可用窗口约 258400）。只在用户明确要求配置 OPD 环境时调用。
---

# OPD Set Up

从项目根目录执行：

```bash
bash scripts/setup_opd_env.sh
```

这个 skill 只负责本项目环境配置，不启动 Ray、不提交训练、不启动 teacher，也不安装
HyTuner 的 worker 服务。

## 配置内容

脚本是幂等的，执行以下操作：

1. 将当前仓库的 `verl/` 以 editable package 安装，不额外覆盖 PyTorch、vLLM 或 CUDA：

   ```bash
   python -m pip install -e ./verl --no-deps
   ```

2. 安装 OPD 数学 reward 和 `tensordict` 依赖：

   ```bash
   python -m pip install --no-cache-dir \
     'tensordict>=0.8.0,<=0.10.0,!=0.9.0' \
     math-verify \
     latex2sympy2-extended
   ```

3. 从 Hugging Face cache 自动解析并检查：

   ```text
   DeepSeek-R1-Distill-Qwen-1.5B
   JustRL-DeepSeek-1.5B
   DAPO-Math-17k
   ```

4. 验证 `torch`、CUDA、`verl`、vLLM、Ray、数学 reward 和 OPD launcher。

5. 参考 hytuner 的 `set-up`，只复用其中的 tcodex 上下文操作：

   - 将 `gpt-5.6-*` 的 `maxInputTokens` 设置为 `272000`；
   - 修改 tcodex 的 model catalog generation 路径，使远程 catalog 刷新时仍保留该上限；
   - 将 `skill-creator/agents/openai.yaml` 中的
     `policy.allow_implicit_invocation` 设置为 `false`，保留显式调用能力；
   - tcodex 自身按 95% 规则计算后，可用上下文约为 `258400`；
   - 修改前分别创建 `.bak-opd-set-up` 备份；
   - 不迁移或同步 tcodex 历史。

脚本执行结束后必须重启 tcodex。重启后检查当前实例的 `models.json`：

```bash
python - <<'PY'
import json
from pathlib import Path

for path in sorted((Path.home() / ".tcodex/instances").glob("*/models.json")):
    data = json.loads(path.read_text())
    for model in data.get("models", []):
        if str(model.get("slug", "")).startswith("gpt-5.6-"):
            assert model.get("context_window") == 272000, (path, model)
            assert model.get("auto_compact_token_limit") == 244800, (path, model)
            print(path, model["slug"], "context configuration OK")
PY
```

## 常用选项

只检查，不修改环境或 tcodex 文件：

```bash
bash scripts/setup_opd_env.sh --dry-run
```

只验证已有 Python 环境和 tcodex：

```bash
bash scripts/setup_opd_env.sh --skip-install
```

在没有 tcodex 的机器上只配置 OPD Python 环境：

```bash
bash scripts/setup_opd_env.sh --skip-tcodex
```

## 运行边界

- 不下载模型和数据；模型、数据必须已经在 HF cache 或由环境变量指定。
- 不执行 `ray start`、`ray stop`、训练、评测或 checkpoint 清理。
- 不安装 FastMCP、Nexus、OneAgent 或其他 HyTuner 专属依赖。
- 不打印 token、API key、secret、完整环境变量或 runtime 配置。
- 训练入口仍是 `on_policy_distillation.sh`；本 skill 只负责让它具备可运行环境。

## 验收标准

配置完成至少应满足：

- `verl` 可 import；
- `tensordict` 满足仓库版本要求；
- `math_verify` 和 `latex2sympy2_extended` 可 import；
- CUDA 可用；
- student、teacher、DAPO 数据和验证数据存在；
- `on_policy_distillation.sh` 的 `DRY_RUN=True` 检查成功；
- 重启 tcodex 后，`gpt-5.6-*` 显示 `context_window=272000`、
  `auto_compact_token_limit=244800`。
