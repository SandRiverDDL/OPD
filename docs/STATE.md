# OPD 当前状态

本文档记录仓库中可复用的稳定状态，供后续任务快速恢复上下文。详细实验过程、完整日志和临时排查记录不应写入此处。

## 仓库概况

- 项目主题：On-Policy Distillation（OPD）及相关 SFT、GRPO 实验。
- 主要训练框架：`verl/`。
- SFT 相关代码：`LlamaFactory/`。
- OPD 训练入口：`on_policy_distillation.sh`。
- GRPO 训练入口：`grpo.sh`。
- 推理、数据处理和验证脚本：`scripts/`。
- 项目使用方法和公开实验说明：`README.md`。

## 当前稳定状态

- 已建立仓库级工作规范：`AGENTS.md`。
- 已建立状态记录文件：`docs/STATE.md`。
- 已新增一键环境脚本：`scripts/setup_opd_env.sh`。
- 已新增项目 skill：`.codex/skills/opd-set-up/SKILL.md`。
- 已新增训练与评估命令手册：`docs/COOKBOOK.md`。
- 当前独立评估入口 `scripts/val/eval/gen_vllm.py` 默认使用仓库内的
  `scripts/val/data/MATH-500/test.parquet`，每道题生成 1 个 response；
  生成引擎为 vLLM，当前代码按题目进行 8 卡数据并行分片，再按 batch
  调用 `llm.generate`；默认 grading 使用规则方法，可选
  `CompassVerifier-3B` 做二次判断。
- 已完成一次 DeepSeek-R1-Distill-Qwen-1.5B 的 MATH-500 评估：
  8 个独立 TP=1 vLLM worker、每卡 batch size 16、禁用 thinking，
  共生成 500 条 response。生成日志：
  `logs/eval/math500_deepseek_qwen15b_8gpu_20260814_163209.log`。
  结果：
  `scripts/val/eval/justrl_eval_outputs/DeepSeek-R1-Distill-Qwen-1.5B/`。
  规则评分 `mean_score=0.828`，`solve_all=414`，`solve_none=86`，
  `format_error_rollouts=12`，平均输出长度约 5328.69 tokens。
- `on_policy_distillation.sh` 已将 actor PPO、rollout/ref log-prob、
  teacher 前向和 vLLM 调度预算拆开：
  `PPO_MAX_TOKEN_LEN_PER_GPU`、`ROLLOUT_LOG_PROB_MAX_TOKEN_LEN_PER_GPU`、
  `REF_LOG_PROB_MAX_TOKEN_LEN_PER_GPU`、`TEACHER_MAX_TOKEN_LEN_PER_GPU`，
  以及 `VLLM_MAX_NUM_BATCHED_TOKENS`、`VLLM_MAX_NUM_SEQS`、
  `VLLM_GPU_MEMORY_UTILIZATION`。
- 已新增并验证训练后自动合并和评估入口：
  `scripts/run_opd_train_then_math500_eval.sh`。该入口固定使用
  `datasets/dapo-math-17k-dedup.parquet` 训练，训练完成后合并
  `global_step_20/actor`，再以绝对路径启动 8 卡 TP=1 vLLM 的 MATH-500
  评估，并执行规则评分。
- 训练数据当前按 `extra_info.index` 检查为 17,917 行、17,917 个唯一 ID；
  `dapo-math-17k-dedup.parquet` 与仓库原始 parquet 行内容相同，因此没有
  可由 ID 去除的重复行。训练时另有 8 条 prompt 因超过
  `MAX_PROMPT_LENGTH=1024` 被 verl 过滤，实际可训练样本为 17,909。
- 最近一次完整运行已结束（2026-08-14）：
  - run：`opd_dedup_math500_20260814_202302_dedup`
  - 配置：8 卡 FSDP/DP、rollout TP=1、`N_RESPONSES=1`、
    `MINI_BATCH_SIZE=64`、`MAX_RESP_LENGTH=7168`、
    `PPO_MAX_TOKEN_LEN_PER_GPU=16384`、vLLM `65536/16/0.85`、
    `TOTAL_TRAINING_STEPS=20`。
  - 训练日志：`logs/opd_dedup_math500_20260814_202302_dedup.train.log`
  - 最终 checkpoint：
    `checkpoint/opd_dedup_math500_20260814_202302_dedup/global_step_20/actor/`
  - 合并模型：
    `checkpoint/opd_dedup_math500_20260814_202302_dedup/hf_step20/`
  - 评估日志：
    `logs/opd_dedup_math500_20260814_202302_dedup.eval.log`
  - 评估 JSONL：500 行，位于
    `scripts/val/eval/justrl_eval_outputs/opd_dedup_math500_20260814_202302_dedup/`
  - 规则评分：`mean_score=0.744`、`solve_all=372`、
    `solve_none=128`、`avg_output_length=3780.768`、
    `format_error_rollouts=106`。
- 已使用与基座一致的长输出配置重新评估上述 `hf_step20`：
  8 个独立 TP=1 vLLM worker、每卡 batch size 16、
  `max_tokens=31744`、`max_model_len=33792`、`temperature=0.7`、
  `top_p=0.95`、禁用 thinking。生成 500/500 条，规则评分
  `mean_score=0.838`、`solve_all=419`、`solve_none=81`、
  `avg_output_length=5380.248`、`format_error_rollouts=12`。
  日志：
  `logs/opd_dedup_math500_20260814_202302_dedup_step20_mnt31744_parallel4.eval.log`；
  结果：
  `scripts/val/eval/justrl_eval_outputs/opd_dedup_math500_20260814_202302_dedup_step20_mnt31744_parallel4/`。
  此结果应与同配置基座 `mean_score=0.828` 对比；此前
  `max_tokens=7168` 的 `0.744` 不属于等长比较。
- 当前没有残留训练、评测或 Ray 任务。

## 已确认的本地环境状态

截至 2026-08-14，当前 `torch-base` 环境已确认：

- Python 3.12.11、PyTorch 2.7.1+CUDA 12.8。
- 当前节点有 8 张 NVIDIA H20 96 GB GPU，基础 CUDA tensor 运算正常。
- 已安装 vLLM 0.10.0、Ray 2.47.1、Transformers 4.54.x、FlashAttention 2.7.4、FlashInfer 0.2.9。
- 仓库内 `verl` 版本为 `0.7.0.dev`，运行时需先将仓库的 `verl/` 加入
  `PYTHONPATH`，或执行 editable install；否则 `python -m verl.trainer.main_ppo`
  无法找到训练模块。
- 当前已安装 `tensordict 0.10.0`、`math-verify 0.9.0` 和
  `latex2sympy2-extended 1.11.0`，数学 reward
  `verl.utils.reward_score.ttrl_math` 已通过 import 和 smoke test。
- 当前仓库没有 `model/DeepSeek-R1-Distill-Qwen-1.5B`、
  `model/JustRL-DeepSeek-1.5B` 或 `model/Qwen2.5-Math-1.5B`。
- Hugging Face 缓存位于
  `/apdcephfs_szcf/share_304335953/halanchen/.cache/huggingface`，已确认包含：
  `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`、
  `hbx/JustRL-DeepSeek-1.5B` 和
  `BytedTsinghua-SIA/DAPO-Math-17k`。
- 仓库内的 `datasets/dapo-math-17k.parquet` 和验证数据可读取；缓存中的 DAPO
  数据实际为 1,791,700 条，脚本现在会优先使用缓存 snapshot 中的 parquet。

## OPD 启动入口状态

- `on_policy_distillation.sh` 与当前仓库的 legacy OPD 实现相匹配；
  其 Hydra 配置关键项可正常组合。
- `on_policy_distillation.sh` 已支持从 HF cache 自动解析 student、teacher 和
  DAPO 数据路径，也支持从环境变量覆盖路径。
- 已完成一次真实 OPD smoke test：使用缓存中的 DeepSeek-R1-Distill-Qwen-1.5B
  student、JustRL-DeepSeek-1.5B teacher、8 条训练样本、8 次 rollout、8 个
  training steps，Ray、vLLM、FSDP、top-k token reward、数学 reward 和 checkpoint
  保存链路均实际跑通。日志：
  `logs/env_setup/opd_smoke_20260814_152249.log`。
- smoke test checkpoint：
  `checkpoint/token_reward_direct_DAPO-Math-17k_DeepSeek-R1-Distill-Qwen-1.5B_JustRL-DeepSeek-1.5B_64-T_1.0-Tch_1.0-n_8-mbs_1-topk_16-topk_strategy_only_stu-rw_student_p-2026-08-14_15-22-50/`。
- `verl_example/opd.sh` 当前不能直接使用：
  - 使用了本仓库配置中不存在的 `distillation.*` 配置组；
  - 默认训练数据路径 `datasets/DAPO-Math-17k/DAPO-Math.parquet` 不存在；
  - 自定义 reward 路径 `verl/recipe/r1_ascend/deepscaler.py` 不存在。
- 在修正环境和模型路径前，不应提交正式 OPD 训练任务。

## 一键环境配置

从项目根目录执行：

```bash
bash scripts/setup_opd_env.sh
```

脚本会安装仓库本地 `verl/` 和 OPD 所需的
`tensordict`、`math-verify`、`latex2sympy2-extended`，解析并检查
Hugging Face cache 中的 student、teacher、DAPO 数据，验证 Python/CUDA、
数学 reward 和 OPD launcher 的路径解析。

脚本只借鉴 HyTuner `set-up` 中的 tcodex 上下文操作，不复制其 Ray worker、
FastMCP、Nexus、OneAgent 或其他项目耦合逻辑。它会将 tcodex
`gpt-5.6-*` 的 `maxInputTokens` 限制为 `272000`，并 patch model catalog
生成路径，避免 tcodex 启动时远程 catalog 刷新后覆盖该上限。tcodex 按
95% 规则计算后，可用上下文约为 `258400`。修改前会创建
`.bak-opd-set-up` 备份。脚本还会将
`skill-creator/agents/openai.yaml` 的
`policy.allow_implicit_invocation` 设置为 `false`，保留显式调用能力。
脚本不迁移 tcodex 历史，也不启动 Ray 或训练。

常用检查方式：

```bash
# 只显示计划，不安装或修改 tcodex
bash scripts/setup_opd_env.sh --dry-run

# 跳过 pip 安装，复用现有环境完成检查和 tcodex 配置
bash scripts/setup_opd_env.sh --skip-install

# 没有 tcodex 时，仅配置和检查 OPD Python 环境
bash scripts/setup_opd_env.sh --skip-tcodex
```

截至 2026-08-14，已验证：

- `bash -n scripts/setup_opd_env.sh`
- `bash -n on_policy_distillation.sh`
- `bash scripts/setup_opd_env.sh --dry-run`
- `bash scripts/setup_opd_env.sh --skip-install`
- 当前 tcodex 实例 `~/.tcodex/instances/37387/models.json` 中的
  `gpt-5.6-sol`、`gpt-5.6-terra`、`gpt-5.6-luna` 均为
  `context_window=272000`、`auto_compact_token_limit=244800`。

## 维护约定

后续只在出现可复用的稳定变化时更新本文件，例如：

- 新增或变更正式入口、关键目录和配置来源。
- 已验证的环境要求、兼容性约束或固定操作方式。
- 正在运行且需要跨会话接续的长期任务。
- 已确认、尚未解决且会影响后续工作的阻塞问题。

每次记录长期任务时，至少包含：

```text
任务：
命令或 job ID：
日志：
关键配置：
已完成阶段：
当前异常：
下一步：
```

不要记录密钥、token、敏感环境变量、完整长日志或未经验证的推测。
