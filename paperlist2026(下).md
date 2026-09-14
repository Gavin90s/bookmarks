# [Goal Alignment in LLM-Based User Simulators for Conversational AI](https://arxiv.org/pdf/2507.20152)
针对 LLM 用户模拟器在多轮对话中无法持续遵循用户目标的错位问题，
提出 UGST(User Goal State Tracking) 框架将目标分解为画像、策略、任务、要求、偏好五类可动态追踪的子组件，
并通过 "UGST用于推理时引导→冷启动 SFT→GRPO 强化学习（通过UGST算reward)" 三阶段训练，使 8B 小模型的目标对齐能力追平甚至超越 70B 大模型，平均成功率最高提升 14.1%。

# [Language model harnesses are compositional generalizers](https://alexzhang13.github.io/blog/2026/harness/)

# [Harness Engineering for Self-Improvement](https://lilianweng.github.io/posts/2026-07-04-harness/#harness-optimization)
````
a harness is code that programs how prompts, tool calls, subagents, control flow, memory,
and workflow logic work together.
````
- 弱点挖掘（Weakness Mining）
- Harness 提议（Harness Proposal）
- 提议验证（Proposal Validation）

````
当前 harness h_t
    │
    ▼
① 弱点挖掘：跑任务 → 收集轨迹 → 聚类失败模式（含三层信息）
    │
    ▼
② Harness 提议：模型作为 proposer → 基于有界上下文 → 提出多样化的窄改动候选
    │
    ▼
③ 提议验证：held-in 测修复 + held-out 测无回归 → 两者都通过才合并
    │
    ▼
新 harness h_{t+1}（或无更新，进入下一轮）
````
