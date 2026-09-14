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

** Self-Harness （Zhang 等人，2026）** 
依靠 LLM Agent 通过一个 "提议 — 评估 — 接受"（propose-evaluate-accept）循环来改进其自身的 harness。
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

** Agentic Harness Engineering（AHE；Lin 等人，2026）**

harness 进化的瓶颈在于可观测性—— 也就是说，当一次 rollout 失败时，我们需要知道是哪个组件导致了失败，并且每一次编辑都应当有证据支撑。

该框架构建了一个由 三大可观测性支柱驱动的闭环：
- 组件可观测性（Component observability）
  
  每一个可编辑的 harness 组件都在文件系统中拥有对应表示，从而使动作空间明确且可追溯。
  一个 harness 包含 7 个组件：系统提示（system prompt）、工具描述（tool description）、工具实现（tool implementation）、
  中间件（middleware）、技能（skill）、子 Agent 配置（sub-agent configuration）和长期记忆（long-term memory）。
  每一种失败模式都被映射到某一个组件上，从而使编辑更有针对性。
  
- 经验可观测性（Experience observability）
  
  将大量原始轨迹分析并总结为一个由证据和失败模式构成的层级结构。每个 harness 都会生成轨迹（traces）。
  使用一个 Agent（"Agent Debugger"）来分析这些轨迹 —— 每条轨迹单独存储在一个文件中 —— 并生成针对每个任务的分析报告，说明失败或成功的根本原因。
  所有单任务报告被聚合为一份基准概览（benchmark overview），供下一步使用；原始轨迹则可在需要时访问。这种分层访问结构更加节省 token。
  
- 决策可观测性（Decision observability）
  每一次编辑都附带一个预测，供下一轮验证。一个 Agent（"Evolve Agent"）读取仓库，决定编辑哪个组件，
  然后产出编辑内容及其背后的推理。每一次编辑都是一个文件级的、可证伪的声明，可以在下一轮中得到验证，并受到两条约束：
  编辑仅作用于 harness 工作区。runs 目录、tracer、verifier 和 LLM 配置均为只读 —— 这杜绝了一系列 reward hacking 行为
  （例如禁用 verifier、偷换模型、或提高推理预算），从而保证每一个记录在案的增益都可归因于 harness 编辑本身。
  编辑是证据驱动的，并附带一个 manifest 条目：失败证据的名称、推断出的根本原因、靶向修复方案，以及预测影响（包括预期修复的任务和存在回归风险的任务）。
