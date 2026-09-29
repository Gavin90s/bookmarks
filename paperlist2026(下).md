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

  # [A Taxonomy of Self-evolving Agents](https://lsl.zone/blog/2026/a-taxonomy-of-self-evolving-agents/)
  ````
  核心框架：Model + Harness + Artifact
  Models（模型）：通常是 LLM，是回应提示的 "大脑"
  Harness（支架）：循环设计、记忆、工具等外围组件，把模型变成智能体
  Artifacts（产出物）：智能体产出的成果 —— 发现的算法、论文、机器人策略等

  1. Artifact Iterative Optimization（产出物迭代优化）
  当前这波浪潮的主力。人设定目标和评估标准，智能体反复生成→验证→改进输出物。
  代表工作：AlphaEvolve（科学 / 算法发现）、Analemma AI 的 FARS（运行 417 小时产出 166 篇全 AI 论文，花费约 18 万美元）、Recursive Superintelligence（发现更优 GPU kernel）
  与旧范式的区别：以前（如 Neural Architecture Search）需要人工定义搜索空间和算子；现在 LLM 本身既是算子又是优化器，搜索空间更大、启发式更强
  趋势：从数字环境（代码、浏览器、模拟器）走向物理世界 ——NVIDIA 的机器人策略搜索、LabOS 生物实验室、Qumus 量子材料实验

  2. Agent Harness Self-improvement（支架自我改进）
  动机：模型训练太贵，能否在不更新权重的前提下改进智能体？两条路径：
  提示 / 记忆层：从经验中提取规则存入 prompt（GEPA）、playbook（ACE）或记忆系统（Mem0）。虽然没动权重，但支架更新等价于参数更新
  工具 / 技能层：生成可复用的工具代码（Alita）或技能封装（Mem-UI、Claude Code Skills、Hermes Agent）。技能本质是上下文管理 —— 不用每次把所有细节塞进上下文窗口
  多智能体扩展：单个智能体塞太多领域知识会变慢、会混淆（比如 "squeeze" 到底是股市还是挤橙子）。于是走向专家分工 + 路由（Eevee、Alita-G）。路由是核心瓶颈，它本身就需要强模型—— 作者的金句："A human is a router."

  3. Model Learning without Gold Answers（无标准答案的模型学习）
  这一层真正更新模型权重，回答 "没有金标准时怎么学"。包括：
  伪标签 / 内部信号：self-training 用模型自己的预测构造伪标签；TTRL 用内部信号；DeepSeek-R1 把信号转成奖励做 RL
  自博弈与环境弱信号：SPIN、Absolute Zero 的自博弈；与环境交互学习。哪怕 "发消息没人回" 这种无响应也是一种弱信号
  Test-time Training（TTT）：一类特殊方法，推理时模型内部就在做梯度更新（DeltaNet 路线）
  与持续学习（Continual Learning）的关系：旧语境下的持续学习核心问题是 "灾难性遗忘"；但今天 LLM 圈说的 continual learning 已更接近自我进化智能体。作者指出同一个术语会随时代漂移（类比 "多模态" 含义的变迁）
  ````
