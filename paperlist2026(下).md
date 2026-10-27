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

  What evolves?（进化的是什么）
  What feedback drives it?（什么反馈驱动）
  Where does the loop close?（循环在哪里闭合）
  "If the loop closes on benchmarks, we get stronger benchmark solvers... If it closes on the physical world,
  agents may become a new way to build and improve real systems. The world is still the hardest environment.
  It is also the place where self-evolving agents matter most."
  ````

# [Discoverative Intelligence(探索式智能)](https://www.apodex.com/blog/apodex-1.0)
````
Apodex-1.0 是 Apodex 公司于 2026 年 6 月 8 日发布的新一代深度研究（deep-research）模型，
本质是一个标准的、会使用工具的 ReAct 智能体。
它的特殊之处在于：当以 Heavy-Duty 模式运行时 —— 即由一个异步智能体团队负责分工、交叉核验，
并在作答前对自己的证据进行审计 —— 同一个模型就变成旗舰级的 Apodex-1.0-H，
官方称其在多个深度研究基准上达到新的 SOTA（含开源与闭源模型）。

核心理念：Discoverative Intelligence
文章的核心论点是：今天最难的研究类任务，瓶颈不在于模型容量，而在于模型被允许与外部世界交互的范围。
长时程研究有一个共同的结构特征 —— 单次前向推理不够用，
单个上下文窗口装不下全部工作；它要求推理与检索、工具调用、验证交错进行，跨越成百上千步和多条并行分支。
这种可靠性无法仅靠参数记忆获得，而必须来自所谓的 "探索式智能"（discoverative intelligence）：
通过主动与外部世界互动来推理，并在提交答案前用这种互动自我核验。

架构：Agent 团队 + 全局验证器 + AgentOS
与 "拉长单条推理链" 的思路不同，Apodex 用加宽框架的方式来扩展探索能力：由 Orchestrator（协调器）调度一个 Heavy-Duty 智能体团队，
专业化的子智能体并行探索，再由一个全局验证器（Global Verifier）在答案提交前审计汇总的证据。
官方称单个任务可协调最多 150 个子智能体、跨 15,000 步。
支撑它的是 AgentOS—— 一个与任务无关（task-agnostic）的运行时，负责托管整个团队。
````

# [Rethinking the Evaluation of Harness Evolution for Agents](https://arxiv.org/pdf/2607.12227)
````
这是一篇批判性评估论文，核心质疑的是当前火热的 "自动 Harness 进化"（Automatic Harness Evolution）研究的评估方法是否成立。
什么是 Harness？
Harness 指的是围绕 LLM agent 的外部脚手架 ——prompt、工具、记忆、验证例程、控制逻辑等。
固定模型不变，光靠改 harness 就能大幅改变 agent 表现，这就是 Harness Engineering。

论文指出的两个根本问题
缺基线对比：现有 harness 进化论文从不和 "更简单的方法" 在相同反馈和推理预算下做对比 —— 比如并行采样、
顺序精炼这些 test-time scaling 方法。
训练 / 测试污染：用基准测试里的单元测试作为搜索信号来进化 harness，然后又在同一个基准上报告最终性能 —— 相当于用考试题做训练集。
核心疑问：Harness 进化带来的提升，到底是真的改进了 harness 设计，还是只是因为多试了几次？
````
四种方法的统一对比框架
| 方法 | 改什么 | 探索方向 |
|---|---|---|
| Parallel Sampling（并行采样） | 不改 harness，固定 | 宽度 —— 同时跑 K 条独立轨迹 |
| Sequential Refinement（顺序精炼） | 不改 harness，固定 | 深度 —— 每条轨迹基于上一条改进 |
| Harness Evolution（Harness 进化） | 跨任务共享的 harness | 跨任务优化，产出可复用的 harness |
| Harness Scaling（Harness 缩放） | 单任务特定的 harness | 实例级适配，每道题单独改 harness |

#### Harness Evolution
````
  一批任务 (x₁, x₂, ..., xₙ)
        │
        ▼
  用当前 harness h₁ 跑所有任务
        │
        ▼
  收集所有轨迹 + 对错结果
        │
        ▼
  Meta agent 读所有经验，总结规律 → 产出新 harness h₂
        │
        ▼
  用 h₂ 再跑一遍这批任务
        │
        ▼
  重复 K 轮，得到最终 harness ĥ
        │
        ▼
  拿 ĥ 去跑从没见过的新任务 x
````

#### Harness Scaling
````
  一道任务 x
    │
    ▼
  用初始 harness h₁ 跑一遍 → y₁
    │
    ▼
  看 y₁ 的结果 + 对错信号
    │
    ▼
  Meta agent 针对【这道题】改 harness → h₂
    │
    ▼
  用 h₂ 再跑【同一道题】→ y₂
    │
    ▼
  重复 K 轮
    │
    ▼
  返回这条任务的最佳轨迹
````

#### [GENERATING DATA-DRIVEN REASONING RUBRICS FOR DOMAIN-ADAPTIVE REWARD MODELING](https://arxiv.org/pdf/2602.06795)

核心思想是：让模型自己 "犯错"，再从错误中总结出检查规则，最后用这些规则作为奖励信号来训练模型改进。
<img width="1552" height="666" alt="image" src="https://github.com/user-attachments/assets/ee93fb53-5a73-4711-ab2d-6d6cd323fd59" />

- Box A：生成错误推理轨迹
````
目标：用一个不完美的模型，在特定复杂领域（这里以大学微积分为例）批量生成 "会做错但推理过程看起来合理" 的样本。
输入：一组问题 + 标准答案（Ground-truth Answers）
问题示例：求某个函数的不定积分
正确答案：72x+c
过程：模型（Model）逐题作答，但输出的是错误的推理轨迹（Incorrect reasoning traces）
错误示例：模型最后得出 72x，忘记加常数项 +c
作用：相当于一个 "错误样本库"，为下一步提炼规则提供素材。
````
- Box B：Rubric Generator（评分规则生成器）
````
目标：把 Box A 里五花八门的错误，提炼成结构化、可复用的 "错误检查清单"。
输入：Box A 的错误轨迹 + 标准答案
输出：一组 Error Rubric Items（错误评分条目），每条采用三元组结构：
1. Error Keyword（错误关键词）
2. Error Description（错误描述）
3. Criteria for Error's Presence（判定标准）
作用：从 "零散的错题" 升级为 "有组织的错误分类体系"，相当于自动生成了一份阅卷扣分标准。
````

- Box C：作为强化学习的奖励函数
````
目标：把 Box B 的 rubric 变成一个自动打分器，用来训练模型。
工作流程：
1.把待评判的推理轨迹映射到相关关键词
2.识别这条轨迹命中了哪些 rubric 条目
3.如果命中任何一条错误条目 → Score: 0（即奖励为 0，说明推理有问题）
效果：经过这种奖励函数的 RL 训练后，模型（Trained Model）开始能输出正确的推理轨迹
对比示例：训练后模型会意识到 "...Wait, I need to add +c to this integral..."
特点：只需要少量人工标签，大部分奖励信号是自动从错误样本中生成的。
````

#### [Breaking the Exploration Bottleneck: Rubric-Scaffolded Reinforcement Learning for Open-Ended LLM Reasoning](https://arxiv.org/pdf/2508.16949)
````
本论文把rubrics同时用作强化学习训练中的两类信号——rollout 阶段的显式“脚手架”引导（随训练逐步衰减，促使模型内化推理模式）
和训练阶段的“可验证奖励”——从而打破 RLVR 在开放式通用推理任务上的探索瓶颈。
````
