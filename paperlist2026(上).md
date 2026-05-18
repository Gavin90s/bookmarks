#### [AGENTRX: Diagnosing AI Agent Failures from Execution Trajectories](https://arxiv.org/pdf/2602.02475)

#### [CL-BENCH: A BENCHMARK FOR CONTEXT LEARNING](https://arxiv.org/pdf/2602.03587)

#### [Recursive Language Models](https://arxiv.org/pdf/2512.24601)

#### [ReSum: Unlocking Long-Horizon Search Intelligence via Context Summarization](https://arxiv.org/pdf/2509.13313)

#### [《Reinforcement Learning via Self-Distillation》（通过自蒸馏进行强化学习）](https://arxiv.org/pdf/2601.20802)
<img width="1512" height="456" alt="image" src="https://github.com/user-attachments/assets/322892f2-1fb9-4f14-8ae2-254f147033f7" />

#### [BENEFITS AND PITFALLS OF REINFORCEMENT LEARNING FOR LANGUAGE MODEL PLANNING:A THEORETICAL PERSPECTIVE](https://www.microsoft.com/en-us/research/wp-content/uploads/2026/03/iclr26_alpine_RL.pdf)
- SFT may introduce co-occurrence-based spurious solutions
- Policy Gradient enabling better generalization, but suffers from diversity collapse
- Q-learning provides two key advantages: off-policy learning and diversity preservation at convergence. Careful reward design is necessary to prevent Q-value bias in Q-learning. 

#### RouterReplay 核心作用
RouterReplay 是针对大模型（尤其是 MoE / 混合专家模型）路由机制设计的「路由行为记录与复现管理器」，
核心目的是精准控制、复现模型推理 / 训练过程中 “路由层选择哪些专家” 的行为，解决路由随机性导致的实验不可复现、训练不稳定等问题。
1. 算法层面的随机设计（主动引入）
这是最主要的来源，是 MoE 模型设计时故意加入的：
Top-K 路由的随机采样：很多 MoE 路由层不会直接选得分最高的 K 个专家，而是按得分做「随机采样」（比如按 softmax 后的概率分布采样）—— 目的是让专家负载更均衡，避免少数专家被频繁调用；
例：专家 A 得分 0.6、专家 B 得分 0.4，不是必选 A，而是有 60% 概率选 A、40% 概率选 B；
Dropout / 噪声注入：路由层会给分数加微小随机噪声，避免路由决策 “过度固化”；
温度系数（Temperature）：调整路由分数的 softmax 温度，温度 > 1 时会让采样更随机。
2. 数值 / 硬件层面的非确定性（被动引入）
即使关闭主动随机，硬件 / 数值计算也会引入微小随机性：
浮点精度差异：不同 GPU / 卡的浮点运算精度（如 FP16/FP32）、计算顺序不同，导致路由分数的微小偏差，最终改变 Top-K 选择；
并行计算顺序：分布式训练 / 推理时，不同卡的计算同步顺序不同，影响路由分数的最终结果；
随机种子未固定：若模型训练 / 推理时未固定全局随机种子，路由层的随机采样会完全不可控。
3. 动态上下文依赖（场景性随机）
在长文本 / 多轮对话场景中，路由决策依赖上下文的动态变化：
前文的路由选择会影响后续输入的特征表示，进而导致后续路由结果变化；
批次内样本的顺序、padding 方式不同，也会间接影响路由决策。

#### Ray instance 的核心作用
Ray = 分布式计算框架（用来做分布式训练、RL、大模型并行）
资源管理：统一管理本机的硬件资源（CPU 核心数、GPU 卡数、内存大小），并分配给 Ray 任务；
任务调度：接收你提交的 Ray 任务（如 @ray.remote 装饰的函数 / 类），调度到空闲资源上执行；
进程 / 节点管理：启动 Ray 的核心服务进程（如 raylet 节点管理器、plasma 共享内存存储），是所有 Ray 任务的 “总控中心”；
状态维护：记录任务运行状态、资源使用情况，支持任务监控、日志查看等。

#### [Evaluating LLM Reasoning Beyond Correctness and CoT](https://arxiv.org/pdf/2510.18134)
LLM 的 “推理” 本质是什么？不应是静态步骤链，而应是动态的观点交互与演化过程；借鉴黑格尔辩证法的 “正题 - 反题 - 合题”（Thesis-Antithesis-Synthesis）三元结构，
将推理定义为：通过观点冲突、矛盾解决实现认知升级的动态轨迹，而非静态的答案生成过程。

#### [DO NOT LET LOW-PROBABILITY TOKENS OVER-DOMINATE IN RL FOR LLMS](https://arxiv.org/pdf/2602.21435)

#### [SALT: Step-level Advantage Assignment for Long-horizon Agents via Trajectory Graph](https://assets.amazon.science/6d/51/5a9c67154242b3674647e2949087/salt-step-level-advantage-assignment-for-long-horizon-agents-via-trajectory-graph.pdf)

#### [LongBench v2: Towards Deeper Understanding and Reasoning on Realistic Long-context Multitasks](https://arxiv.org/pdf/2412.15204)

#### [Recursive Language Models](https://arxiv.org/pdf/2512.24601)

#### [KIMI K2: OPEN AGENTIC INTELLIGENCE](https://arxiv.org/pdf/2507.20534?)

#### [ROLL:Reinforcement Learning Optimization for Large-Scale Learning:An Efficient and User-Friendly Scaling Library](https://arxiv.org/pdf/2506.06122)

#### [训练加速40倍、打破“不可能三角”：MiniMax Agent RL 架构解密](https://mp.weixin.qq.com/s/n3vFJ3edKmMqWPWiTLwhGA)

#### [Agentic Reasoning for Large Language Models](https://arxiv.org/pdf/2601.12538)

#### [Reinforcement Learning via Self-Distillation](https://arxiv.org/pdf/2601.20802)

#### [SFT Trainer](https://huggingface.co/docs/trl/sft_trainer)

#### [AngelSlim:腾讯speculative decoding](https://github.com/Tencent/AngelSlim)

#### [SYNERGIZING UNDERSTANDING AND GENERATION WITH INTERLEAVED ANALYZING-DRAFTING THINKING]()

#### [Sutradhara: An Intelligent Orchestrator-Engine Co-design for Tool-based Agentic Inference](https://arxiv.org/pdf/2601.12967)

#### [AdaReasoner: Dynamic Tool Orchestration for Iterative Visual Reasoning](https://arxiv.org/pdf/2601.18631)

#### [Improving Long-Context Summarization with Multi-Granularity Retrieval Optimization](https://www.microsoft.com/en-us/research/wp-content/uploads/2026/01/AAAI_Chenxueyu.pdf)

#### [GENERALIZATION OF RLVR USING CAUSAL REASONING AS A TESTBED](https://arxiv.org/pdf/2512.20760)

#### [The Subtle Art of Defection: Understanding Uncooperative Behaviors in LLM based Multi-Agent Systems](https://cdn.amazon.science/ae/82/3acb183c44469b5da7284a2ffb03/eacl-industry-camera-ready.pdf)

#### [VehicleWorld: A Highly Integrated Multi-Device Environment for Intelligent Vehicle Interaction](https://arxiv.org/pdf/2509.06736)

#### [SKILL-MIX: A FLEXIBLE AND EXPANDABLE FAMILY OF EVALUATIONS FOR AI MODELS](https://arxiv.org/pdf/2310.17567)

#### [RAGEN-2: Reasoning Collapse in Agentic RL](https://arxiv.org/pdf/2604.06268)

#### [SortedRL: Accelerating RL Training for LLMs through Online Length-Aware Scheduling](https://arxiv.org/pdf/2603.23414)

#### [HIERARCHY-OF-GROUPS POLICY OPTIMIZATION FOR LONG-HORIZON AGENTIC TASKS](https://arxiv.org/pdf/2602.22817)

#### [NaviAgent: Bilevel Planning on Tool Navigation Graph for Large-Scale Orchestration](https://arxiv.org/pdf/2506.19500)

#### [Hindsight Credit Assignment for Long-Horizon LLM Agents](https://arxiv.org/pdf/2603.08754)

#### [SkillRouter: Skill Routing for LLM Agents at Scale](https://arxiv.org/pdf/2603.22455)

#### [STEP: Success-Rate-Aware Trajectory-Efficient Policy Optimization](https://arxiv.org/pdf/2511.13091)

#### [Journey Before Destination: On the importance of Visual Faithfulness in Slow Thinking](https://cdn.amazon.science/d0/03/4e132ffe43edaa0f1fcc614ba384/648-journey-before-destination.pdf)

#### [Align to Structure: Aligning Large Language Models with Structural Information](https://arxiv.org/pdf/2504.03622)

#### [SELF-ALIGNED REWARD: TOWARDS EFFECTIVE AND EFFICIENT REASONERS](https://arxiv.org/pdf/2509.05489)

#### [METASYNTH: Meta–Prompting–Driven Agentic Scaffolds for Diverse Synthetic Data Generation](https://arxiv.org/pdf/2504.12563)

#### [Where Did It All Go Wrong? A Hierarchical Look into Multi-Agent Error Attribution](https://arxiv.org/pdf/2510.04886)

#### [Which Agent Causes Task Failures and When? On Automated Failure Attribution of LLM Multi-Agent Systems](https://arxiv.org/pdf/2505.00212)

#### [Efficient Tool Use with Chain-of-Abstraction Reasoning](https://arxiv.org/pdf/2401.17464)
<img width="1422" height="1510" alt="image" src="https://github.com/user-attachments/assets/b41dc32e-0313-4f43-9762-15536a7d406f" />

#### [Learning to Reason Over Time: Timeline Self-Reflection for Improved Temporal Reasoning in Language Models](https://cdn.amazon.science/c8/af/efaf81c04dfc946ab697bd1b7cb2/4038-learning-to-reason-over-t.pdf)

#### [REIC: RAG-Enhanced Intent Classification at Scale](https://arxiv.org/pdf/2506.00210)

#### [Towards Compositional Generalization of LLMs via Skill Taxonomy Guided Data Synthesis](https://arxiv.org/pdf/2601.03676)

#### [Enhancing LLM-as-a-Judge via Multi-Agent Collaboration](https://arxiv.org/pdf/2603.00993)
<img width="2196" height="1436" alt="image" src="https://github.com/user-attachments/assets/1b4a2296-f0c1-49b5-8ae6-1792b7a53d11" />

#### [On Synthetic Data Strategies for Domain-Specific Generative Retrieval](https://aclanthology.org/2025.acl-long.392.pdf)

#### [Exploring Quality and Diversity in Synthetic Data Generation for Argument Mining](https://aclanthology.org/2025.emnlp-main.1351.pdf)

#### [Structuring the Unstructured: A Multi-Agent LLM Framework for Transforming Ambiguous SOPs into Code](https://aclanthology.org/2025.emnlp-industry.163.pdf)
````
核心定位：亚马逊提出 SYNTACT 多智能体 LLM 框架，专门把模糊、非结构化自然语言 SOP 作业流程，自动转换成结构化规范流程 + 可执行工作流代码。
核心架构：三段流水线智能体
消歧器(Clarifier)：找出 SOP 歧义、信息缺失，结合知识库补全信息；
规划器(Planner)：拆解任务、匹配工具 API、生成标准化流程骨架；
执行器(Implementor)：加入类型约束、解析分支循环，输出可直接运行的工作流。
````

#### [DS2-INSTRUCT: Domain-Specific Data Synthesis for Large Language Models Instruction Tuning](https://aclanthology.org/2026.findings-eacl.176.pdf)
- 关键词构建：基于任务定义，双向扩展构建分层领域关键词池。
- 指令生成：结合布鲁姆认知层级，生成多难度、多维度指令集。
- 质量过滤：通过多响应投票校验，筛选高质量指令 - 响应对。

#### [DFLOW: Diverse Dialogue Flow Simulation with Large Language Models](https://arxiv.org/pdf/2410.14853)
````
核心问题：面向任务的对话系统 (Task-Oriented Dialogue, TOD) 开发需要能够遵循特定任务逻辑的数据，但现有数据模拟方法存在显著局限：
过度关注话语层面的多样性（语言、主题、对话行为），忽视了对话层面的任务逻辑多样性
人工构建对话流程 (dialogue flow) 成本高、难度大，现有数据集存在对话流程标注稀疏问题
缺乏高效、可泛化的自动对话流程生成机制
研究目标：设计一种自动数据模拟方法，生成遵循任务逻辑和约束的多样化面向任务对话，解决小模型在对话状态理解任务中的性能瓶颈

DFLOW 框架包含三个核心步骤，形成 "任务计划→对话流程→对话生成" 的完整流水线：
1. 任务计划生成 (Task Plan Generation)
输入：任务指令 + 上下文示例
过程：使用 LLM 规划器生成决策树结构的任务计划，包含多种可能的任务轨迹
系统动作分类（作为决策树节点）：
Yes/No 问题：二元选择，引导至不同分支
多选题：多选项选择，继续到下一有效动作
用户信息请求：文本输入，继续到下一有效动作
推荐：提供最终系统建议，标志流程结束
2. 对话流程采样 (Dialogue Flow Sampling)
DFS 解析：对任务计划应用深度优先搜索，提取所有有效轨迹作为对话流程
错误处理流程增强：
超出范围请求流程：处理违反系统约束的用户输入，生成错误提示并引导有效输入
提前终止对话流程：模拟用户在任务完成前结束对话的场景
3. 对话生成 (Dialogue Generation)
输入：对话流程
过程：LLM 合成器生成多轮对话，每轮与流程中的步骤关联
质量过滤：自动过滤重复话语或与流程步骤无关的低质量对话
````
#### [AutoEval-ToD: Automated Evaluation of Task-oriented Dialog Systems](https://aclanthology.org/2025.naacl-long.508.pdf)
````
任务型对话系统（ToD）是客服、技术支持、智能推荐等场景自动化用户交互的核心载体，可完成多轮对话、理解用户意图并执行特定任务（如订票、故障排查）。
ToD 因设计复杂、交互动态性强，评估难度极高，传统评估远不止判断任务是否完成，还需覆盖响应质量、对话连贯性、检索准确率、用户体验、交互效率等多重维度。
现有评估高度依赖人工标注，存在三大致命缺陷：效率低下、主观偏差大、规模化部署成本极高，严重制约 ToD 系统的迭代与落地。
领域迫切需要一套可靠、可扩展、系统化的评估框架，以全面量化 ToD 系统性能。

AutoEval-ToD 自动化评估框架
该论文提出基于大语言模型（LLMs）的端到端自动化评估框架 AutoEval-ToD，核心流程为两步：
框架主动与目标 ToD 系统开展交互对话；
同步解析 ToD 的对外响应内容与内部运行状态，从多关键维度完成性能评估。

通用化接入能力：以易用 API 形式封装，可与任意 ToD 系统无缝集成，支持持续评估与性能追踪。
内置用户模拟器：可模拟真实业务场景的用户交互，自动生成对应评估指标，替代人工对话测试。
全维度综合评估：突破传统单一指标局限，构建覆盖 ToD 核心能力的立体化评估体系。

五大核心评估维度
框架聚焦任务型对话系统的 5 项关键能力展开评估：
检索性能（Retrieval Performance）
用户体验（User Experience）
轮次优化（Turn Optimization）
领域合规性（Domain Compliance）
响应质量（Response Quality）
````

#### [Stabilizing MoE Reinforcement Learning by Aligning Training and Inference Routers](https://arxiv.org/pdf/2510.11370)
