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
