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
