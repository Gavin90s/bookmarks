#### [Unified Active Retrieval for Retrieval Augmented Generation](https://arxiv.org/pdf/2406.12534)
````
Active Retrieval，指基于用户输入判断是否需要进行检索的过程。论文基于Self-aware、Time-aware、Knowledge-aware、
Intent-aware 四个维度，构建了一个决策树用于判断是否需要检索。
````
![image](https://github.com/user-attachments/assets/6cfbb1fe-616c-4a96-b05a-1e9f947c0ef4)

#### [ReEval: Automatic Hallucination Evaluation for Retrieval-Augmented Large Language Models via Transferable Adversarial Attacks](https://aclanthology.org/2024.findings-naacl.85.pdf)
````
通过 Answer Swapping 和 Context Enriching 来生成测试集，用于做 RAG 的 Hallucination Evaluation。
````
#### [CharXiv: Charting Gaps in Realistic Chart Understanding in Multimodal LLMs](https://arxiv.org/pdf/2406.18521)
````
开源了CharXiv图表任务的benchmark，包含2类图表问题：
1）关于检查图表基本元素的描述性问题，
2) 需要综合图表中复杂视觉元素的信息进行推理的问题。
多模态大语言模型在简单的图表任务上，能力已经接近human。在需要推理的复杂任务上，效果还不太行。
````

#### [Meta Reasoning for Large Language Models](https://arxiv.org/pdf/2406.11698)

#### [BRIGHT: A Realistic and Challenging Benchmark for Reasoning-Intensive Retrieval](https://arxiv.org/pdf/2407.12883)
````
开源了 BRIGHT Reasoning-Intensive 检索任务 benchmark。
MTEB 排行榜上的领先模型 [38] 在 nDCG@10 上取得了 59.0 的得分，而在 BRIGHT 上，该模型的 nDCG@10 得分为 18.0。
通过将查询与大语言模型（LLMs）生成的 Chain-of-Thought 推理进行增强，可以将性能提升多达 12.2 分。
````

#### [Synthetic Multimodal Question Generation](https://arxiv.org/pdf/2407.02233)
````
SMMQG 由五个步骤组成：

1. 从来源集合 S 中抽取一个种子来源。
2. 从种子来源中提取一个实体。
3. 使用步骤 2 中的实体作为查询，从 S 中检索候选来源。
4. 问题生成模型从候选来源中选择问题来源，并利用这些来源生成问题和答案。
5. 模型被要求验证生成的问题是否符合期望的问题风格和模式，以及生成的答案是否正确回答了问题。
````

#### [MOSS-MED: Medical Multimodal Model Serving Medical Image Analysis](https://dl.acm.org/doi/pdf/10.1145/3688005)

#### [TOOLSANDBOX: A Stateful, Conversational, Interactive Evaluation Benchmark for LLM Tool Use Capabilities](https://arxiv.org/pdf/2408.04682)
````
TOOLSANDBOX 包含了 有状态的工具执行（stateful tool execution），
工具之间的隐式状态依赖（implicit state dependencies between tools）
支持 on-policy 对话评估的内置用户模拟器（user simulator）
以及用于任意轨迹（arbitrary trajectory）上的中间和最终里程碑的动态评估策略。
````

#### [Se2: Sequential Example Selection for In-Context Learning](https://aclanthology.org/2024.findings-acl.312.pdf)
````
本文提出了一种在In-Context Learning场景的Sequential Example 选择方法，
它利用大型语言模型（LLM）对不同上下文（varying context）的反馈，
帮助捕捉示例之间的相互关系（inter-relationships）和顺序信息，显著丰富了ICL提示的上下文性和相关性。
````
![image](https://github.com/user-attachments/assets/82eb6a1b-a3bd-4219-b6d2-978162c85241)

#### [META-REWARDING LANGUAGE MODELS:Self-Improving Alignment with LLM-as-a-Meta-Judge](https://arxiv.org/pdf/2407.19594)
````
Meta-Rewarding 提升模型的判断（judge）和遵循指令（follow instructions）的能力。
LLMs 用来生成 response(作为actor）、judge response (作为judge)、以及对 judge 后的结果进行判断（作为Meta-judge）。
通过Llama-3-8B-Instruct 在 AlpacaEval 2 上的胜率从 22.9% 提高到 39.4%，在Arena-Hard上的胜率从 20.6% 提高到 29.1%。
````

#### [Direct Preference Knowledge Distillation for Large Language Models](https://arxiv.org/pdf/2406.19774)

#### [Attribute First, then Generate: Locally-attributable Grounded Text Generation](https://arxiv.org/pdf/2403.17104)
````
将传统的端到端生成过程(end-to-end generation process)分解为三个直观的步骤：内容选择(content selection)、
句子规划(sentence planning)和顺序句子生成(sentence generation)。
````

#### [PRewrite: Prompt Rewriting with Reinforcement Learning](https://arxiv.org/pdf/2401.08189)

#### [Reader-LM：将原始HTML转换为干净Markdown的小型语言模型](https://mp.weixin.qq.com/s/p2KrZKpcYnkc28geheInVA)

#### [Investigating Content Planning for Navigating Trade-offs in Knowledge-Grounded Dialogue](https://arxiv.org/pdf/2402.02077)
````
Knowledge-grounded 对话生成包含两个优化目标 specificity 和 attribution。
specificity指的是“符合对话流的一致性要求”。attribution指的是要忠实于参考文档。
将对话流程分为3步：
1.使用对话历史 x 和参考文档e，生成对话计划plan c=G(x, e)。
2.对话计划编辑器EQ迭代地修改plan c，生成 plan c_n = EQ(c, x, e)。
3.plan c_n 反馈给 G 生成输出响应 y = G(c_n, x, e)。
````
![image](https://github.com/user-attachments/assets/57259957-5c77-4cd5-bc28-459f9b089484)

#### [Heterogeneous LoRA for Federated Fine-tuning of On-Device Foundation Models](https://arxiv.org/pdf/2401.06432)

#### [“We Need Structured Output”: Towards User-centered Constraints on Large Language Model Output](https://storage.googleapis.com/gweb-research2023-media/pubtools/7754.pdf)
````
文章从用户层面出发，分析了对 LLMs 的 Structured Output的需求。
Low-level constraints：1.Structured Output 2.Multiple Choice 3.Length Constraints
High-level constraints：1.Semantic Constraints 2.Stylistic Constraints 3.Preventing Hallucination
````

#### [Large Language Models are Effective Text Rankers with Pairwise Ranking Prompting](https://storage.googleapis.com/gweb-research2023-media/pubtools/7775.pdf)

#### [ImageInWords:Unlocking Hyper-Detailed Image Descriptions](https://arxiv.org/pdf/2405.02793)
````
ImageInWords（IIW），一个精心设计的human-in-the-loop的注释框架(annotation framework)，
用于策划超详细(hyper-detailed)的图像描述，以及由此过程产生的新数据集。
````
![Uploading image.png…]()


