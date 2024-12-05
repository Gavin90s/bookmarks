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
![image](https://github.com/user-attachments/assets/d1a2d919-9732-43c3-b26a-24e8b3a0b5fc)

#### [CodecLM: Aligning Language Models with Tailored Synthetic Data](https://arxiv.org/pdf/2404.05875)
````
通过 seed instruction 生成 synthetic instruction，来增加训练数据丰富度。
````

#### [Prompt Cache: Modular Attention Reuse for Low-Latency Inference](https://mlsys.org/virtual/2024/poster/2643)

#### [RewriteLM: An Instruction-Tuned Large Language Model for Text Rewriting](https://arxiv.org/pdf/2305.15685)

#### [Learning to Rewrite Prompts for Personalized Text Generation](https://storage.googleapis.com/gweb-research2023-media/pubtools/7614.pdf)

#### [Improving Text Embeddings with Large Language Models](https://arxiv.org/pdf/2401.00368)

#### [Introducing Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval)

#### [AGENT AI: SURVEYING THE HORIZONS OF MULTIMODAL INTERACTION](https://arxiv.org/pdf/2401.03568)

#### [SKELETON-OF-THOUGHT: PROMPTING LLMs FOR EFFICIENT PARALLEL GENERATION](https://arxiv.org/pdf/2307.15337)
````
针对LLM解码过程计算利用率低和延时大的问题，在数据层面，通过提示词引导LLM自主规划出答案提纲，
并利用提纲中不同部分的可并行性，解决LLM解码过程计算单元利用率低和延时大的问题。
````
![image](https://github.com/user-attachments/assets/cc70c768-2932-4145-89bf-40cd7bdd614b)

#### [Table Meets LLM: Can Large Language Models Understand Structured Table Data? A Benchmark and Empirical Study](https://www.microsoft.com/en-us/research/uploads/prod/2023/12/wsdm24-SUC.pdf)

#### [AIOpsLab: A Holistic Framework for Evaluating AI Agents for Enabling Autonomous Cloud](https://www.microsoft.com/en-us/research/uploads/prod/2024/10/AIOpsLab-6705feab5dcdb.pdf)

#### [End-to-End Automatic Speech Recognition](https://www.microsoft.com/en-us/research/uploads/prod/2024/08/SPS2024_E2E-ASR.pdf)
````
以大型语言模型 (LLM) 为中心的 AI 可能是 E2E ASR 的下一个趋势。
基于 LLM 的模型，如 VALL-E、SpeechX、VioLA、AudioPaLM、SALMONN、WavLLM 和 Speech-LLM，正在推动语音理解和生成的边界。
````

#### [Learning to Generate Answers with Citations via Factual Consistency Models](https://assets.amazon.science/b9/c2/2a961e5849c8b3d2b5037920a35e/learning-to-generate-answers-with-citations-via-factual-consistency-models.pdf)
````
弱监督微调(Weakly-supervised fine-tuning) 是一种在有限标注资源下提高模型性能的有效方法，
它通过利用弱标签和自我生成的伪标签来减少对精确标注数据的需求，并采用多种策略来提高模型的鲁棒性和准确性。
事实一致性模型（Factual Consistency Models）的核心目标是确保机器生成文本（如摘要、翻译或回答）与源数据或已知事实一致，不产生虚假信息。
````
![image](https://github.com/user-attachments/assets/1ac92996-9110-4b33-a153-964a07833788)


#### [Controlled Automatic Task-Specific Synthetic Data Generation for Hallucination Detection](https://assets.amazon.science/c8/da/d717c3d447d8a7944c85dccda43d/controlled-automatic-task-specific-synthetic-data-generation-for-hallucination-detection.pdf)

#### [Meta Knowledge for Retrieval Augmented Large Language](https://assets.amazon.science/c8/94/ded22c51481491a0ac359d3b87e8/meta-knowledge-for-retrieval-augmented-large-language-models.pdf)
````
创新点：
1、用合成问答生成代替传统文档分块框架。 传统 RAG 管道严重依赖文档分块，但这会导致信息丢失，并使检索模型难以提取相关信息。
 这项研究提出用合成问答生成来代替文档分块，从而减轻信息丢失，并提高检索的准确性。
2、引入元知识摘要 (MK Summary) 以增强查询。 为了进一步提高零样本搜索增强，该研究引入了 MK Summary 的概念。
MK Summary 是基于元数据的文档集群内容的高级摘要。 在推理时，用户查询会根据感兴趣的元数据动态增强，从而为该用户提供定制的响应。
这使得检索器能够跨多个文档进行推理，从而提高搜索结果的深度、覆盖范围和相关性。
````
![image](https://github.com/user-attachments/assets/22b9bb73-f017-4bae-b397-0d57172a8d7c)

