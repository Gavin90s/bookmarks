#### [Speculative RAG: Enhancing Retrieval Augmented Generation through Drafting](https://arxiv.org/pdf/2407.08223)
````
SPECULATIVE RAG
1.先由一个小的、蒸馏的专家语言模型并行产生的多个RAG草稿(drafts)。每个草稿(drafts)都是从不同的检索文档子集中生成的，
提供了多样化的证据视角，同时减少了每个草稿的输入Token数量。
2.利用一个更大的通用语言模型来高效验证对每个子集的理解，并减轻了长上下文中潜在的位置偏见(position bias)。
````

#### [Re-Invoke: Tool Invocation Rewriting for Zero-Shot Tool Retrieval](https://aclanthology.org/2024.findings-emnlp.270.pdf)
![image](https://github.com/user-attachments/assets/f83a05ae-2fa6-4e48-9a4d-2ad9ee448166)

#### [Harnessing Multimodal Large Language Models for Multimodal Sequential Recommendation](https://arxiv.org/pdf/2408.09698v2)

#### HiFiVFS: High Fidelity Video Face Swapping

#### [Tell Me More! Towards Implicit User Intention Understanding of Language Model Driven Agents](https://arxiv.org/pdf/2402.09205)
![image](https://github.com/user-attachments/assets/42335981-f0c8-4435-9686-4c1815e364ab)

#### [SUFFICIENT CONTEXT: A NEW LENS ON RETRIEVAL AUGMENTED GENERATION SYSTEMS](https://arxiv.org/pdf/2411.06037)

#### [Towards Knowledge Checking in Retrieval-augmented Generation: A Representation Perspective](https://assets.amazon.science/b9/82/66f769564f0282f4ea06e22ef42a/towards-knowledge-checking-in-retrieval-augmented-generation-a-representation-perspective.pdf)
**Internal Knowledge Checking**：用于判断LLMs内化的知识能否回复query。rep-PCA 达到 75% 的准确率，而 rep-Con 达到 79% 的准确率。

**Helpfulness Checking**("Helpfulness" here refers to the relevance of information to the query, information directly addressing the question is
considered helpful.)

>> **Informed Helpfulness Checking** when the LLM has internal knowledge about the query, check external information helpfulness. rep-PCA 达到 79% 的准确率，而 rep-Con 达到 81% 的准确率。

>> **Uninformed Helpfulness Checking** when the LLM lacks internal knowledge about the query, check external information helpfulness. rep-PCA 达到 81% 的准确率，而 rep-Con 达到 85% 的准确率。

>> **Contradiction Checking** to check if internal knowledge has any contradictions with the retrieved external information. rep-PCA 达到 91% 的准确率，而 rep-Con 达到 95% 的准确率。

#### [Token Pruning Optimization for Efficient Multi-Vector Dense Retrieval](https://assets.amazon.science/a3/46/81ba78eb4a4c9b90e5939b8df2bd/token-pruning-optimization-for-efficient-multi-vector-dense-retrieval.pdf)
![image](https://github.com/user-attachments/assets/a3a0506b-4a42-42f5-b97a-eebb1223ef47)

#### [Chain-of-Instructions: Compositional Instruction Tuning on Large Language Models](https://assets.amazon.science/89/c9/421aa5e04e39bb1aba36ae9cc4bf/chain-of-instructions-compositional-instruction-tuning-on-large-language-models.pdf)

#### [DeepSeek 入门到精通](https://docs.qq.com/pdf/DR05PZXNMd2RoUkFQ?)

#### [How to Talk to Language Models: Serialization Strategies for Structured Entity Matching](https://www.amazon.science/publications/how-to-talk-to-language-models-serialization-strategies-for-structured-entity-matching)

####  [AutoEval-ToD: Automated Evaluation of Task-oriented Dialog Systems](https://assets.amazon.science/ff/f0/596370ca4d1cbcd414b2a079aa77/autoeval-tod-automated-evaluation-of-task-oriented-dialog-systems.pdf)

#### [TOAD: Task-Oriented Automatic Dialogs with Diverse Response Styles](https://aclanthology.org/2024.findings-acl.494.pdf)
````
在对话系统中，**verbosity（冗长程度）和mirroring（镜像）**是两种用于控制系统响应风格的参数。以下是它们的含义和作用：
1. Verbosity（冗长程度）
Verbosity 表示系统响应的详细程度或信息量。它通常分为三个级别：
LV（Low Verbosity，低冗长程度）：响应简洁、直接，提供最少的信息，适合快速回答。
MV（Mid Verbosity，中冗长程度）：响应适中，提供足够的信息，但不会过于冗长。
HV（High Verbosity，高冗长程度）：响应详细、全面，提供丰富的信息，适合需要详细解释的场景。
示例：
LV：用户问：“今天天气如何？”
系统回答：“晴天。”
MV：用户问：“今天天气如何？”
系统回答：“今天是晴天，气温适中。”
HV：用户问：“今天天气如何？”
系统回答：“今天天气晴朗，气温在20-25摄氏度之间，非常适合户外活动。”
2. Mirroring（镜像）
Mirroring 是一种对话策略，表示系统在回答时是否模仿用户的语言风格、情感或表达方式。它的作用是让用户感到系统更自然、更贴近自己，从而增强对话的流畅性和亲和力。
M（Mirroring，镜像）：系统会模仿用户的语言风格，包括语气、词汇和表达方式。
非镜像：系统以固定的、标准的语言风格回答，不模仿用户。
示例：
用户：嘿，今天天气咋样？（口语化）
系统（Mirroring）：嘿，今天挺好的，晴天呢！（模仿口语化）
用户：嘿，今天天气咋样？
系统（非镜像）：今天天气晴朗，适合外出。
总结
Verbosity 控制回答的详细程度，从简洁到详细。
Mirroring 控制回答是否模仿用户的语言风格，增强对话的自然感和亲和力。
````

#### [Structured object language modeling (SoLM): Native structured objects generation conforming to complex schemas with self-supervised denoising](https://assets.amazon.science/9a/d6/456c33b44d2fb1453fc481b8f6d9/structured-object-language-modeling-solm-native-structured-objects-generation-conforming-to-complex-schemas-with-self-supervised-denoising.pdf)
````
self-supervised denoising training
````
![image](https://github.com/user-attachments/assets/e01426e9-d9d4-4549-b587-edd1a648423a)

#### [BFS-Prover: Scalable Best-First Tree Search for LLM-based Automatic Theorem Proving](https://arxiv.org/pdf/2502.03438)

#### [SFT Memorizes, RL Generalizes: A Comparative Study of Foundation Model Post-training](https://arxiv.org/pdf/2501.17161?)

####  [INTERNET OF AGENTS: WEAVING A WEB OF HETEROGENEOUS AGENTS FOR COLLABORATIVE INTELLIGENCE](https://arxiv.org/pdf/2407.07061)

#### [Agentic flow // let’s see what the agents can do](https://noailabs.medium.com/agentic-flow-lets-see-what-the-agents-can-do-e920c72f64b2)

#### [MCTS bilibili视频](https://www.bilibili.com/video/BV1CJ411A7K9/?spm_id_from=333.337.search-card.all.click)
https://github.com/Wangmerlyn/MCTS-GSM8k-Demo

#### [hugging face GRPO Trainer](https://huggingface.co/docs/trl/main/en/grpo_trainer)
- Generating completions
- computing the advantage
- estimating the KL divergence
- computing the loss.

#### [sgd, adam, adamw 讲解视频](https://www.bilibili.com/video/BV1NZ421s75D/?spm_id_from=333.337.search-card.all.click)

#### [From Online Softmax to FlashAttention](https://courses.cs.washington.edu/courses/cse599m/23sp/notes/flashattn.pdf)
````
FlashAttention，一种通过模仿在线 Softmax 的技巧来对自注意力计算进行分块的关键创新，
其核心目标是避免在 GPU 全局内存中存储中间的 logits 和注意力分数，从而融合整个多头注意力层，实现更快且更节省内存的自注意力计算。
````

#### [Long Context RAG Performance of LLMs](https://www.databricks.com/blog/long-context-rag-performance-llms)

#### [Chain of Agents: Large Language Models Collaborating on Long-Context Tasks](https://openreview.net/pdf?id=LuCLf4BJsr)
````
使用 Chain-of-Agents 解决 Long-Context 问题。
````

#### [Mastering Text Generation: Unveiling the Secrets of Decoding Strategies in Large Language Models](https://medium.com/@himankvjain/mastering-text-generation-unveiling-the-secrets-of-decoding-strategies-in-large-language-models-e89f91b9b7f1)

#### [旋转位置编码rope讲解](https://www.bilibili.com/video/BV1F1421B7iv/?spm_id_from=333.337.search-card.all.click)
https://www.zhihu.com/tardis/bd/art/647109286?source_id=1001

#### [Chain-of-Retrieval Augmented Generation](https://arxiv.org/pdf/2501.14342)

#### [RLTHF: Targeted Human Feedback for LLM Alignment](https://www.arxiv.org/pdf/2502.13417)

#### [vAttention: Dynamic Memory Management for Serving LLMs without PagedAttention](https://www.microsoft.com/en-us/research/wp-content/uploads/2024/05/vattention_arxiv24.pdf)

#### [Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention](https://arxiv.org/pdf/2502.11089)

#### [MMED-RAG: VERSATILE MULTIMODAL RAG SYSTEM FOR MEDICAL VISION LANGUAGE MODELS](https://arxiv.org/pdf/2410.13085)

#### [Inference-Time Scaling for Generalist Reward Modeling](https://huggingface.co/papers/2504.02495)

"Lost-in-the-middle issue" 在处理需要识别相关上下文信息的任务时，如文档问答、键值对索引等，大型语言模型（LLMs）对相关信息的位置非常敏感。
当相关信息位于输入提示的开头或结尾时，模型能够取得较好的效果。然而，当相关信息位于提示的中间部分时，模型的性能会显著下降。
这种现象表明，当前的语言模型在长输入上下文中不能稳健地利用信息，尤其是当相关信息出现在输入上下文的中间时，模型性能会显著下降。

#### [Know Your RAG: Dataset Taxonomy and Generation Strategies for Evaluating RAG Systems](https://aclanthology.org/2025.coling-industry.4.pdf)
````
基于(context, query)将问题分类为fact_single、summary、reasoning.
提出了基于 multi-steps 生成问题的方法。
````

#### [PIKE-RAG: sPecIalized KnowledgE and Rationale Augmented Generation](https://arxiv.org/pdf/2501.11551)
- factual question
- bridging question
- quantitative question
- comparative question
- summerizing question
- predictive question
- creative question

- Linkable-Reasoning Questions multi-step reasoning and linking across multiple sources.
- Predictive Questions extend beyond the available data, requiring inductive reasoning and structuring of retrieved facts into analyzable forms,
such as time series, for future-oriented predictions. 
- Creative Questions engage domain specific logic and creative problem-solving, encouraging the generation of innovative solutions by
synthesizing knowledge and identifying patterns or influencing factors.
![image](https://github.com/user-attachments/assets/2edd4719-6ae5-487d-8e0a-960587882e45)

- task decomposition & coordination
- knowledge retrieval
- knowledge organization
- knowledge centric reasoning
![image](https://github.com/user-attachments/assets/a3be1b18-a644-48aa-97c1-1c5435ca78b6)

#### [chatgpt temperature & top_p 参数怎么调节](https://community.openai.com/t/cheat-sheet-mastering-temperature-and-top-p-in-chatgpt-api/172683)

#### [Learning to Reason under Off-Policy Guidance](https://arxiv.org/pdf/2504.14945)

#### [Enhancing the Reasoning Ability of Multimodal Large Language Models via Mixed Preference Optimization](https://arxiv.org/pdf/2411.10442)

#### [ReSearch: Learning to Reason with Search for LLMs via Reinforcement Learning](https://arxiv.org/pdf/2503.19470)

#### [LLMS GET LOST IN MULTI-TURN CONVERSATION](https://arxiv.org/pdf/2505.06120)

#### [Chain-of-Reasoning: Towards Unified Mathematical Reasoning in Large Language Models via a Multi-Paradigm Perspective](https://arxiv.org/pdf/2501.11110)
本文引入了推理链（Chain-of-Reasoning, CoR），这是一个整合了多种推理范式（自然语言推理、算法推理和符号推理）的新型统一框架，以实现协同合作。

#### [Scaling Laws of Synthetic Data for Language Models](https://arxiv.org/pdf/2503.19551)

#### [https://www.microsoft.com/en-us/research/wp-content/uploads/2025/04/75_DEDUCE_DEDUCTIVE_CONSISTENCY.pdf](https://www.microsoft.com/en-us/research/wp-content/uploads/2025/04/75_DEDUCE_DEDUCTIVE_CONSISTENCY.pdf)
本文提出了一套生成 deductive consistency 推理数据的方法。
- 基准问题（benchmark problem）：是一个标准的、用于测试的数学或逻辑问题。
- 模板化（templatize）：将解决方案的结构提取出来，形成一个通用的模板。例如，一个数学问题的解决方案可能被提取成一系列的代码逻辑。
- 可执行的代码解决方案：将模板化的解决方案转化为可以运行的代码，用于生成新的问题实例。
- 更新变量值：前提数量（number of premises）：指推理过程中需要考虑的初始条件或已知信息的数量。推理步骤数量（number of hops）：指从前提到结论需要经过的推理步骤数量。
- 生成新的问题：通过改变变量值，生成一个与原始基准问题结构相同但数值不同的新问题。
![image](https://github.com/user-attachments/assets/2dec08c2-ab44-4ca4-9fee-8d890b1e9b32)

#### [Plan∗RAG: Efficient Test-Time Planning for Retrieval Augmented Generation](https://arxiv.org/pdf/2410.20753)

#### [Agentic Reasoning and Tool Integration for LLMs via Reinforcement Learning](https://www.microsoft.com/en-us/research/wp-content/uploads/2025/04/AgenticReasoning.pdf)

#### [The N+ Implementation Details of RLHF with PPO: A Case Study on TL;DR Summarization](https://arxiv.org/pdf/2403.17031)

#### [Search-o1: Agentic Search-Enhanced Large Reasoning Models](https://arxiv.org/pdf/2501.05366)
