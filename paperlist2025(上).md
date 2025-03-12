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

#### [MCTS](https://www.bilibili.com/video/BV1CJ411A7K9/?spm_id_from=333.337.search-card.all.click)
https://github.com/Wangmerlyn/MCTS-GSM8k-Demo
