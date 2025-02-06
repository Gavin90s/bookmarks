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

