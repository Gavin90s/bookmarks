#### [Speculative RAG: Enhancing Retrieval Augmented Generation through Drafting](https://arxiv.org/pdf/2407.08223)
````
SPECULATIVE RAG
1.先由一个小的、蒸馏的专家语言模型并行产生的多个RAG草稿(drafts)。每个草稿(drafts)都是从不同的检索文档子集中生成的，
提供了多样化的证据视角，同时减少了每个草稿的输入Token数量。
2.利用一个更大的通用语言模型来高效验证对每个子集的理解，并减轻了长上下文中潜在的位置偏见(position bias)。
````
