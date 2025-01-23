#### [SELF-CONSISTENCY IMPROVES CHAIN OF THOUGHT REASONING IN LANGUAGE MODELS](https://openreview.net/pdf?id=1PL1NIMMrw)
使用称为self-consistency的decoding strategy代替naive greedy decoding 逻辑推理能力。
<img width="676" alt="image" src="https://github.com/Gavin90s/bookmarks/assets/8350994/dd53305c-0ad6-45bf-9a56-5820aa16965f">

#### [Improving In-Context Few-Shot Learning via Self-Supervised Training](https://arxiv.org/pdf/2205.01703.pdf)
在预训练之后，下游任务微调之前，使用自监督学习(self-supervision) 添加一些预训练任务，可以提升模型的 In-Context Few-Shot Learning。 
<img width="972" alt="image" src="https://github.com/Gavin90s/bookmarks/assets/8350994/1158e1f5-8341-46dd-b8f7-2ef876312a14">

#### 提取句向量的transformer (sentence transformer)


sentence transformer（bi-encoders）


<img width="669" alt="image" src="https://github.com/Gavin90s/bookmarks/assets/8350994/9c31cf6c-2b8e-46b6-a765-3067bb8e76e1">


交叉编码器（cross-encoder）


<img width="659" alt="image" src="https://github.com/Gavin90s/bookmarks/assets/8350994/68933a47-146d-48bb-a059-d6dffe161de4">


用Cross-Encoder对所有挖掘出的段落进行分类, 使用Cross-Encoder帮助Bi-Encoder挖掘困难负例。

#### Constitutional AI
Constitutional AI is a technique that aims to imbue systems with “values” defined by a “constitution”³. This makes the behavior of systems both easier to understand and simpler to adjust as needed³. The system uses a set of principles to make judgments about outputs, hence the term “Constitutional”

#### [大规模语言模型从理论到实践](https://intro-llm.github.io/chapter/LLM-TAP.pdf)

#### [Red Teaming Language Models to Reduce Harms: Methods, Scaling Behaviors, and Lessons Learned](https://arxiv.org/pdf/2209.07858.pdf)

#### [TRAIN SHORT, TEST LONG: ATTENTION WITH LINEAR BIASES ENABLES INPUT LENGTH EXTRAPOLATION](https://arxiv.org/pdf/2108.12409.pdf)

#### [Adafactor: Adaptive Learning Rates with Sublinear Memory Cost](https://arxiv.org/pdf/1804.04235.pdf)

#### [Scaling Laws and Interpretability of Learning from Repeated Data](https://arxiv.org/pdf/2205.10487.pdf)

#### [GPT-4 Technical Report](https://arxiv.org/pdf/2303.08774.pdf)

#### [TruthfulQA: Measuring How Models Mimic Human Falsehoods](https://arxiv.org/pdf/2109.07958.pdf)

#### [Generator-Retriever-Generator: A Novel Approach to Open-domain](https://arxiv.org/pdf/2307.11278.pdf)

#### [A Synthetic Data Generation Framework for Grounded Dialogues](https://aclanthology.org/2023.acl-long.608v2.pdf)

#### [GLM-Dialog: Noise-tolerant Pre-training for Knowledge-grounded Dialogue Generation](https://arxiv.org/pdf/2302.14401v1.pdf)

#### [RLAIF: Scaling Reinforcement Learning from Human Feedback with AI Feedback](https://arxiv.org/pdf/2309.00267.pdf)

#### [PaLM 2 Technical Report](https://arxiv.org/pdf/2305.10403.pdf)

#### [IncreLoRA: Incremental Parameter Allocation Method for Parameter-Efficient Fine-tuning](https://arxiv.org/pdf/2308.12043.pdf)

#### [GLM-Dialog: Noise-tolerant Pre-training for Knowledge-grounded Dialogue Generation](https://arxiv.org/pdf/2302.14401v1.pdf)

#### [MULTITASK PROMPTED TRAINING ENABLES ZERO-SHOT TASK GENERALIZATION](https://arxiv.org/pdf/2110.08207.pdf)

#### [Training language models to follow instructions with human feedback](https://arxiv.org/pdf/2203.02155.pdf)

#### [Efficient Open Domain Multi-Hop Question Answering with Few-Shot Data Synthesis](https://arxiv.org/pdf/2305.13691.pdf)

#### [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/pdf/2307.08691.pdf)

#### [TruthfulQA: Measuring How Models Mimic Human Falsehoods](https://arxiv.org/pdf/2109.07958.pdf)

#### [EXTENDING CONTEXT WINDOW OF LARGE LANGUAGE MODELS VIA POSITION INTERPOLATION](https://arxiv.org/pdf/2306.15595.pdf)

#### [ByteTransformer: A High-Performance Transformer Boosted for Variable-Length Inputs](https://arxiv.org/pdf/2210.03052.pdf)

#### [D4: Improving LLM Pretraining via Document De-Duplication and Diversification](https://arxiv.org/pdf/2308.12284.pdf)

#### [复旦大学：大规模语言模型从理论到实践](https://intro-llm.github.io/chapter/LLM-TAP.pdf)

#### [Instruction Tuning for Large Language Models: A Survey](https://arxiv.org/pdf/2308.10792.pdf)

#### [C-Pack: Packaged Resources To Advance General Chinese Embedding](https://arxiv.org/pdf/2309.07597.pdf)

#### [New Intent Discovery with Pre-training and Contrastive Learning](https://arxiv.org/pdf/2205.12914.pdf)

#### [Enhancing Chat Language Models by Scaling High-quality Instructional Conversations](https://arxiv.org/pdf/2305.14233.pdf)

#### [Making Retrieval-Augmented Language Models Robust to Irrelevant Context](https://openreview.net/pdf?id=ZS4m74kZpH)

#### [Self-Adaptive In-Context Learning: An Information Compression Perspective for In-Context Example Selection and Ordering](https://arxiv.org/pdf/2212.10375.pdf)

#### [FLAMES: BENCHMARKING VALUE ALIGNMENT OF CHINESE LARGE LANGUAGE MODELS](https://arxiv.org/pdf/2311.06899.pdf)

#### [Continuous Training and Fine-tuning for Domain-Specific Language Models in Medical Question Answering](https://arxiv.org/pdf/2311.00204.pdf)

#### [System 2 Attention (is something you might need too)](https://arxiv.org/pdf/2311.11829.pdf)

#### [Query2doc: Query Expansion with Large Language Models](https://arxiv.org/pdf/2303.07678.pdf)
````
在BM25检索前，先使用 LLMs 对 query 进行 expansion. 相对于 BM25， query expansion 能够提升效果相对 3%～ 15%。
````

#### [Learning to Retrieve In-Context Examples for Large Language Models](https://arxiv.org/pdf/2307.07164.pdf)

#### [Learning To Retrieve Prompts for In-Context Learning](https://arxiv.org/pdf/2112.08633.pdf)
````
用BM25检索到candidates，然后使用LM scorer = f(y｜x, prompt) 打分，得分高的当作正样本，得分低的当作hard negatives,
然后训练dense retriever。
````
![image](https://github.com/Gavin90s/bookmarks/assets/8350994/dab86d4b-0d43-4c1e-bd07-4e4aab242b09)

#### [Unified Demonstration Retriever for In-Context Learning](https://arxiv.org/pdf/2305.04320.pdf)
````
训练一个 Unified Demonstration Retriever（UDR），而不是 task-specific retrievers for several tasks separately。
训练 UDR, 使用语言模型将训练数据整理成unified listwise ranking formulation 训练 Bi-encoder embeddin 特征提取。
Retrieval Methods 包括：BM25、SBert(Sentence Bert)、Instructor、DR-Target、EPR、UDR。
````
````
UDR 的loss包含2个部分：1、rank list 排序的loss。2、constractive loss。
````
![image](https://github.com/Gavin90s/bookmarks/assets/8350994/2a1e47cc-f0a9-4c46-8afe-2c427317b95a)
![image](https://github.com/Gavin90s/bookmarks/assets/8350994/1a56ebc2-2865-4027-b45b-775f3f139a4a)


![image](https://github.com/Gavin90s/bookmarks/assets/8350994/8a73443d-6c60-4319-9e2f-e8aaf92deeff)

#### [SELF-INSTRUCT: Aligning Language Models with Self-Generated Instructions](https://arxiv.org/pdf/2212.10560.pdf)

#### [Fine-Tuning LLaMA for Multi-Stage Text Retrieval](https://arxiv.org/pdf/2310.08319.pdf)

#### [Large Language Models Need Holistically Thought in Medical Conversational QA](https://arxiv.org/pdf/2305.05410.pdf)

#### [Efficient Fine-tuning Large Language Models for Knowledge-Aware Response Planning](https://assets.amazon.science/a2/5f/eb28bdfe4bff878c240003fe018f/efficient-fine-tuning-large-language-models-for-knowledge-aware-response-planning.pdf)
````
亚马逊alex团队的论文，提出按2个阶段来finetune LLMs提升效果，先进行f(q,c)的finetune，然后f(q）finetune。
````
![image](https://github.com/Gavin90s/bookmarks/assets/8350994/95dd986b-8a80-48c0-bd12-48538348ede0)

#### [Data-Efficient Alignment of Large Language Models with Human Feedback Through Natural Language](https://arxiv.org/pdf/2311.14543.pdf)
````
使用模型来对标注提供一些反馈。在这篇文章里面，提出了一种 critique-and-revision (CnR) model 用来给数据标注提供一些critiques,
包括正面和负面的反馈，并生成改进以后的回复答案。
````
![image](https://github.com/Gavin90s/bookmarks/assets/8350994/82f96f45-e949-4be3-bac9-1e53e9f1b04f)

#### [PhraseSumm: Abstractive Short Phrase Summarization](https://assets.amazon.science/60/9a/67f0f2c24ad39c2c46b5f527b3c1/phrasesumm-abstractive-short-phrase-summarization.pdf)

#### [Topic Knowledge based Controlled Generation for Long Documents using Retrieval-based Language Models](https://assets.amazon.science/f2/22/aba0563b43b1be5ef6d6bdd03d50/topic-knowledge-based-controlled-generation-for-long-documents-using-retrieval-based-language-models.pdf)

#### [Integrating Summarization and Retrieval for Enhanced Personalization via Large Language Models](https://assets.amazon.science/2e/c9/53f2570d413eba5a5ca95c812c67/integrating-summarization-and-retrieval-for-enhanced-personalization-via-large-language-models.pdf)

#### [Automated Few-shot Classification with Instruction-Finetuned Language Models](https://assets.amazon.science/8e/85/67de5e2b43d4a19e979cc19a4ac1/automated-few-shot-classification-with-instruction-finetuned-language-models.pdf)

#### [PROTEGE: Prompt-based Diverse Question Generation from Web Articles](https://assets.amazon.science/4c/31/6432e28e4a71900d5784aba175ad/protege-prompt-based-diverse-question-generation-from-web-articles.pdf)

#### [Too much of product information : Don’t worry, let’s look for evidence!](https://assets.amazon.science/47/eb/7bf0577c482782b5ac31a25794b3/too-much-of-product-information-dont-worry-lets-look-for-evidence.pdf)

#### [Measuring and mitigating dialog-to-API constraint violations of in-context learning](https://assets.amazon.science/82/6f/6e3316d540f8acde5fdb5fa42ff1/measuring-and-mitigating-dialog-to-api-constraint-violations-of-in-context-learning.pdf)

#### [INVITE: A testbed of automatically generated invalid questions to evaluate large language models for hallucinations](https://assets.amazon.science/3f/73/dd51d8824cc0b9b6d3cf3a263908/invite-a-testbed-of-automatically-generated-invalid-questions-to-evaluate-large-language-models-for-hallucinations.pdf)

#### [Improving Contextual Query Rewrite for Conversational AI Agents through User-preference Feedback Learning](https://assets.amazon.science/e8/83/3f5985ed48da948959d43e11ec9f/improving-contextual-query-rewrite-for-conversational-ai-agents-through-user-preference-feedback-learning.pdf)

#### [Context Aware Query Rewriting for Text Rankers using LLM](https://arxiv.org/pdf/2308.16753v1.pdf)

#### [Llama 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/pdf/2307.09288.pdf)

#### [Code Llama: Open Foundation Models for Code](https://arxiv.org/pdf/2308.12950.pdf?fbclid=IwAR1lnwBO8AEH9CcL7HSpR4ZypJgzDTKtEnUPoyn81SBwvn3pcFcxUqAgZmE&trk=public_post_comment-text)

#### [Augmented Language Models: a Survey](https://arxiv.org/pdf/2302.07842.pdf)

#### [Evaluating Open-Domain Question Answering in the Era of Large Language Models](https://arxiv.org/pdf/2305.06984.pdf)

#### [Unified Contextual Query Rewriting](https://assets.amazon.science/83/56/ec1598c841c198a415bc05153d19/unified-contextual-query-rewriting.pdf)

#### [Entity Contrastive Learning in a Large-Scale Virtual Assistant System](https://assets.amazon.science/c9/cb/a03f4e6241779d3641767b27fd7e/entity-contrastive-learning-in-a-large-scale-virtual-assistant-system.pdf)

#### [RobustQA: Benchmarking the Robustness of Domain Adaptation for Open-Domain Question Answering](https://assets.amazon.science/2e/9e/b5c88464449d8acf6d77b02784ef/robustqa-benchmarking-the-robustness-of-domain-adaptation-for-open-domain-question-answering.pdf)

#### [Generating Summaries with Controllable Readability Levels](https://assets.amazon.science/c5/e1/be005cb346ad98ac346a0ecffe19/generating-summaries-with-controllable-readability-levels.pdf)

#### [InstructPTS: Instruction-Tuning LLMs for Product Title Summarization](https://assets.amazon.science/f5/bb/857ae49241318362917b8e7bdcd0/instructpts-instruction-tuning-llms-for-product-title-summarization.pdf)

#### [Characterizing Attribution and Fluency Tradeoffs for Retrieval-Augmented Large Language Models](https://arxiv.org/pdf/2302.05578.pdf)

#### [Generative Information Retrieval (slides)](https://docs.google.com/presentation/d/19lAeVzPkh20Ly855tKDkz1uv-1pHV_9GxfntiTJPUug/edit?pli=1#slide=id.g22efd0a58a4_2_0)

#### [Rethinking the Role of Token Retrieval in Multi-Vector Retrieval](https://arxiv.org/pdf/2304.01982.pdf)

#### [Retrieval-based Language Models and Applications, 陈丹琦](https://acl2023-retrieval-lm.github.io/)

#### [Unsupervised Dense Information Retrieval with Contrastive Learning](https://arxiv.org/pdf/2112.09118.pdf)
```` Facebook dense retrieval use Unsupervised Contrastive Learning ````

#### [Llama Guard: LLM-based Input-Output Safeguard for Human-AI Conversations](https://arxiv.org/pdf/2312.06674.pdf)

#### [MoQA: Benchmarking Multi-Type Open-Domain Question Answering](https://aclanthology.org/2023.dialdoc-1.2.pdf)

#### [RECITATION-AUGMENTED LANGUAGE MODELS](https://arxiv.org/pdf/2210.01296.pdf)

#### [Rethinking the Role of Token Retrieval in Multi-Vector Retrieval](https://arxiv.org/pdf/2304.01982.pdf)

#### [PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel](https://arxiv.org/pdf/2304.11277.pdf)

#### [Tool Documentation Enables Zero-Shot Tool-Usage with Large Language Models](https://arxiv.org/pdf/2308.00675.pdf)

#### [Unsupervised Event Chain Mining from Multiple Documents](https://dl.acm.org/doi/pdf/10.1145/3543507.3583295)

#### [q2d: Turning Questions into Dialogs to Teach Models How to Search](https://arxiv.org/pdf/2304.14318.pdf)

#### [AppAgent: Multimodal Agents as Smartphone Users](https://arxiv.org/pdf/2312.13771.pdf)

#### [STRUM: Extractive Aspect-Based Contrastive Summarization](https://storage.googleapis.com/gweb-research2023-media/pubtools/pdf/d758fb82e22be0a5fae06063d57641c787085a90.pdf)
