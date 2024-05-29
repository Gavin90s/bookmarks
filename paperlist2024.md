#### [AppAgent: Multimodal Agents as Smartphone Users](https://arxiv.org/pdf/2312.13771.pdf)
#### [QWEN TECHNICAL REPORT](https://arxiv.org/pdf/2309.16609.pdf)
#### [PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel](https://arxiv.org/pdf/2304.11277.pdf)
https://huggingface.co/docs/transformers/perf_infer_gpu_one
#### [Sparse Low-rank Adaptation of Pre-trained Language Models](https://arxiv.org/pdf/2311.11696.pdf)
#### [WEAK-TO-STRONG GENERALIZATION: ELICITING STRONG CAPABILITIES WITH WEAK SUPERVISION](https://arxiv.org/pdf/2312.15710.pdf)
#### [Towards Safer Generative Language Models: A Survey on Safety Risks, Evaluations, and Improvements](https://arxiv.org/pdf/2302.09270.pdf)
#### [Do LLMs Possess a Personality? Making the MBTI Test an Amazing Evaluation for Large Language Models](https://arxiv.org/pdf/2307.16180.pdf)
#### [ReAct: Synergizing Reasoning and Acting in Language Models](https://react-lm.github.io/)
#### [TOOLLLM: FACILITATING LARGE LANGUAGE MODELS TO MASTER 16000+ REAL-WORLD APIS](https://arxiv.org/pdf/2307.16789.pdf)
#### [SPOTLIGHT: MOBILE UI UNDERSTANDING USING VISION-LANGUAGE MODELS WITH A FOCUS](https://arxiv.org/pdf/2209.14927.pdf)
````
文章提出了一个视觉语言模型，可以理解截图（screenshot）中的页面元素的功能，可以理解页面元素。
可以完成多个手机 UI Task，如
1、Widget Captioning
2、Screen Summarization
3、Command Grounding
4、Tappability Prediction 
````
#### [Salient Phrase Aware Dense Retrieval: Can a Dense Retriever Imitate a Sparse One?](https://scontent-hkg4-2.xx.fbcdn.net/v/t39.8562-6/317501525_8752991914741698_9132568194086426998_n.pdf?_nc_cat=111&ccb=1-7&_nc_sid=e280be&_nc_ohc=6ohBcU_8Ok4AX_TLuvM&_nc_ht=scontent-hkg4-2.xx&oh=00_AfD9QUPISMV_icRdtK8F9aDjQqkN3m95idBRFA_RZS9ohA&oe=65C4F04E)
````
使用 BM25 来作为Teacher 模型，训练 Salient Phrase Aware Dense Retrieval, 从而提升模型的zero-shot能力
````
<img width="806" alt="image" src="https://github.com/Gavin90s/bookmarks/assets/8350994/038a7265-2cd0-48fb-afc2-3bb3b05a23c0">

#### [The Effect of Context-aware LLM-based NPC Conversations on Player Engagement in Role-playing Video Games](https://projekter.aau.dk/projekter/files/536738243/The_Effect_of_Context_aware_LLM_based_NPC_Dialogues_on_Player_Engagement_in_Role_playing_Video_Games.pdf)

#### [AGENTBENCH: EVALUATING LLMS AS AGENTS](https://arxiv.org/pdf/2308.03688.pdf)
````
AGENTBENCH 包含了8个不同的应用环境，用于评估LLM-as-Agent的推理能力(reasoning abilities)
和决策能力(decision-making abilities）。开源的LLMs在长上下文理解，决策和指令遵循能力上存在不足，
不利用开发LLM agents。在代码和高质量的多轮对齐数据集上训练，可以提升agent的能力。
````
![image](https://github.com/Gavin90s/bookmarks/assets/8350994/06485259-eff9-4cb7-a57a-b8c3d3d204d2)

#### [Best Practices for LLM Evaluation of RAG Applications](https://www.databricks.com/blog/LLM-auto-eval-best-practices-RAG)
````
评价回复生成质量的指标：有用性(helpfulness)、相关性(relevance)、准确性(accuracy)、深度(depth)、
创造性(creativity)和细节水平(level of detail)进行评估。
````
````
自动化评测分为3步：
1、生成评测数据集。从Databricks文档中的100个问题和上下文创建了一个数据集。上下文表示与问题相关的（大块）文档。
2、生成答案表。使用评估数据集，在不同的语言模型生成答案，并将 "问题-上下文-答案对" 存储在名为“答案表”的数据集中。
在这项研究中，我们使用了GPT-4、GPT-3.5、Claude-v1、Llama2-70b-chat、Vicuna-33b和mpt-30b-chat。
3、生成分数。在给出答案的情况下，我们使用各种LLM来生成分数并对分数进行推理。这些分数是Correctness（加权60%）、
Comprehensiveness（加权20%）、Readability（加权20%）的综合得分。
````
![image](https://github.com/Gavin90s/bookmarks/assets/8350994/ba177595-8319-4063-905b-529e794157da)

#### [ROLELLM: BENCHMARKING, ELICITING, AND ENHANCING ROLE-PLAYING ABILITIES OF LARGE LANGUAGE MODELS](https://openreview.net/pdf?id=i4ULDEeBss)
````
本文推出了RoleLLM，并开源了rolebench基准测试。
RoleLLM包括四个阶段：
（1）Role Profile Construct 100个角色；
（2） Context-Based Instruction Generation（Context Instruction）生成 instruction；
（3）Role Prompting用GPT（RoleGPT）进行说话风格模仿；
（4）Role-Conditioned Instruction Tuning（RoCIT）和角色定制用于微调开源模型。
````
![image](https://github.com/Gavin90s/bookmarks/assets/8350994/27dfe443-dde4-4e99-adfa-e74f73e1a463)

#### [Sparse, Dense, and Attentional Representations for Text Retrieval](https://arxiv.org/pdf/2005.00181.pdf)
````
ME-Bert首次从理论和实验两个角度论证了当document的长度越长时，单个向量的固定维度（bert为768）难以有效的表征doc的语义信息。
想要获得更好的效果，必须扩大表征的维度或者增加表征向量的个数。由于增加个数对检索性能的影响要小于扩大维度，
因此作者最终选择以多向量的表征方式来编码单个doc。具体实现起来也很简单，作者选取了每个doc前8个token的向量作为最终的doc表征，
在训练过程中，计算query向量与这8个向量的相似度并取最大值作为最终得分，用cross-entropy损失去优化模型参数。
当然也做了一些随机负例与难负例的采样。
````
#### [InDi: Informative and Diverse Sampling for Dense Retrieval](https://assets.amazon.science/39/b7/5ce986a64af6a9c21d163aedf307/indi-informative-and-diverse-sampling-for-dense-retrieval.pdf)

#### [CRUD-RAG: A Comprehensive Chinese Benchmark for Retrieval-Augmented Generation of Large Language Models](https://arxiv.org/pdf/2401.17043.pdf)

#### [Benchmarking Large Language Models in Retrieval-Augmented Generation](https://arxiv.org/pdf/2309.01431.pdf)

#### [UHGEval: Benchmarking the Hallucination of Chinese Large Language Models via Unconstrained Generation](https://arxiv.org/pdf/2311.15296.pdf)

#### [LAVE: LLM-Powered Agent Assistance and Language Augmentation for Video Editing](https://arxiv.org/pdf/2402.10294.pdf)
````
基于LLMs的视频剪辑工具。
````
![image](https://github.com/Gavin90s/bookmarks/assets/8350994/091bc756-1118-4d00-a786-f44e7dcb1a52)

#### [AGENTTUNING: ENABLING GENERALIZED AGENT ABILITIES FOR LLMS](https://arxiv.org/pdf/2310.12823.pdf)
````
本质上就是用imitation learning代替reinforcement learning的工作. 
众所周知,各家大模型的语义理解能力不相上下, 和GPT-4差就差在了推理能力上. 要提高推理能力, 就需要在强化学习上下功夫.
具体的, Reward Modal有很高的标注成本; 基于PPO的RLHF和KL散度具有很高的训练成本; 这里需要大量人力财力以及时间的打磨.
传统任务上使用imitation learning最大的困难在于高质量的轨迹标注, 相比于reinforcement learning的reward标注高出了好几个数量级.
本文巧妙地使用了GPT-4做imitation learning的轨迹标注, 节省了标记成本以及训练成本.
使用交叉验证的方式混合数据集来替代KL散度以保证泛化能力, 进一步节省了训练成本.
这么好的降本增效方法一定会被大家广泛采纳.
PS. 有点naive了, 具体到各个数据集还是需要很多工程上的工作的,这才是本文的工作量.
https://huggingface.co/datasets/THUDM/AgentInstruct/tree/main/data
````
#### [Efficient Tool Use with Chain-of-Abstraction Reasoning](https://arxiv.org/pdf/2401.17464.pdf)

#### [QuRating: Selecting High-Quality Data for Training Language Models](https://arxiv.org/pdf/2402.09739.pdf)

#### [Attribute Structuring Improves LLM-Based Evaluation of Clinical Text Summaries](https://arxiv.org/pdf/2403.01002.pdf)

#### [Reformatted Alignment](https://arxiv.org/pdf/2402.12219.pdf)
````
ReAlign 主要用于知识密集型任务，可以显著提升通用对齐能力(general alignment ability),
数学推理能力(math reasoning),事实性(factuality),
和LLMs回复生成的可读性(readability of the LLMs)。
````
![image](https://github.com/Gavin90s/bookmarks/assets/8350994/e303e75a-9fdd-484b-94c7-e32ed0071ee8)

#### [QuRating: Selecting High-Quality Data for Training Language Models](https://arxiv.org/pdf/2402.09739.pdf)
````
QuRating 一种用于选择预训练数据的方法，该方法能够捕捉人类直觉感知到的文本抽象特质（abstract qualities of texts）。
在本文中，我们研究了四种特质——写作风格(qualities—writing style)、所需的专业知识(required expertise)、
事实和琐事(facts & trivia)、以及教育价值(educational value)。 
````

#### [Synthetic Data (Almost) from Scratch: Generalized Instruction Tuning for Language Models](https://arxiv.org/pdf/2402.13064.pdf)
````
GLAN 仅利用预先策划(pre-curated)的人类知识(knowledge)和能力(capabilities)分类体系(taxonomy)作为输入，
并生成跨所有学科的大规模合成指令(synthetic instruction)数据。具体来说，通过将人类知识和能力分解为不同领域、
子领域，并最终分解为不同学科，半自动地构建分类体系， LLMs 发挥了促进作用。
随后，我们为每个学科生成了一个全面的科目列表，并继续为每个科目设计量身定制的教学大纲，同样利用了 LLMs。
凭借教学大纲中每节课程详细描述的精细关键概念，我们能够生成多样化指令，涵盖人类知识和技能的整个领域。
````
![image](https://github.com/Gavin90s/bookmarks/assets/8350994/bc2f516e-65be-4f01-8451-b526cf87de95)

#### [Balanced Data Sampling for Language Model Training with Clustering](https://arxiv.org/pdf/2402.14526.pdf)

#### [MathScale: Scaling Instruction Tuning for Mathematical Reasoning](https://arxiv.org/pdf/2403.02884.pdf)

#### [Ziya2: Data-centric Learning is All LLMs Need](https://arxiv.org/pdf/2311.03301.pdf)

#### [Improving Text Embeddings with Large Language Models](https://arxiv.org/pdf/2401.00368.pdf)
````
它还是沿用SimCSE和E5这样的双塔结构+InfoNCE对比学习训练，但是加入了两味基于大模型的新成分：
1、用GPT-4合成高质量训练数据。
2、使用mistral-7b作为backbone进行微调，[EOS]状态作为text embedding。
````
#### [Text Embeddings by Weakly-Supervised Contrastive Pre-training](https://arxiv.org/pdf/2212.03533.pdf)

#### [Scaling Laws for Downstream Task Performance of Large Language Models](https://arxiv.org/pdf/2402.04177.pdf)

#### [Toward Informal Language Processing: Knowledge of Slang in Large Language Models](https://assets.amazon.science/bf/e7/91b07f7d4052ac3d96c960545ff8/toward-informal-language-processing-knowledge-of-slang-in-large-language-models.pdf)
````
使用LLMs进行俚语检测。
````
#### [Semantic matching for text classification with complex class descriptions](https://assets.amazon.science/91/de/33bafd8e40669a8c5e22563d7e18/semantic-matching-for-text-classification-with-complex-class-descriptions.pdf)
![image](https://github.com/Gavin90s/bookmarks/assets/8350994/99355196-a217-4829-9332-550018ca2079)

#### [Identifying Shopping Intent in Product QA for Proactive Recommendations](https://assets.amazon.science/b6/58/4ab7e9bb4fd3b1080e2608310833/identifying-shopping-intent-in-product-qa-for-proactive-recommendations.pdf)
````
在语音助手做商品推荐
````

#### [Prompt Perturbation Consistency Learning for Robust Language Models](https://assets.amazon.science/16/e1/b790e6c647aea33749aa5bdf2d51/prompt-perturbation-consistency-learning-for-robust-language-models.pdf)
````
通过 Prompt Perturbation 提升 LLMs 在 slot filling 和 intent classification 的效果。
````

#### [Query Expansion by Prompting Large Language Models](https://arxiv.org/pdf/2305.03653)
````
通过 q + LLM(prompt) 输出作为 Query Expansion 来代替传统的Pseudo-Relevance Feedback (PRF) query扩展方法。
````
![image](https://github.com/Gavin90s/bookmarks/assets/8350994/5b5d829e-12ab-400c-8a92-2e58a9af1107)

#### [Improving RAG with Query expansion & reranking models](https://aksdesai1998.medium.com/improving-rag-with-query-expansion-reranking-models-31d252856580)
#### [Query2doc: Query Expansion with Large Language Models](https://arxiv.org/pdf/2303.07678)
