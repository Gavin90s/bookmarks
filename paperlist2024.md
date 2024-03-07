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
````

