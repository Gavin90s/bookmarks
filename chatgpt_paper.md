为了写本ChatGPT笔记，过去两个月翻了大量中英文资料/paper(中间一度花了大量时间去深入RL)，大部分时间读的更多是中文资料，2月最后几天读的更多是英文paper，正是2月底这最后几天对ChatGPT背后技术原理的研究才真正进入状态(后还组建了一个“ChatGPT之100篇论文阅读组”，我和10来位博士、业界大佬从23年2.27日起100天读完ChatGPT相关技术的100篇论文)，当然 还在不断深入，由此而感慨： 

读的论文越多，你会发现大部分人对ChatGPT的技术解读都是不够准确或全面的，毕竟很多人没有那个工作需要或研究需要，去深入了解各种细节
因为100天100篇这个任务，让自己有史以来一篇一篇一行一行读100篇，​之前看的比较散 不系统 抠的也不细
比如回顾“Attention is all you need”这篇后，对优化博客内的Transformer笔记便有了很多心得
总之，读的论文越多，博客内相关笔记的质量将飞速提升 自己的技术研究能力也能有巨大飞跃

且考虑到为避免上篇文章篇幅太长而影响完读率，故把这100论文的清单抽取出来独立成本文

Attention Is All You Need，Transformer原始论文
GPT：Improving Language Understanding by Generative Pre-Training
GPT2：Language Models are Unsupervised Multitask Learners
GPT3原始论文：Language Models are Few-Shot Learners
ICL原始论文
Evaluating Large Language Models Trained on Code，Codex原始论文
预测当前序列的最后一个词时 可以选取概率最大的词(softmax最高的值)，但没法全局最优且不具备多样性，当然 可以使用束搜索 一次性获取多个解
论文中用的是核采样，预测的各个词根据概率从大到小排序，选取前些个概率加起来为95%的词
CoT原始论文：Chain-of-Thought Prompting Elicits Reasoning in Large Language Models
28 Jan 2022 · Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed Chi, Quoc Le, Denny Zhou
也从侧面印证，instructGPT从22年1月份之前 就开始迭代了
Training language models to follow instructions with human feedback
InstructGPT原始论文

RLHF原始论文
PPO原始论文
《Finetuned Language Models Are Zero-Shot Learners》，2021年9月Google提出FLAN大模型，其基于Instruction Fine-Tuning
FLAN is the instruction-tuned version of LaMDA-PT
Scaling Instruction-Finetuned Language Models，Flan-T5(2022年10月)
从三个方面改变指令微调，一是改变模型参数，提升到了540B，二是增加到了1836个微调任务，三是加上Chain of thought微调的数据
LLaMA: Open and Efficient Foundation Language Models，2023年2月Meta发布了全新的650亿参数大语言模型LLaMA，开源，大部分任务的效果好于2020年的GPT-3
Language Is Not All You Need: Aligning Perception with Language Models，微软23年3月1日发布的多模态大语言模型论文
GLM: General Language Model Pretraining with Autoregressive Blank Infilling，国内唐杰团队的

A Comprehensive Survey on Pretrained Foundation Models: A History from BERT to ChatGPT：https://arxiv.org/pdf/2302.09419，预训练基础模型的演变史
LaMDA: Language Models for Dialog Applications，Google在21年5月对外宣布内部正在研发对话模型LaMDA
Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer
Pre-train, Prompt, and Predict: A Systematic Survey of Prompting Methods in Natural Language Processing，作者来自CMU的刘鹏飞，这是相关资源
Multimodal Chain-of-Thought Reasoning in Language Models
23年2月，亚马逊的研究者则在这篇论文里提出了基于多模态思维链技术改进语言模型复杂推理能力的思想
Offsite-Tuning: Transfer Learning without Full Model
对于许多的私有基础模型，数据所有者必须与模型所有者分享他们的数据以微调模型，这是非常昂贵的，并引起了隐私问题（双向的，一个怕泄露模型，一个怕泄露数据）
Emergent Abilities of Large Language Models
Google 22年8月份发的，探讨大语言模型的涌现能力


Large Language Models are Zero-Shot Reasoners
来自东京大学和谷歌的工作，关于预训练大型语言模型的推理能力的探究，“Let's think step by step”的梗即来源于此篇论文
PaLM: Scaling Language Modeling with Pathways，这是翻译之一
22年4月发布，是Google的Pathways架构或openAI GPT2/3提出的小样本学习的进一步扩展
PaLM-E: An Embodied Multimodal Language Model，Google于23年3月6日发布的关于多模态LLM
Visual ChatGPT: Talking, Drawing and Editing with Visual Foundation Models，微软于23年3月8日推出visual ChatGPT(另，3.9日微软德国CTO说，将提供多模态能力的GPT4即将一周后发布)
At the same time, Visual Foundation Models, such as Visual Transformers or Stable Diffusion, although showing great visual understanding and generation capabilities, they are only experts on specific tasks with one round fixed inputs and outputs. 

To this end, We build a system called {Visual ChatGPT}, incorporating different Visual Foundation Models, to enable the user to interact with ChatGPT by 
1) sending and receiving not only languages but also images 
2) providing complex visual questions or visual editing instructions that require the collaboration of multiple AI models with multi-steps. 
3) providing feedback and asking for corrected results. 

We design a series of prompts to inject the visual model information into ChatGPT, considering models of multiple inputs/outputs and models that require visual feedback
《The Natural Language Decathlon:Multitask Learning as Question Answering》，GPT-1、GPT-2论文的引用文献，Salesforce发表的一篇文章，写出了多任务单模型的根本思想
Deep Residual Learning for Image Recognition，ResNet论文，短短9页，Google学术被引现15万多
这是李沐针对ResNet的解读，另 这是李沐针对一些paper的解读列表
The Flan Collection: Designing Data and Methods for Effective Instruction Tuning

AN IMAGE IS WORTH 16X16 WORDS:TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE
Transformer杀入CV界
Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
Swin Transformer V2: Scaling Up Capacity and Resolution
第一篇的解读戳这，第二篇的解读戳这里
Denoising Diffusion Probabilistic Models
2020年提出Diffusion Models(所谓diffusion就是去噪点的意思)
CLIP: Connecting Text and Images - OpenAI
CLIP由OpenAI在2021年1月发布，超大规模模型预训练提取视觉特征，图片和文本之间的对比学习(简单粗暴理解就是发微博/朋友圈时，人喜欢发一段文字然后再配一张或几张图，CLIP便是学习这种对应关系)

2021年10月，Accomplice发布的disco diffusion，便是第一个结合CLIP模型和diffusion模型的AI开源绘画工具，其内核便是采用的CLIP引导扩散模型(CLIP-Guided diffusion model)
Hierarchical Text-Conditional Image Generation with CLIP Latents
DALL.E 2论文2022年4月发布(至于第一代发布于2021年初)，通过CLIP + Diffusion models，达到文本生成图像新高度
High-Resolution Image Synthesis with Latent Diffusion Models
2022年8月发布的Stable Diffusion基于Latent Diffusion Models，专门用于文图生成任务
这些是相关解读：图解stable diffusion(翻译版之一)、这是另一解读，这里有篇AI绘画发展史的总结

Stable Diffusion和之前的Diffusion扩散化模型相比, 重点是做了一件事, 那就是把模型的计算空间，从像素空间经过数学变换，在尽可能保留细节信息的情况下降维到一个称之为潜空间(Latent Space)的低维空间里，然后再进行繁重的模型训练和图像生成计算
Aligning Text-to-Image Models using Human Feedback，这是解读之一
ChatGPT的主要成功要归结于采用RLHF来精调LLM，近日谷歌AI团队将类似的思路用于文生图大模型：基于人类反馈（Human Feedback）来精调Stable Diffusion模型来提升生成效果
目前的文生图模型虽然已经能够取得比较好的图像生成效果，但是很多时候往往难以生成与输入文本精确匹配的图像，特别是在组合图像生成方面。为此，谷歌最新的论文提出了基于人类反馈的三步精调方法来改善这个问题

