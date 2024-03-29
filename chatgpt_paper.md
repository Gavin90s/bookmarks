```
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


提升 自己的技术研究能力也能有巨大飞跃，且考虑到为避免上篇文章篇幅太长而影响完读率，故把这100篇论文的清单抽取出来分享给大家：
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
Large Language Models are Zero-Shot Reasoners
来自东京大学和谷歌的工作，关于预训练大型语言模型的推理能力的探究，“Let's think step by step”的梗即来源于此篇论文
Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer
19年10月，Google发布T5模型(transfer text to text transformer)，虽也基于transformer，但区别于BERT的编码器架构与GPT的解码器架构，T5是transformer的encoder-decoder架构，这是解读之一 的
用的750G的训练数据，其训练方法则为：BERT-style的MASK法/replace span(小段替换)/Drop法，以及类似BERT对文本的15%做破坏、且replace span时对3的小段破坏
LaMDA: Language Models for Dialog Applications， 这是简要解读之一
21年5月，Google对外宣布内部正在研发对话模型LaMDA，基于transformer decoder架构，在微调阶段 使用58K的对话数据，过程类似真人的对话过程，给定一个Query，比如 How old is Rafael Nadal? ，如果人知道答案，那么直接回答35岁即可，如果不知道，则需要去 Research 一下，借助搜索引擎找到答案，然后再回答35岁
《Finetuned Language Models Are Zero-Shot Learners》
21年9月，Google提出FLAN大模型，其基于LaMDA-PT做Instruction Fine-Tuning
FLAN is the instruction-tuned version of LaMDA-PT
PaLM: Scaling Language Modeling with Pathways
22年3月，Google的Barham等人发布了Pathways系统，用于更高效地训练大型模型
Pathways 的愿景 —— 一个很接近人脑的框架：一个模型，可以做多任务，多模态
且在做任务时，只是 sparsely activated，只使用一部分的参数

22年4月，Google发布PaLM模型，基于Transformer decoder架构，参数规模最大的版本达到惊人的5400亿参数(8B 62B 540B)，使用multi-query注意力、SwiGLU激活函数以及RoPE位置嵌入，这是翻译之一
且在每个Transformer块中使用 "平行 "表述(Wang & Komatsuzaki,2021)
是Google的Pathways架构或OpenAI GPT2/3提出的小样本学习的进一步扩展

PaLM首次展示了Pathways的大规模使用——能够以高效的方式在数千或数万个加速器芯片上训练一个模型
具体来说，通过Pathways，PaLM 540B在两个通过数据中心网络连接的TPU v4 Pod上训练，使用模型和数据并行的组合，在每个Pod中使用3072个TPU v4芯片，连接到768台主机，能够有效地将训练扩展到6144个芯片，而不需要使用任何pipeline并行，其效率水平是以前这种规模的模型所不能达到的

以前的大多数大型语言模型
  要么是在单个TPU系统上训练的(比如GLaM by Du等人2021年，LaMDA by Thopilan等人)
 要么是使用由Huang等人在2019年提出的pipeline并行，从而在GPU集群(Megatron-Turing NLG 530B by Smith等人2022年)，或多个TPU v3 pod(Gopher by Rae等人2021年)上扩展，最大规模为4096个TPU v3芯片

另，在自然语言、代码和数学推理等任务中表现的都很不错
此外，预训练数据集由一个7800亿个token组成的语料库，该数据集是由过滤过的网页(占比27%)、书籍(占比13%)、Wikipedia(占比4%)、新闻文章(占比1%)、Github源代码(占比5%，包括Java、HTML、Javascript、Python、PHP、C#、XML、C++和C，总计196GB的源代码)，和社交媒体对话(占比50%)组成的，这个数据集是也用于训练LaMDA和GLaM
Emergent Abilities of Large Language Models
Google 22年8月份发的，探讨大语言模型的涌现能力
Scaling Instruction-Finetuned Language Models，Flan-T5(2022年10月)
从三个方面改变指令微调，一是改变模型参数，提升到了540B，二是增加到了1836个微调任务，三是加上Chain of thought微调的数据
Multimodal Chain-of-Thought Reasoning in Language Models
23年2月，亚马逊的研究者则在这篇论文里提出了基于多模态思维链技术改进语言模型复杂推理能力的思
LLaMA: Open and Efficient Foundation Language Models，2023年2月24日Meta发布了全新的650亿参数大语言模型LLaMA，开源，大部分任务的效果好于2020年的GPT-3
这是针对该论文的解读之一
Language Is Not All You Need: Aligning Perception with Language Models，微软23年3月1日发布的多模态大语言模型Kosmos-1的论文
PaLM-E: An Embodied Multimodal Language Model(论文地址)，Google于23年3月6日发布的关于多模态LLM：PaLM-E，可让能听懂人类指令且具备视觉能力的机器人干活
Visual ChatGPT: Talking, Drawing and Editing with Visual Foundation Models，微软于23年3月8日推出visual ChatGPT(另，3.9日微软德国CTO说，将提供多模态能力的GPT4即将一周后发布)
At the same time, Visual Foundation Models, such as Visual Transformers or Stable Diffusion, although showing great visual understanding and generation capabilities, they are only experts on specific tasks with one round fixed inputs and outputs. 

To this end, We build a system called {Visual ChatGPT}, incorporating different Visual Foundation Models, to enable the user to interact with ChatGPT by 
1) sending and receiving not only languages but also images 
2) providing complex visual questions or visual editing instructions that require the collaboration of multiple AI models with multi-steps. 
3) providing feedback and asking for corrected results. 

We design a series of prompts to inject the visual model information into ChatGPT, considering models of multiple inputs/outputs and models that require visual feedback
A Comprehensive Survey on Pretrained Foundation Models: A History from BERT to ChatGPT：https://arxiv.org/pdf/2302.09419，预训练基础模型的演变史
Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer
Pre-train, Prompt, and Predict: A Systematic Survey of Prompting Methods in Natural Language Processing，作者来自CMU的刘鹏飞，这是相关资源

另一篇类似的，Pre-Trained Models: Past, Present and Future
Offsite-Tuning: Transfer Learning without Full Model
对于许多的私有基础模型，数据所有者必须与模型所有者分享他们的数据以微调模型，这是非常昂贵的，并引起了隐私问题（双向的，一个怕泄露模型，一个怕泄露数据）
《The Natural Language Decathlon:Multitask Learning as Question Answering》，GPT-1、GPT-2论文的引用文献，Salesforce发表的一篇文章，写出了多任务单模型的根本思想
Deep Residual Learning for Image Recognition，ResNet论文，短短9页，Google学术被引现15万多
这是李沐针对ResNet的解读，另 这是李沐针对一些paper的解读列表
End-to-End Object Detection with Transformers
DETR by 2020年5月，这是针对DETR的解读之一
回顾下20年之前的模型提出史(我18年写过一篇：一文读懂目标检测：R-CNN、Fast R-CNN、Faster R-CNN、YOLO、SSD)
2014 R-CNN
2015 Fast R-CNN、Faster R-CNN
2016 YOLO、SSD
2017 Mask R-CNN、YOLOv2
2018 YOLOv3
2019 CenterNet
2020 DETR

20年之后，CV迎来了生成式下的多模态时代
2020年
 6月 DDPM
10月 DDIM、Vision Transformer
 2021年
1月 CLIP、DALL·E
 3月 Swin Transformer
 11月 MAE、Swin Transformer V2

  2022年
1月 BLIP
4月 DALL·E 2
 8月 Stable Diffusion、BEiT-3
  
2023年
 1月 BLIP2
3月 Visual ChatGPT、GPT-4
AN IMAGE IS WORTH 16X16 WORDS:TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE
发表于2020年10月的Vision Transformer原始论文，代表Transformer正式杀入CV界
Swin Transformer: Hierarchical Vision Transformer using Shifted Windows，发表于21年3月
Swin Transformer V2: Scaling Up Capacity and Resolution
第一篇的解读戳这，第二篇的解读戳这里
Auto-Encoding Variational Bayes，苏剑林关于VAE的解读之一
WGAN
Denoising Diffusion Probabilistic Models，2020年6月提出DDPM，即众人口中常说的diffusion model
这是苏剑林关于DDPM的相对通俗的系列解读，这是另一份解读：What are Diffusion Models?(该解读的中文笔记)
CLIP: Connecting Text and Images - OpenAI，这是针对CLIP论文的解读之一
CLIP由OpenAI在2021年1月发布，超大规模模型预训练提取视觉特征，图片和文本之间的对比学习(简单粗暴理解就是发微博/朋友圈时，人喜欢发一段文字然后再配一张或几张图，CLIP便是学习这种对应关系)

2021年10月，Accomplice发布的disco diffusion，便是第一个结合CLIP模型和diffusion模型的AI开源绘画工具，其内核便是采用的CLIP引导扩散模型(CLIP-Guided diffusion model)
且后续有很多基于CLIP的一系列改进模型，比如Lseg、GroupViT、ViLD、GLIP
Hierarchical Text-Conditional Image Generation with CLIP Latents，这是解读之一
DALL·E 2论文2022年4月发布(至于第一代发布于2021年初)，通过CLIP + Diffusion models，达到文本生成图像新高度
High-Resolution Image Synthesis with Latent Diffusion Models
2022年8月发布的Stable Diffusion基于Latent Diffusion Models，专门用于文图生成任务
这些是相关解读：图解stable diffusion(翻译版之一)、这是另一解读，这里有篇AI绘画发展史的总结

Stable Diffusion和之前的Diffusion扩散化模型相比, 重点是做了一件事, 那就是把模型的计算空间，从像素空间经过数学变换，在尽可能保留细节信息的情况下降维到一个称之为潜空间(Latent Space)的低维空间里，然后再进行繁重的模型训练和图像生成计算
BLIP (from Salesforce) released with the paper BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation by Junnan Li, Dongxu Li, Caiming Xiong, Steven Hoi.
BLIP-2 (from Salesforce) released with the paper BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models by Junnan Li, Dongxu Li, Silvio Savarese, Steven Hoi.
Image as a Foreign Language: BEIT Pretraining for All Vision and Vision-Language Tasks，这是针对该论文的解读之一
2022年8月，微软提出的多模态预训练模型BEiT-3

BEiT: BERT Pre-Training of Image Transformers
BEiT-2: Masked Image Modeling with Vector-Quantized Visual Tokenizers
Aligning Text-to-Image Models using Human Feedback，这是解读之一
ChatGPT的主要成功要归结于采用RLHF来精调LLM，近日谷歌AI团队将类似的思路用于文生图大模型：基于人类反馈（Human Feedback）来精调Stable Diffusion模型来提升生成效果
目前的文生图模型虽然已经能够取得比较好的图像生成效果，但是很多时候往往难以生成与输入文本精确匹配的图像，特别是在组合图像生成方面。为此，谷歌最新的论文提出了基于人类反馈的三步精调方法来改善这个问题

SELF-INSTRUCT: Aligning Language Model with Self Generated Instructions，代码地址，解读1、解读2
3月中旬，斯坦福发布Alpaca：只花100美元，人人都可微调Meta家70亿参数的LLaMA大模型
而斯坦福团队微调LLaMA的方法，便是来自华盛顿大学Yizhong Wang等去年底提出的这个Self-Instruct

具体而言，论文中提出，首先从自生成指令种子集中的175个人工编写的「指令-输出」对开始，然后，提示text-davinci-003使用种子集作为上下文示例来生成更多指令
而斯坦福版Alpaca，就是花了不到500美元使用OpenAI API生成了5.2万个这样的示例微调LLaMA搞出来的

 Constitutional AI: Harmlessness from AI Feedback
OpenAI之前一副总裁离职搞了个ChatGPT的竞品，ChatGPT用人类偏好训练RM再RL(即RLHF)，Claude则基于AI偏好模型训练RM再RL(即RLAIF)

 Improving alignment of dialogue agents via targeted human judgements
DeepMind的Sparrow，这个工作发表时间稍晚于instructGPT，其大致的技术思路和框架与 instructGPT 的三阶段基本类似，但Sparrow 中把奖励模型分为两个不同 RM 的思路

Why Can GPT Learn In-Context? Language Models Secretly Perform Gradient Descent as Meta-Optimizers代码地址，这篇文章则将ICL看作是一种隐式的Fine-tuning
WHAT LEARNING ALGORITHM IS IN-CONTEXT LEARNING? INVESTIGATIONS WITH LINEAR MODELS
Meta-learning via Language Model In-context Tuning
Large language models are zero-shot reasoners. arXiv preprint arXiv:2205.11916, 2022
Transformer-XL: Attentive language models beyond a fixed-length context
Tensor programs v: Tuning large neural networks via zero-shot hyperparameter transfer. arXiv preprint arXiv:2203.03466, 2022
Opt: Open pre-trained transformer language models. arXiv preprint arXiv:2205.01068, 2022
Language models are unsupervised multitask learners. 2019
```
