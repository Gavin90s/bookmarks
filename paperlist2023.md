#### [Progressive Prompts: Continual Learning For Language Models](https://arxiv.org/pdf/2301.12314.pdf)
```
大模型在业务上使用，避免知识遗忘，实现知识迁移学习。
https://ai.googleblog.com/2022/04/learning-to-prompt-for-continual.html
```

#### [Understanding the Effectiveness of Very Large Language Models on Dialog Evaluation](https://arxiv.org/pdf/2301.12004.pdf)
```
定义了LLM大模型的评估方法以及对话的评估指标，还有prompt learning 和 in-context learning 中样本的选择。
```

#### [Selective Annotation Makes Language Models Better Few-Shot Learners](https://arxiv.org/pdf/2209.01975.pdf)(Provide Many Prompt Samples)
```
通过从 unlabeled 数据集中筛选数据进行标注，然后给每一条 test 数据选择一条相似的标注数据进行In-context learning。
在low resource情况下，In-context learning比finetune的方式效果要好。
```

#### [Scaling Instruction-Finetuned Language Models](https://arxiv.org/pdf/2210.11416.pdf)
```
证明了 Instruction-Finetuned 的有效性和必要性。
1、instruction finetuning：the size of the model and the number of finetuning tasks—improve performance.
2、CoT finetuning is critical for reasoning abilities.
3、Instruction finetuning generalizes across models.
4、Instruction finetuning improves usability and mitigates some potential harms.
5、Instruction finetuning is relatively compute-efficient.
```

#### [Multitask Prompted Training Enables Zero-Shot Task Generalization](https://arxiv.org/pdf/2110.08207.pdf)
```
1、multitask training enables zero-shot task generalization by showing that our model matches or exceeds the performance
of GPT-3 (Brown et al.,2020) on 9 out of 11 held-out datasets, despite being about 16× smaller. 
2、training on more prompts per dataset consistently improves the median and decreases the variability of performance
on held-out tasks. 
```

#### [Language Models are Few-Shot Learners](https://arxiv.org/pdf/2005.14165.pdf)(GPT3)

#### [The Power of Scale for Parameter-Efficient Prompt Tuning](https://aclanthology.org/2021.emnlp-main.243.pdf)
```
学习 “soft prompts”  用于 specific downstream tasks。prompts tuning随着模型参数增大，能取得明显效果提升，可以接近或超过model tuning。soft prompts tuning可以保留 zero-shot, few-shot 的能力。
```
<img width="419" alt="image" src="https://user-images.githubusercontent.com/8350994/227443419-95330419-d301-44c8-983a-7908301bc832.png">


#### [Question-Generation-Paper-List](https://github.com/teacherpeterpan/Question-Generation-Paper-List)
```
Question Generation 输入 article contents 输出 questions。
Answer system 通过输入article contents和 question 文件，生成回复。
```

#### [Question Answering Leaderboards](https://paperswithcode.com/task/question-answering)
```
Question Answering is the task of answering questions (typically reading comprehension questions), 
but abstaining when presented with a question that cannot be answered based on the provided context.

Question answering can be segmented into domain-specific tasks like community question answering 
and knowledge-base question answering. Popular benchmark datasets for evaluation question 
answering systems include SQuAD, HotPotQA, bAbI, TriviaQA, WikiQA, and many others. 
Models for question answering are typically evaluated on metrics like EM and F1. 
Some recent top performing models are T5 and XLNet.
```

#### [AUTOPROMPT: Eliciting Knowledge from Language Models with Automatically Generated Prompts](https://arxiv.org/pdf/2010.15980.pdf)

AUTOPROMPT，用MLM语言模型实现情感分析。每个输入x<sub>inp</sub>都被放入到一个自然语言提示符 x<sub>prompt</sub> 中，包含单个[MASK] token。
Prompt是使用模板 λ 创建的，它结合了x<sub>inp</sub>和x<sub>trig</sub> tokens。x<sub>trig</sub> tokens在所有输入之间共享，并使用gradient-based
search确定x<sub>trig</sub> tokens。然后通过p([MASK]|x<sub>prompt</sub>) 预测类别标签y的值。标签y的集合是通过算法自动生成的。
<img width="976" alt="image" src="https://user-images.githubusercontent.com/8350994/227159328-fa8d13c6-e8df-45ad-b5da-fd7ff5c3ae04.png">

#### [Reading Wikipedia to Answer Open-Domain Questions](https://aclanthology.org/P17-1171.pdf)
系统分为两大块，首先有一个 Document Retriever，对给定的问题 question，从所有维基百科文章中检索（这里面注意SQuAD 实际上是有具体的位置的，但是WebQuestion等等数据集就是没有的，这里面她是通过远程监督的方式做的处理）；检索到文章后切分成段落，然后用一个称之为 Document Reader 的模块在段落中预测答案位置并给出分数。后者其实就是标准的阅读理解模型了，完全可以替换成其他的机器阅读理解模型。

<img width="617" alt="image" src="https://user-images.githubusercontent.com/8350994/227428539-e2da805b-21a3-497d-8c64-f47dc98a0ff3.png">

#### [REALM: Retrieval-Augmented Language Model Pre-Training](https://arxiv.org/pdf/2002.08909.pdf)
REALM通过Neural Knowledge Retriever检索文本知识库增强语言模型预训练，整个系统是端到端训练的。Retriever计算每个文档的embedding，并都缓存起来，异步更新。最佳匹配文档通过最大内积搜索（MIPS) 得到。

<img width="476" alt="image" src="https://user-images.githubusercontent.com/8350994/227415246-17b4ee9e-96de-4f0e-9f10-66983a82de53.png">

#### [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/pdf/2005.11401.pdf)
RAG 解决的痛点是之前的研究比如REALM是基于MLM模型的，只方便做提取任务，其他能做的任务就比较受限，用generator可以赋予其更大的能力。
https://ai.facebook.com/blog/retrieval-augmented-generation-streamlining-the-creation-of-intelligent-natural-language-processing-models/

<img width="833" alt="image" src="https://user-images.githubusercontent.com/8350994/227430284-f716c91f-2a61-43db-b374-9ce61dc73bdd.png">

#### [Open Domain Question Answering with A Unified Knowledge Interface](https://arxiv.org/pdf/2110.08417.pdf)
论文提出了一个用于 ODQA 的 verbalizer-retriever-reader 框架，来自维基百科的文字、表格、图被用作增强知识来源。

<img width="943" alt="image" src="https://user-images.githubusercontent.com/8350994/227441640-2313a245-5751-4d27-8f49-2684e9397182.png">


#### [OpenPrompt: An Open-source Framework for Prompt-learning](https://arxiv.org/pdf/2111.01998.pdf)
OpenPrompt是一个research-friendly Prompt Tunning框架, 具有效率(efficiency)，模块化(modularity)、可扩展性(extendibility)的特点。用户可以在一个统一的范式中使用不同的PLM、任务格式(task formats)和提示模块，并评估效果。

<img width="836" alt="image" src="https://user-images.githubusercontent.com/8350994/227429179-8e33a1df-7d84-4cac-a4e7-a269635d864d.png">


