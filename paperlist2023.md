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

#### [The Power of Scale for Parameter-Efficient Prompt Tuning] (https://aclanthology.org/2021.emnlp-main.243.pdf)
学习 “soft prompts”  用于 specific downstream tasks。可以保留 zero-shot, few-shot 的能力。
