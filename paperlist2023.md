#### PROGRESSIVE PROMPTS: CONTINUAL LEARNING FOR LANGUAGE MODELS
```
大模型在业务上使用，避免知识遗忘，实现知识迁移学习。
```
```
https://ai.googleblog.com/2022/04/learning-to-prompt-for-continual.html
```

#### Understanding the Effectiveness of Very Large Language Models on Dialog Evaluation
```
定义了LLM大模型的评估方法以及对话的评估指标，还有prompt learning 和 in-context learning 中样本的选择。
```
```
https://arxiv.org/pdf/2301.12004.pdf
```

#### SELECTIVE ANNOTATION MAKES LANGUAGE MODELS BETTER FEW-SHOT LEARNERS
```
通过从 unlabeled 数据集中筛选数据进行标注，然后给每一条 test 数据选择一条相似的标注数据进行In-context learning。
在low resource情况下，In-context learning比finetune的方式效果要好。
```
```
https://arxiv.org/pdf/2209.01975.pdf
```

#### Scaling Instruction-Finetuned Language Models
```
证明了 Instruction-Finetuned 的有效性和必要性。
1、instruction finetuning：the size of the model and the number of finetuning tasks—improve performance.
2、CoT finetuning is critical for reasoning abilities.
3、Instruction finetuning generalizes across models.
4、Instruction finetuning improves usability and mitigates some potential harms.
5、Instruction finetuning is relatively compute-efficient.
```
```
https://arxiv.org/pdf/2210.11416.pdf
```
