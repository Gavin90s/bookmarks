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
