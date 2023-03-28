#### [Towards Boosting the Open-Domain Chatbot with Human Feedback](https://arxiv.org/pdf/2208.14165.pdf)
发布了一个高质量中文多轮chitchat数据集Diamonte，并详细介绍了多轮对话数据集的构建流程。

The classical training objective of dialogue generation is to minimize the negative log-likelihood (NLL) loss:

&emsp;&emsp; L<sub>NLL</sub> = − log pθ(r<sub>H</sub>|c) 
    
The preference estimation (PE) loss is then defined as:
 
&emsp;&emsp; L<sub>PE</sub> = −1/3*[log(σ(s(c, r<sub>H</sub>) − s(c, r<sub>M</sub>)))+log(σ(s(c, r<sub>H</sub>) − s(c, r<sub>R</sub>))) + log(σ(s(c, r<sub>M</sub>) − s(c, r<sub>R</sub>)))]

 <img width="1013" alt="image" src="https://user-images.githubusercontent.com/8350994/227929218-db0821f5-e135-4497-b1e3-d828419e60dc.png">

#### [Link the World: Improving Open-domain Conversation with Dynamic Spatiotemporal-aware Knowledge](https://arxiv.org/pdf/2206.14000.pdf)
发布了DuSinc数据集。可以根据dialog context and spatiotemporal state(时空状态）检索信息服务，进行实时问答服务。

<img width="852" alt="image" src="https://user-images.githubusercontent.com/8350994/227996238-d1102093-6bf2-4970-b40d-84f4b0a9cf3b.png">

基于DuSinc的模型训练才有share parameter的方式。预测query和response,计算NLL(Negative Log Likelihood) loss。

<img width="1015" alt="image" src="https://user-images.githubusercontent.com/8350994/227996545-51b7c64a-afe7-4262-9cea-ce1dd78eb2d3.png">

#### [PLATO-K: Internal and External Knowledge Enhanced Dialogue Generation](https://arxiv.org/pdf/2211.00910.pdf)
PLATO-K 通过两个阶段的训练，来强化内部知识的记忆和外部知识的利用。第一阶段，PLATO-K 通过预训练和finetune，从大量的对话数据中学习必要的知识。第二阶段，PLATO-K 学习在response中使用搜索到外部信息。

<img width="922" alt="image" src="https://user-images.githubusercontent.com/8350994/228116207-58960dda-fe02-4bbf-9b13-79833118f4c8.png">

将社会媒体评论(social media comments)和网络文本(web texts)转化为对话的流程：

<img width="912" alt="image" src="https://user-images.githubusercontent.com/8350994/228116367-87621aab-7191-406c-98fa-531afd2bf0fb.png">


#### [PLATO-XL: Exploring the Large-scale Pre-training of Dialogue Generation](https://arxiv.org/pdf/2109.09519.pdf)
PLATO-XL 拥有11B参数, 训练自**中英文社交媒体(social media)对话**。PLATO-XL在中英文闲聊、任务型对话上都取得了SOTA的效果。在预训练过程中，使用了multi-party aware pretraining区分对话的类型。

社交媒体(social media)对话的特点：

&emsp;&emsp;1) 在某个上下文环境(contexts)中，会存在不同层次(multi-level)评论(comments).

&emsp;&emsp;2) 多个用户(multiple users)会参与到同一个对话中。 

<img width="986" alt="image" src="https://user-images.githubusercontent.com/8350994/228120021-57db1546-208f-4798-878d-c412f3716456.png">

任何一条从根节点(root node)到叶子节点(tree node)的路径都能被视为一段包含对话上下文和回复的完整对话。

<img width="1082" alt="image" src="https://user-images.githubusercontent.com/8350994/228124645-9da62cb6-00ce-48f8-b330-96cf03ddc0bc.png">


#### [PLATO: Pre-trained Dialogue Generation Model with Discrete Latent Variable](https://aclanthology.org/2020.acl-main.9.pdf)
上下文部分(context)使用bi-directional attention，在response 部分使用uni-directional attention，使用discrete latent variables保证结果的多样性。模型的网络结构如下，

<img width="819" alt="image" src="https://user-images.githubusercontent.com/8350994/228151633-b33153ac-9ff4-48ab-8c41-0e191f326b7d.png">

预训练的LOSS包括，L = L<sub>NLL</sub> + L<sub>BOW</sub> + L<sub>RS</sub> 
<br/> negative log-likelihood (NLL) loss, bag-of-words (BOW) loss and response selection (RS) loss.


#### [百度PLATO 知乎](https://www.zhihu.com/question/493911256/answer/2931945930)
