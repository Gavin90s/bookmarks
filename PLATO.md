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


#### [PLATO-XL: Exploring the Large-scale Pre-training of Dialogue Generation](https://arxiv.org/pdf/2211.00910.pdf)


#### [PLATO: Pre-trained Dialogue Generation Model with Discrete Latent Variable](https://aclanthology.org/2020.acl-main.9.pdf)


#### [百度PLATO 知乎](https://www.zhihu.com/question/493911256/answer/2931945930)
