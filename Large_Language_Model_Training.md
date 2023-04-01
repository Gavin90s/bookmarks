#### RMS LayerNorm
&emsp;&emsp;Normalization一般都包含了减均值（center）和除以标准差（scale）两个部分，但近来的一些工作逐渐尝试去掉center这一步，甚至有些工作的结果显示去掉center这一步后性能还略有提升。<br/>
&emsp;&emsp;<img width="347" alt="image" src="https://user-images.githubusercontent.com/8350994/229290093-69f6619c-88ab-4630-bec9-1cab35f0389a.png">
<br/>&emsp;&emsp;2019年的论文[《Root Mean Square Layer Normalization》](https://arxiv.org/abs/1910.07467) 比较了去掉center后的Layer Normalization，文章称之为RMS Norm。RMS Norm也就是L2 Normalization的简单变体而已，但这篇论文总的结果显示：RMS Norm比Layer Normalization更快，效果也基本一致。
<br/>&emsp;&emsp;除了这篇文章外，RMS Norm还被Google用在了T5中，并且在另外的一篇文章 [《Do Transformer Modifications Transfer Across Implementations and Applications?》](https://arxiv.org/abs/2102.11972)中做了比较充分的对比实验，显示出RMS Norm的优越性。这样看来，未来RMS Norm很可能将会取代Layer Normalization而成为Transformer的标配。
<br/>&emsp;&emsp;一个直观的猜测是，center操作，类似于全连接层的bias项，储存到的是关于数据的一种先验分布信息，而把这种先验分布信息直接储存在模型中，反而可能会导致模型的迁移能力下降。所以T5不仅去掉了Layer Normalization的center操作，它把每一层的bias项也都去掉了。
