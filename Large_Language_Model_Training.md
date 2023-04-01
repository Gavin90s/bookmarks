#### RMS LayerNorm
&emsp;&emsp;Normalization一般都包含了减均值（center）和除以标准差（scale）两个部分，但近来的一些工作逐渐尝试去掉center这一步，甚至有些工作的结果显示去掉center这一步后性能还略有提升。<br/>
&emsp;&emsp;<img width="347" alt="image" src="https://user-images.githubusercontent.com/8350994/229290093-69f6619c-88ab-4630-bec9-1cab35f0389a.png">
<br/>&emsp;&emsp;2019年的论文[《Root Mean Square Layer Normalization》](https://arxiv.org/abs/1910.07467) 比较了去掉center后的Layer Normalization，文章称之为RMS Norm。RMS Norm也就是L2 Normalization的简单变体而已，但这篇论文总的结果显示：RMS Norm比Layer Normalization更快，效果也基本一致。
<br/>&emsp;&emsp;除了这篇文章外，RMS Norm还被Google用在了T5中，并且在另外的一篇文章 [《Do Transformer Modifications Transfer Across Implementations and Applications?》](https://arxiv.org/abs/2102.11972)中做了比较充分的对比实验，显示出RMS Norm的优越性。这样看来，未来RMS Norm很可能将会取代Layer Normalization而成为Transformer的标配。
<br/>&emsp;&emsp;一个直观的猜测是，center操作，类似于全连接层的bias项，储存到的是关于数据的一种先验分布信息，而把这种先验分布信息直接储存在模型中，反而可能会导致模型的迁移能力下降。所以T5不仅去掉了Layer Normalization的center操作，它把每一层的bias项也都去掉了。

#### SwiGLU
Swish激活函数：
<br/>&emsp;&emsp;&emsp;&emsp;𝑆𝑤𝑖𝑠ℎ=𝑥⋅𝑠𝑖𝑔𝑚𝑜𝑖𝑑(𝛽𝑥)
<br/>我们不难发现，激活函数就是对x乘以一些数，以对某些值进行约束。
<br/>GLU（Gated Linear Unit），其一般形式为：
<br/>&emsp;&emsp;&emsp;&emsp;𝐺𝐿𝑈(𝑥)=𝜎(𝑊𝑥+𝑏)⊗(𝑉𝑥+𝑐)
<br/>![image](https://user-images.githubusercontent.com/8350994/229295009-b83833d1-b5c2-4272-ad5a-7364bd0d70dc.png)
<br/>What does the SwiGLU activation function look like?
<br/>The SwiGLU activation function is a piecewise linear function that is defined as follows:
<br/>&emsp;&emsp;&emsp;&emsp;SwiGLU(x) = max(x, 0) + min(α(x - ReLU(x)), 0)
<br/>where x is the input to the function, ReLU(x) is the rectified linear unit function (i.e., max(x, 0)), and α is a tunable parameter that controls the shape of the negative part of the function.
<br/>The SwiGLU activation function is designed to address some of the limitations of the ReLU function, which can result in "dead" neurons that do not contribute to the output of a neural network. By introducing a piecewise linear negative slope, the SwiGLU function can help to prevent this problem and improve the performance of neural networks.
