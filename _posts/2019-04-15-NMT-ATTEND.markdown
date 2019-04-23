---
layout:     post
title:      "论文阅读笔记"
subtitle:   " \"Neural Machine Translation by Jointly Learning to Align and Translate\""
date:       2019-04-22 
author:     "gyr"
header-style: text
header-img: "img/post-bg1.jpg"
mathjax: true
catalog: true
tags:
    - 论文笔记
    - attention model
    - image caption
    
---

## 前言
先开一篇关于attention论文的阅读笔记，准确的说是三篇论文一起:

第一篇是:[Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473v2)是Attention Model的开篇之作，虽然本文主要应用的是机器翻译的领域，但是之后对各个领域都产生的深远影响，并且产生了很多变形与应用。

第二篇是:[Show, Attend and Tell: Neural Image Caption Generation with Visual Attention](https://arxiv.org/abs/1502.03044v1)
是Image caption任务的论文，他在之前研究的基础上加入了Attention Model的思想,从而带来了较好的效果。

第三篇是:[Attention Is All You Need](https://arxiv.org/abs/1706.03762)也是机器翻译任务，文中使用了完全依赖于attention机制的网络结构，取得了比较好的效果。

---

# 1.Neural Machine Translation by Jointly Learning to Align and Translate

### 背景

神经机器翻译是前些年来提出的一种新兴的机器翻译方法,神经机器翻译试图建立和训练一个单一的、大型的神经网络,它能读取一个句子并输出正确的翻译。所提出的神经机器翻译模型大多属于一类编码器解码器(Encoder-Decoder),具体关于机器翻译的内容还有更多的应用,但不是本次研究的重点,以后有机会再学习。

### Encoder-Decoder模型

#### The Encoder
对于一个长度为T的文本序列X $ = (x_1,x_2,...,x_T) $,中每一个词都被表示成一个向量 $ \omega_i \in R^{|V|} $,i=1,2...T,|V|为词汇表大小。通常这个$ \omega_i $是一条one_hot向量。接下来就是要对要将每个词映射到一个低维的语义空间，也就是一个词向量。记映射矩阵为$ C \in R^{K \times |V|} $,则第$i$个词表示的词向量为，$s_i = C\omega_i$，向量维度*K*通常的取值在100到500之间。这个词向量在整个模型训练的过程中也会逐步更新，会变得更加有意义。

![encoder过程](/img/encoder过程.png)

紧接着就可以用一个RNN模型压缩表示语言序列：$$h_i = \phi_\theta(h_{i-1},s_i)=\Theta(Ws_i+Uh_{i-1}+b)$$

其中$\Theta$表示隐含层的激活函数，常见的有sigmoid、tanh等。$h_T$表示整个句子的压缩结果，为了探究$h_T$的性质Sutskever et al.2014利用主成分分析找到向量$h_T$的两个主成分，并画出下图。

![主成分](/img/pca.png)

从图中可以看出，压缩向量确实可以包含原语句中的语义信息，在二维的空间表达中，距离越近语义越接近。

#### The Decoder

decoder部分同样实用RNN实现，根据encoder，我们得到了一个句子的压缩表示$c = q({h_1,...,h_T})$,接着就开始计算RNN的隐藏状态$z_t,t=1,...,T$,$z_t$的计算公式如下：

$$z_t=\Theta(c,y_{t-1},z_{t-1})$$

其中$y_0$是结束标记“<EOS>”用以表示解码的开始与结束,$y_{t-1}$表示前一个单词,$z_0$为一个全零的向量,$z_{t-1}$表示decoderRNN上一层的隐藏状态。根据每一层的隐藏状态，我们可以得到$y_t$的条件概率为最后的输出结果（每个单词的概率）:
$$P(y_t|y_{<t},x) = softmax(W_sz_t + b_z) $$

当然在机器翻译中如何选择最优的翻译结果，还有一套方法，不是本文重点，就不记录了，大家可以去研究研究

![decoder过程](/img/decoder过程.png)


#### 综合模型
一个完整的Encoder-Decoder模型就是讲上述的两个过程相结合，结构如下：

![完整模型](/img/encoder-decoder.png)
    

### Attention Model

终于进入了正题了，首先再看一下Encoder的过程，Encoder需要将整个句子压缩成一个固定的向量 $c = q({h_1,...,h_T})$,然后在Decoder过程中用这个向量c解码成需要翻译的目标。对于向量c我们可以看做对整个句子信息的非线性压缩组合的结果，对于每一个需要翻译的目标，使用的是相等的信息，然而我们在人为做翻译的时候往往并不如此。比如像下面这样一句英文长句：
    
    *There is no difference, but there is just the same kind of difference, between the mental occupations of a man of science and those of an ordinary person，as there is between the operations and methods of a bake or of a butcher weighing out his goods in common scales, and complex analysis by means of his balance and finely graded weights.*

我们人为在翻译时不会直接将整个句子直接进行翻译，而是在整句阅读后依据句子逻辑来对句子进行拆分然后再进行翻译。依据我们人类的经验而言很明显，我们这种翻译的方式一定会更加准确和高效。很多研究者的实验结果，以BLEU作为衡量翻译效果的指标，可以发现随着句子长度的增加，翻译的效果是下降的：

![BLEU](/img/BLEU.png)
    
因此我们需要建立一种方式，能够表达这种人类在翻译过程中甚至是很多人类活动中这种“分段识别”的方法，被称为“注意力”，也就是我们说的Attention Model。

我们回到Decoder过程中，原模型中用于编码的LSTM网络传递的中间状态的计算方式为：$z_t=\Theta(c,y_{t-1},z_{t-1})$，c为之前解码过程中生成的固定长度的向量。我们想要引入注意力机制，因此对于不同的输入状态{$ y_{t-1},z_{t-1} $}，采用不同的的向量c，因此中间状态变为：

$$ z_t=\Theta(c_{t},y_{t-1},z_{t-1}) $$
    
其中$ c_t $的计算公式为$ c_t = \displaystyle{\sum_{j = 1}^{T_x}{\alpha_{tj}h_j}} $,
$ \alpha_{tj} $的计算公式为$ \alpha_{tj} = \frac{exp(e_{tj})}{\sum_{k=1}^{T_x}exp(e_{tk})} $,
$ e_{tj} $的计算公式为$ e_{tj} = a(z_{t−1}, h_j) $

综上所述网络的主体结构如下：
![网络结构](/img/NetWork.png)


#### 双向循环网络--$ h_j $

通常的RNN，如式上面Encoder-Decoder所述，按照从第一个符号$ X_1 $到最后一个符号$ X_Tx $的顺序读取输入序列X。然而，在建议的方案中，我们希望每个单词的注释不仅要总结前面的单词，还要总结下面的单词。因此，论文提出了使用双向RNN的方式。对于双向RNN模型，每层的隐藏状态表达如下,这里字符不好表达，还请谅解。

$$ h_j^{left} = f(W_{x}^{left}X_j+W_h^{left}h_{j-1}^{left}) $$

$$ h_j^{right} = f(W_{x}^{right}X_j+W_h^{right}h_{j-1}^{right}) $$

$$ h_j = [h_j^{left},h_j^{right}] $$
    
#### RNN神经网络--加入门控

对于解码过程中RNN的神经元，论文使用了门控隐藏单元，类似于Hochreiter和Schmidhuber(1997)早先提出的lstm单元，与它共享更好的建模和学习长期依赖关系的能力，这样一定程度上避免了RNN网络在反向传播过程中容易出现的梯度消失的问题。首先是两个门控单元$\Gamma_t^z$是更新门，表示每个隐藏单元保持其先前的激活的比例,$\Gamma_t^r$是重置门，表示应该重置多少来自上一层隐藏状态的信息：

$$ {\Gamma_t^z} = \sigma({W_z}e(y_t)+U_zz_{t-1}+C_zc_t) $$

$$ {\Gamma_t^r} = \sigma({W_r}e(y_t)+U_rz_{t-1}+C_rc_t) $$   
    
计算隐藏层的传递状态$ z_t $:

$$ z_t' = tanh({W}e(y_{t-1})+U[\Gamma_t^r * z_{t-1}]+Cc_t) $$

$$ z_t = [(1-{\Gamma_t^z}) * z_{t-1}] + [{\Gamma_t^z} * z_t'] $$  

其中[x * y]是点乘运算
    
#### 对齐模型

对齐模型最主要的就是我们需要将$ T_x $的词向量翻译成$ T_y $。根据上述的过程，我们需要将encoder过程的结果与decoder的输入相匹配，也就是“对齐”

$$ \alpha(z_{t-1},h_j) = v_{\alpha}tanh(W_{\alpha}z_{t-1} + U_{\alpha}h_j{}) $$
    
---

# 2.Show, Attend and Tell:Neural Image Caption Generation with Visual Attention

### 背景
attention机制不仅仅用在了机器翻译领域，对其他领域也有影响。这篇文章是在image caption任务下加入了attention机制，对该领域产生了深远影响。

Image caption顾名思义，即可让算法根据输入的一幅图自动生成对应的描述性文字。有点类似于看图说话的感觉，该任务类似于机器翻译，网络主要结构也是encoder-decoder这一系列模型以及它们的变体。与机器翻译的区别在于编码结果是用CNN将图片信息提取的结果。图片描述早期做法如下图所示：
![image caption](/img/SAT/image_caption.png)

其中，show and tell 是早期提出的一种解决image caption的一种有效的网络结构，结构如下：
![show and tell](/img/SAT/show_and_tell.png)

### 加入attention机制的image caption
本篇文章在show and tell的基础上加入了attention的机制，并且提出了两种attention形式，在实验数据集上效果有显著提升。网络主体结构如下：
![show attend and tell](/img/SAT/show_attend_and_tell.png)

#### ENCODER: CONVOLUTIONAL FEATURES
模型输入时一张原始图片，输出是1-of-K单词序列编码y,K是词汇表大小，C是y的长度

$$ y = \{y_1,...,y_c\},y_i \in R^K $$

这篇文章使用卷积神经网络来提取出一组特征向量作为对图片的描述，这组向量包含L个向量，每个向量维度为D。不同于以往采用全连接层作为图像特征，这次是直接使用卷积层conv5_3作为特征。

$$ a = \{a_1,...,a_L\},a_i \in R^D $$

#### DECODER: LONG SHORT-TERM MEMORY NETWORK
解码方式与第一篇类似，RNN单元使用的是lstm。

#### attention机制
attention机制与第一篇文章类似，本篇论文表达如下：

$$ e_{ti} = f_{att}(a_i,h_{t-1}) $$

$$ \alpha_{ti} = \frac{exp(e_{ti})}{\sum_{k = 1}^{L} {exp(e_{tk})}} $$

$$ \hat{z}_t = \phi(\{a_i\},\{\alpha_i\}) $$

本篇文章状态初始化如下：

$$ c_0 = f_{init,c}({\frac{1}{L}{\sum_{i}^{L} {a_i}}}) $$

$$ h_0 = f_{init,h}({\frac{1}{L}{\sum_{i}^{L} {a_i}}}) $$

最后采用deep output layer来计算对应位置的单词条件概率

$$ p(y_t|{a,{y^{t-1}_1}}) \varpropto exp(L_o(Ey_{t-1} + L_h h_t + L_z \hat{z}_t)) $$ 

### Learning Stochastic “Hard” vs Deterministic “Soft” Attention
本篇文章主要采用两种形式的attention机制$ f_{att} $：随机hard attention 和确定soft attention

#### Stochastic “Hard” Attention
定义一个变量$ s_t $ 来表示在生成第t个单词时，模型的注意力应该置于何处。$ s_{ti} $ 是一个one-hot向量，来表示是否使用第i个位置上的图像特征。在这里，假设$ s_{ti} $服从于一个参数为$ \{ \alpha_i\} $ 的multinoulli分布，$ \hat{z}_t $是一个随机变量
![equ_8_9](/img/SAT/equ_8_9.png)

显然该任务的目标是使得$ p(y | a) $ 最大，本篇论文定义一个新的目标函数$ L_s $,这个目标函数是原始目标函数的边缘对数似然函数的变分下界，在优化过程中，可以对新的目标函数求导
![equ_10_11](/img/SAT/equ_10_11.png)

由于$ s_{ti} $是服从于multinoulli分布，因此可以使用蒙特卡罗方式估计梯度
![equ_12](/img/SAT/equ_12.png)

为了降低估计变量的方差，采用了移动平均估计，训练过程中，第k个mini-batch的参数设定为：
![b_k](/img/SAT/b_k.png)

另外为了减少方差，文章中还加入了multinoulli分布的熵函数$ H[s] $,最终梯度计算公式如下,其中$ \lambda_r,\lambda_e$是超参数，需要自己设定
![Ls_W](/img/SAT/Ls_W.png)

#### Deterministic “Soft” Attention
学习随机注意机制需要每次对位置注意力进行抽样，我们可以直接使用$\hat{z}_t$ 的期望,和上篇文章attention一致，现在模型是光滑的，可以使用反向传播进行求解。学习这个软注意机制可以被理解为近似地优化原始目标函数$ p(y | a) $的对数似然
![equ_13](/img/SAT/equ_13.png)

本文定义一个归一化加权几何平均值NWGM，$n_t = L_o(Ey_{t-1} + L_h h_t + L_z \hat{z}_t)$：
![NWGM](/img/SAT/NWGM.png)
NWGM是一个softmax单元出来的结果，并且有：
![NWGM_1](/img/SAT/NWGM_1.png)
这就意味着学习这个软注意机制近似于优化原始目标函数$ p(y | a) $的对数似然.

#### DOUBLY STOCHASTIC ATTENTION
上述结果可以看到在生成第t个单词时，每个位置的权重和为1,$ \sum_i {\alpha_{ti}} = 1$,这是因为每个位置的权重是softmax结果。在训练soft attention时，本文引入了一个双重随机正则,还希望在每个位置上,生成不同的单词时的权重和也尽量为1,$ \sum_t {\alpha_{ti}} \approx 1$,目的是在整个生成过程中，让模型平等地对待图片的每一区域。

另外本文还定义了阈值：

$$ \beta  = \sigma(f_{\beta}(h_{t-1}))$$

本文发现该$ \beta $可以使得attention重点放在图片中的目标上：

$$ \hat{z}_t = \Psi(\{\alpha_i\},\{a_i\}) = \beta \sum_{i}^{L}{\alpha_i a_i} $$

最终目标函数为：

$$ L_d = -log(p(y | x)) + \lambda {\sum_{i}^{L} {(1 - \sum_{t}^{C} {\alpha_{ti}}})^2} $$

---

# 3.Attention Is All You Need
## 背景
机器翻译的网络大多数建立在RNN上使用attention，这样的网络在计算上没有办法进行并行运算并且结构较复杂，所以本文提出了一种完全依赖于attention的网络叫做Transfomer。Transfomer是第一个完全依赖于自我注意来计算其输入和输出的表示，而不使用序列对齐的RNNs或卷积的转换模型。

## Model Architecture
本文Transfomer整体模型结构如下：
![fig_1](/img/AYNIA/fig_1.png)

### Encoder and Decoder Stacks

#### Encoder
Encoder是由6个相同的组件组成，每个组件包含两种隐藏层，一种是multi-head self-attention机制，另一种是优化后的全连接前向传播网络层。每个隐藏层之间加入了残差结构和标准化结构:

$$ LayerNorm(x + Sublayer(x)) $$

#### Decoder
Decoder也是由6个相同的组件组成，每个组件包含3个隐藏层，其中两个与Encoder中相同，另外还加入了与Encoder输出相连接的attention层，同样每个隐藏层都加入了残差和标准化结构。并且为了确保位置i的预测只能依赖于位置i以下的已知输出，对Decoder的self-attention进行了mask
![encoder_6_decoder](/img/AYNIA/encoder_6_decoder.png)
![encoder_decoder](/img/AYNIA/encoder_decoder.png)

### Attention
本文的attention机制叫做Multi-Head Attention，它是由h个Scaled Dot-Product Attention构成的，如图：
![Multi-Head](/img/AYNIA/Multi-Head.png)

#### Scaled Dot-Product Attention
一个attention函数可以描述为将Q(query)和一组键K(key)-值V(value)对 映射到输出，其中Q、K、V和输出都是向量。其中最重要的一步是计算Q和K的相似度，其中常见的相似度计算方式有如下几种,本文选择的是第一种方式，并且对相似度进行了标准化处理(除以$ \sqrt{d_k} $)。之所以选择第一种，是因为相对于加法运算来说，虽然时间复杂度一样，但点乘计算在计算上更快，内存空间更有效。本文发现在$ d_k$较小时，点乘模型和加法模型在效果上近似，当$ d_k$较大时，点乘模型比加法模型效果差，怀疑是由于点乘量级过大导致softmax梯度过小，所以将相似度计算除以了$ \sqrt{d_k} $，所以最终attention模型为：
![Q-K](/img/AYNIA/Q-K.png)
![atten](/img/AYNIA/atten.png)

#### Multi-Head Attention
公式为：
![multi-head-eq](/img/AYNIA/multi-head-eq.png)
它是将多个$ head_i$连接到一起，再全连接（线性变换）到输出结果。其中$ head_i$是由输入Q、K、V进行一次线性变换之后的Scaled Dot-Product Attention结果。

#### Applications of Attention in our Model
在encoder-decoder attention层，Q来自于上一层decoder输出，K和V来自于encoder输出（K和V一样);<br>
在encoder层是self-attention，Q、K、V相同，都是上一层encoder输出结果;<br>
在decoder层也有self-attention层，Q、K、V相同，都是上一层decoder输出结果，不同的是这里加入了mask。
![mask](/img/AYNIA/mask.png)

### Position-wise Feed-Forward Networks
公式如下：
![eq_2](/img/AYNIA/eq_2.png)

### position embedding
论文采用self-attention的一个缺点就是无法使用word的位置信息。RNN可以使用位置信息，因为当位置变化时（比如互换两个word的位置）RNN的输出就会改变。而self-attention各个位置可以说是相互独立的，输出只是各个位置的信息加权输出，并没有考虑各个位置的位置信息。CNN也有类似位置无关的特点。以往的pe往往采用类似word_embedding式的可以随网络训练的位置参数，google提出了一种固定的pe算法：
![pe_sin](/img/AYNIA/pe_sin.png)
即在偶数位置，此word的pe是sin函数，在奇数位置，word的pe是cos函数。论文说明了此pe和传统的训练得到的pe效果接近。并且因为 sin(α+β)=sinα cosβ+cosα sinβ 以及 cos(α+β)=cosα cosβ−sinα sinβ，位置 p+k 的向量可以用位置 p 的向量的线性变换表示，这也说明此pe不仅可以表示绝对位置，也能表示相对位置。

---
参考博客（待整理）

