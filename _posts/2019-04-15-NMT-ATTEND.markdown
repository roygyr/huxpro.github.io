---
layout:     post
title:      "论文阅读笔记"
subtitle:   " \"Neural Machine Translation by Jointly Learning to Align and Translate\""
date:       2019-04-15 10:30:00
author:     "Roy严皓越"
header-style: text
header-img: "img/post-bg1.jpg"
mathjax: true
catalog: true
tags:
    - 论文笔记
    - attention model
    - image caption
    
---


> “机器学习整理面试常见问的整理先跳票，先开一篇论文的笔记. ”


## 前言
说好的机器学习面试问题整理成功跳票，意料之内，哈哈哈。先开一篇论文的阅读笔记，准确的说是两篇一起:

第一篇是:[Show, Attend and Tell: Neural Image Caption Generation with Visual Attention](https://arxiv.org/abs/1502.03044v1)
是Image caption任务的论文，他在之前研究的基础上加入了Attention Model的思想,从而带来了较好的效果。（所以第二篇的思想才是重点，我也会先看第二篇）

第二篇是:[Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473v2)是Attention Model的开篇之作，虽然本文主要应用的是机器翻译的领域，但是之后对各个领域都产生的深远影响，并且产生了很多变形与应用。

---

# Neural Machine Translation by Jointly Learning to Align and Translate

### 背景

神经机器翻译是前些年来提出的一种新兴的机器翻译方法，神经机器翻译试图建立和训练一个单一的、大型的神经网络，它能读取一个句子并输出正确的翻译。所提出的神经机器翻译模型大多属于一类编码器解码器(Encoder-Decoder)，具体关于机器翻译的内容还有更多的应用，但不是本次研究的重点，以后有机会再学习。

### Encoder-Decoder模型

#### The Encoder
对于一个长度为T的文本序列X $ = (x_1,x_2,...,x_T) $
,中每一个词都被表示成一个向量 $ \omega_i \in R^v $,i=1,2...T。通常这个$ \omega_i $是一条one_hot向量。接下来就是要对要将每个词映射到一个低维的语义空间，也就是一个词向量。记映射矩阵为$ C \in R^{KV} $,则第$i$个词表示的词向量为，$s_i = C\omega_i$，向量维度*K*通常的取值在100到500之间。这个词向量在整个模型训练的过程中也会逐步更新，会变得更加有意义。

![encoder过程](/img/encoder过程.png)

紧接着就可以用一个RNN模型压缩表示语言序列：$$h_i = \Phi_\theta(h_{i-1},s_i)=\Theta(Ws_i+Uh_{i-1}+b)$$

其中$\Theta$表示隐含层的激活函数，常见的有sigmoid、tanh等。$h_T$表示整个句子的压缩结果，为了探究$h_T$的性质Sutskever et al.2014利用主成分分析找到向量$h_T$的两个主成分，并画出下图。

![主成分](/img/pca.png)

从图中可以看出，压缩向量确实可以包含原语句中的语义信息，在二维的空间表达中，距离越近语义越接近。

#### The Decoder

decoder部分同样实用RNN实现，根据encoder，我们得到了一个句子的压缩表示$c = q({h_1,...,h_T})$,接着就开始计算RNN的隐藏状态$z_t,t=1,...,T$,z_t的计算公式如下：

$$z_t=\Theta(c,y_{t-1},z_{t-1})$$

其中$y_0$是结束标记“<EOS>”用以表示解码的开始与结束，$z_0$为一个全零的向量。根据每一层的隐藏状态，我们可以得到$y_t$的条件概率为最后的输出结果:
$$P(y_t|y_{<t},x) = softmax(W_Sz_t + b_s) $$

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

$$ z_t=\Theta(c_{i},y_{t-1},z_{t-1}) $$
    
其中$ c_i $的计算公式为$ c_i = \displaystyle{\sum_{j = 1}^{T_x}{a_{ij}h_j}} $,
$ a_{ij} $的计算公式为$ a_{ij} = \frac{exp(e_{ij})}{\sum_{k=1}^{T_x}exp(e_{ik})} $,
$ e_{ij} $的计算公式为$ e_{ij} = a(z_{i−1}, hj) $

综上所述网络的主题结构如下：
![网络结构](/img/NetWork.png)


#### 双向循环网络--$ h_j $

通常的RNN，如式上面Encoder-Decoder所述，按照从第一个符号$ X_1 $到最后一个符号$ X_Tx $的顺序读取输入序列X。然而，在建议的方案中，我们希望每个单词的注释不仅要总结前面的单词，还要总结下面的单词。因此，论文提出了使用双向RNN的方式。对于双向RNN模型，每层的隐藏状态表达如下,这里字符不好表达，还请谅解。

$$ h_t^{left} = f(W_{x}^{left}X_t+W_h^{left}h_{t-1}^{left}) $$

$$ h_t^{right} = f(W_{x}^{right}X_t+W_h^{right}h_{t-1}^{right}) $$

$$ h_t = [h_t^{left},h_t^{right}] $$
    
#### RNN神经网络--加入门控

对于解码过程中RNN的神经元，论文使用了门控隐藏单元，类似于Hochreiter和Schmidhuber(1997)早先提出的长短期内存(LSTM)单元，与它共享更好的建模和学习长期依赖关系的能力，这样一定程度上避免了RNN网络在反向传播过程中容易出现的梯度消失的问题。首先是两个门控单元：

$$ {\Gamma_i^z} = \sigma({W_z}e(y_i)+U_zz_{t-1}+C_zc_i) $$

$$ {\Gamma_i^r} = \sigma({W_r}e(y_i)+U_rz_{t-1}+C_rc_i) $$   
    
计算隐藏层的传递状态$ z_i $:

$$ z_i' = tanh({W}e(y_i)+U[\Gamma_i^r * z_{t-1}]+Cc_i) $$

$$ z_i = [(1-{\Gamma_i^z}) * z_{i-1}] + [{\Gamma_i^z} * z_i'] $$  

其中[x * y]是点乘运算
    
#### 对齐模型

对其模型最主要的就是我们需要将$ T_x $的词向量翻译成$ T_y $。根据上述的过程，我们需要将encoder过程的结果与decoder的输入相匹配，也就是“对齐”

$$ \alpha(z_{i-1},h_j) = v_{\alpha}tanh(W_{\alpha}z_{i-1} + U_{\alpha}h_j{}) $$
    
---

# Show, Attend and Tell:Neural Image Caption Generation with Visual Attention

### 背景
attention机制不仅仅用在了机器翻译领域，对其他领域也有影响。这篇文章是在image caption任务下加入了attention机制，对该领域产生了深远影响。

Image caption顾名思义，即可让算法根据输入的一幅图自动生成对应的描述性文字。有点类似于看图说话的感觉，该任务类似于机器翻译，网络主要结构也是encoder-decoder这一些列模型以及他们的变体。与机器翻译的区别在于编码结果是用CNN将图片信息提取的结果。图片描述早期做法如下图所示：
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

$$ e_{ti} = f_att(a_i,h_{t-1}) $$

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
学习随机注意机制需要每次对位置注意力进行抽样，我们可以直接使用$ \hat{z}_t $ 的期望,和上篇文章attention一致，现在模型是光滑的，可以使用反向传播进行求解。学习这个软注意机制可以被理解为近似地优化原始目标函数$ p(y | a) $的对数似然
![equ_13](/img/SAT/equ_13.png)

本文定义一个归一化加权几何平均值NWGM，：
![NWGM](/img/SAT/NWGM.png)
NWGM是一个softmax单元出来的结果，并且有：
![NWGM_1](/img/SAT/NWGM_1.png)
这就意味着学习这个软注意机制近似于优化原始目标函数$ p(y | a) $的对数似然

#### DOUBLY STOCHASTIC ATTENTION
上述结果可以看到在生成第t个单词时，每个位置的权重和为1,$ \sum_i {\alpha_{ti}} = 1$，这是因为每个位置的权重是softmax结果。在训练soft attention时，本文引入了一个双重随机正则，还希望在每个位置上，生成不同的单词时的权重和也尽量为1，$ \sum_t {\alpha_{ti}} \approx 1$，目的是让attention平等的对待图片的每一区域。

另外本文还定义了阈值：

$$ \beta  = \sigma(f_{\beta}(h_{t-1}))$$

本文发现该$ \beta $可以使得attention权重重点放在图片中的目标上

$$ \hat{z}_t = \Psi\(\{\alpha_i\},\{a_i\}) = \beta \sum_{i}^{L}{\alpha_i a_i} $$

最终目标函数为：

$$ L_d = -log(p(y | x)) + \lambda {\sum_{i}^{L} {(1 - \sum_{t}^{C} {\alpha_{ti}}})^2} $$



