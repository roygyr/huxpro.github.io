---
layout:     post
title:      "论文阅读笔记"
subtitle:   " \"Neural Machine Translation by Jointly Learning to Align and Translate\""
date:       2019-04-15 10:30:00
author:     "Roy严皓越"
header-img: "img/post-bg1.jpg"
header-mask: 0.3
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

## 背景

神经机器翻译是前些年来提出的一种新兴的机器翻译方法，神经机器翻译试图建立和训练一个单一的、大型的神经网络，它能读取一个句子并输出正确的翻译。所提出的神经机器翻译模型大多属于一类编码器解码器(Encoder-Decoder)，具体关于机器翻译的内容还有更多的应用，但不是本次研究的重点，以后有机会再学习。

### Encoder-Decoder模型

#### The Encoder
对于一个长度为T的文本序列*X $ = (x_1,x_2,...,x_T) $
,中每一个词都被表示成一个向量$ \omega_i \in R^v $,i=1,2...T。通常这个$ \omega_i $是一条one_hot向量。接下来就是要对要将每个词映射到一个低维的语义空间，也就是一个词向量。记映射矩阵为$C \in R^{K*V}$,则第$i$个词表示的词向量为，$s_i = C\omega_i$，向量维度*K*通常的取值在100到500之间。这个词向量在整个模型训练的过程中也会逐步更新，会变得更加有意义。

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

终于进入了正题了，首先再看一下Encoder的过程，Encoder需要将整个句子压缩成一个固定的向量 $c = q({h_1,...,h_T})$ ,然后在Decoder过程中用这个向量c解码成需要翻译的目标。对于向量c我们可以看做对整个句子信息的非线性压缩组合的结果，对于每一个需要翻译的目标，使用的是相等的信息，然而我们在人为做翻译的时候往往并不如此。比如像下面这样一句英文长句：
    
    *There is no difference, but there is just the same kind of difference, between the mental occupations of a man of science and those of an ordinary person，as there is between the operations and methods of a bake or of a butcher weighing out his goods in common scales, and complex analysis by means of his balance and finely graded weights.*

我们人为在翻译时不会直接将整个句子直接进行翻译，而是在整句阅读后依据句子逻辑来对句子进行翻译
    
---

#未完待续~

