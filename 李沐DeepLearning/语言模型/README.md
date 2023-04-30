### 一. 语言模型的概念
在自然语言处理(NLP)中，语言模型是一种用来对语言进行建模的统计模型。其主要目的是计算给定一段文本序列的概率值或对下一个词或字符的预测值。

语言模型通常基于概率模型来构建，它考虑了语言的各种特征，例如语法、语义和上下文。具体来说，语言模型可以根据一定的训练数据学习到一个概率分布，该分布可以描述一个给定的文本序列中每个单词出现的概率，或者是下一个单词的预测概率。这些概率可以用来评估一个给定的文本序列是否合理，或者给出一个可能的下一个单词或短语。

### 二. 学习语言模型
首先，从基本的概率规则开始：假设给定一个长度为$T$的文本序列，文本序列中的词元依次为$x_1,x_2,...,x_T$。于是 $x_t(1\le t \le T)$可以被认为是文本序列在时间步$t$的观测或是标签。则语言模型的目标是估计文本序列的联合概率：

$P(x_1,x_2,...,x_T)=\prod_{t=1}^{T}P(x_t|x_{t-1},...,x_2,x_1)$ (1)

为了训练语言模型，需要计算一个单词的概率以及给定前面若干个词的情况下出现某个单词的条件概率。这些概率本质上就是语言模型的参数。

在这里列出条件概率的定义：

条件概率（Conditional Probability）是指在已知某个事件发生的前提下，另一个事件发生的概率。一般来说，它是指一个事件B在已知事件A发生的条件下发生的概率，用$P(B|A)$来表示。

例如，以下为包含了四个单词的文本序列的概率：

$P(deep,learning,is,fun)=P(deep)P(learning|deep)P(is|deep,learning)P(fun|deep,learning,is)$

接下来，可以估计文本中所有以deep为开头序列的概率，记为$\hat{P}(deep)$，则有：

$\hat{P}(deep)=\frac{n(deep)}{n(total)}$

推广到一对连续的单词对，现在计算$\hat{P}(learning|deep)$，有：

$\hat{P}(learning|deep)=\frac{n(deep,learning)}{n(deep)}$

其中$n(x)$和$n(x,x')$分别是单个词和一对连续单词出现的次数。由于在文本序列中，类似“deep learning”这样的连续单词对出现的次数相对于单个词的出现次数低得多，要准确预测此类单词对的出现概率十分困难。对于三个以上的词组组合，对于这种词组组合的概率预测更加困难。

下面提供了一种方法，对于两个以及两个以上的词组组合的计数中分别添加一个小常量。这种方法被称为拉普拉斯平滑（Lapalace Smoothing）。但是这种方法很容易使得模型变得无效。首先，所有的计数都需要被储存，计算量大；其次，这种方法忽略了单词本身的意思，完全使用数字去计算概率使得无法根据单词的上下文调整；最后 ，很多包含多个单词的词组是没出现过的，如果只统计之前看到过的单词也就无法正确预测某些之前没出现过的长序列词组组合。

因此，下面引入了马尔可夫模型来解决长序列单词组和的概况计算问题。

### 三. 马尔可夫模型以及n元语法
马尔可夫模型（Markov model）是一种基于马尔可夫过程的概率模型。它描述了一个随机过程中，每个状态的转移是基于前一个状态的概率分布，并且当前状态只与前一个状态有关，而与之前的状态无关。

马尔可夫模型可以分为不同的类型，如一阶马尔可夫模型、二阶马尔可夫模型等。一阶马尔可夫模型是指当前状态只与前一个状态有关，而二阶马尔可夫模型是指当前状态不仅与前一个状态有关，还与前两个状态有关。

把马尔可夫模型应用到语言模型当中，若$P(x_{t+1}|x_t,...x_1)=P(x_{t+1}|x_t)$ ,即序列在时间步$t+1$处的观测只与序列在时间步$t$处的观测有关，则序列上的分布满足一阶马尔可夫性质。阶数越高，对应的依赖关系就越长。由此可推导出了许多可以应用于序列建模的近似公式：

$P(x_1,x_2,x_3,x_4)=P(x_1)P(x_2)P(x_3)P(x_4)$

$P(x_1,x_2,x_3,x_4)=P(x_1)P(x_2|x_1)P(x_3|x_2)P(x_4|x_3)$  (2)

$P(x_1,x_2,x_3,x_4)=P(x_1)P(x_2|x_1)P(x_3|x_1,x_2)P(x_4|x_2,x_3)$

通常，涉及一个、两个和三个变量的概率公式分别被称为一元语法（unigram）、二元语法（bigram）和三元语法（trigram）模型。

接下来将使用代码实现语言模型。

### 四. 自然语言统计
```python
tokens = d2l.tokenize(d2l.read_time_machine()) #提取词元
corpus = [token for line in tokens for token in line] #生成语料库
vocab = d2l.Vocab(corpus) #构造词汇表

print(vocab.token_freqs[:10])
```
这里使用了Dive to Deep Learning的d2l库方便读入文本数据和构造词汇表。

在有了词汇表之后就可以画词频图
```python
freq = [freq for token, freq in vocab.token_freqs]
d2l.plot(freq, xlabel='token: x', ylabel='freq: y', xscale='log', yscale='log')
```
<div align=center>
<img src="https://github.com/pod2c/Machine_Learning/blob/87ead1b3081c0ec2111426a0d759b86db694e3c6/%E6%9D%8E%E6%B2%90DeepLearning/%E5%9B%BE%E7%89%87/%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B/fig1.png"/></br>
图1：一元语法词频图
</div>


某些出现频次过高的词元会被划分为停用词（Stop Words）。停用词是指在文本处理中经常要忽略的词汇，因为这些词通常不对文本的意义产生重要贡献。常见的停用词包括代词、介词、连词、冠词等。另外，在英文中还有一些高频词如 "the" "and" "a" 等被认为是停用词。

接着是二元语法和三元语法的词频，即数据集中两个词元为一组的组合和三个词一组的组合出现的频率。
```python
#二元语法
bi_tokens = [pair for pair in zip(corpus[:-1], corpus[1:])]
bi_vocab = d2l.Vocab(bi_tokens)
bi_vocab.token_freqs[:10]
#三元语法
tri_tokens = [triple for triple in zip(corpus[:-2], corpus[1:-1], corpus[2:])]
tri_vocab = d2l.Vocab(tri_tokens)
tri_vocab.token_freqs[:10]
```
画出词频图
```python
bi_freq = [freq for token, freq in bi_vocab.token_freqs]
tri_freq = [freq for token, freq in tri_vocab.token_freqs]
d2l.plot([freq, bi_freq, tri_freq], xlabel='token: x', ylabel='freq: y', xscale='log', yscale='log', legend=['uni_freq','bi_freq','tri_freq'])
```
<div align=center>
<img src="https://github.com/pod2c/Machine_Learning/blob/87ead1b3081c0ec2111426a0d759b86db694e3c6/%E6%9D%8E%E6%B2%90DeepLearning/%E5%9B%BE%E7%89%87/%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B/fig2.png"/></br>
图2：词频图
</div>

通过词频图可以看出，词频以一种明确的方式衰减。在剔除前几个单词后，后续的单词遵循双对数坐标上的一条直线衰减。这意味着词频满足齐普夫定律（Zipf's Law），即第 
 个词的词频 
 为：

 $n_i\propto \frac{1}{i^{\alpha} }$ (3)

 等价于

$\log n_i=-\alpha \log i + c$  (4)

其中$\alpha$是刻画分布的指数，$c$是常数。

这告诉我们想要通过计数统计和平滑来建模单词是不可行的， 因为这样建模的结果会大大高估尾部单词的频率，也就是所谓的不常用单词。

### 五. 读取长序列数据
在模型训练长序列数据时，需要将这些长序列数据拆分开变成几段短序列方便模型读取。模型一次处理具有预定义长度的一个小批量序列。现在需要解决的是如何在原始文本序列中随机生成一个小批量数据的标签和特征以供读取。

由于文本序列的长度可以被任意分割，在这里可以定义一个时间步数 
 ，利用这个时间步数将文本序列分割为若干个具有相同时间步数的子序列。并且可以任意选择偏移量来指示分割开始的初始位置。例子如下，设$n=5$，则有

<div align=center>
<img src="https://github.com/pod2c/Machine_Learning/blob/87ead1b3081c0ec2111426a0d759b86db694e3c6/%E6%9D%8E%E6%B2%90DeepLearning/%E5%9B%BE%E7%89%87/%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B/fig3.png"/></br>
图3：分割出来的子序列（图源：Dive to Deep Learning）
</div>

如图3，不同的偏移量会导致产生不同的子序列。为了保证随机性，在这里选择随机偏移量作为起始位置。下面将实现随机采样和顺序分区来分割文本序列。

### 六. 随机采样
在随机采样（random sampling）中，每个子序列都是在原始长序列上任意捕获的短序列。在每次采样的过程中，采样之后两个相邻的子序列在原始长序列上不一定是相邻的。对于语言模型，特征（feature）是到目前为止能观测到的词元，而标签（label）则是位移了一个词元的原始序列。
```python
def seq_data_iter_random(corpus, batch_size, num_steps): #num_steps为随即偏移量
    # 依据随即偏移量，对数据集进行顺序分区
    corpus = corpus[random.randint(0, num_steps-1):] 
    # 计算子序列的数量
    num_subseqs = (len(corpus)-1) // num_steps
    # 建立长度为num_steps的子序列的起始索引
    indices = list(range(0, num_subseqs * num_steps, num_steps))
    random.shuffle(indices)
    
    def data(pos):
        # 返回指定区间长度的序列
        return corpus[pos: pos + num_steps]
    
    num_batches = num_subseqs // batch_size
    for i in range(0, num_batches * batch_size, batch_size): 
        #每个batch的起始位置
        iter_indices_per_batch = indices[i: i + batch_size]
        X = [data(j) for j in iter_indices_per_batch]
        Y = [data(j + 1) for j in iter_indices_per_batch]
        # 保留一次迭代的特征和标签
        yield torch.tensor(X), torch.tensor(Y)
```
下面随机生成一个任意长度的序列来验证随机采样的效果。

设序列长度为34，时间步长为5，batch_size为2
```python
seq = list(range(34))
for X, Y in seq_data_iter_random(seq, 2, 5):
    print("X:",X, "\nY:", Y)
```
运行结果：

<div align=center>
<img src="https://github.com/pod2c/Machine_Learning/blob/87ead1b3081c0ec2111426a0d759b86db694e3c6/%E6%9D%8E%E6%B2%90DeepLearning/%E5%9B%BE%E7%89%87/%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B/fig4.png"/></br>
图4：随机采样
</div>

从图4可以看出，一共生成了3组子序列，并且每一组序列里的特征都是随机采样的，任意相邻的两条子序列在原始长序列中不相邻。

### 七. 顺序分区
在随机采样中，得到的两个相邻的子序列之间在原始序列上是不相邻的。如果想要得到两个在原始序列上也相邻的子序列则需要使用顺序分区（Sequential Partitioning）。这种方法在基于小批量的迭代过程中保留了拆分的子序列的顺序，因此被称为顺序分区。
```python
def seq_data_iter_sequential(corpus, batch_size, num_steps):
    #设置随机偏移量
    offset = random.randint(0, num_steps) 
    #计算使用偏移量分割之后每个步长内的词元数量
    num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size 
    #储存按照偏移量分割之后的原始长序列
    Xs = torch.tensor(corpus[offset: offset + num_tokens])
    Ys = torch.tensor(corpus[offset + 1: offset + 1 + num_tokens])
    Xs, Ys = Xs.reshape(batch_size, -1), Ys.reshape(batch_size, -1)
    #计算一共产生的batch数量
    num_batches = Xs.shape[1] // num_steps
    for i in range(0, num_batches * num_steps, num_steps):
        #从之前储存的分割完成的原始长序列当中按时间步长提取子序列
        X = Xs[:, i:i + num_steps]
        Y = Ys[:, i:i + num_steps]
        yield X, Y
```
与随机采样不同的是，顺序分区在最后迭代提取一对子序列的时候，前一个子序列保持不变，而它的相邻序列会直接从在原始序列上与其相邻的下一个词元开始提取，这样就能保证提取出来的两个子序列在原始序列上也是相邻的。

在之前生成的随机长序列上测试效果：

<div align=center>
<img src="https://github.com/pod2c/Machine_Learning/blob/87ead1b3081c0ec2111426a0d759b86db694e3c6/%E6%9D%8E%E6%B2%90DeepLearning/%E5%9B%BE%E7%89%87/%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B/fig5.png"/></br>
图5：顺序分区
</div>

由图5可知，同样一共生成了三组子序列，可以看出其中两两相邻的子序列在原始长序列上也是相邻的。

接着将随机采样函数和顺序分区采样函数整合到一个类当中，做成一个数据迭代器：
```python
class SeqDataLoader:
    def __init__(self, batch_size, num_steps, use_random_iter, max_tokens):
        if use_random_iter:
            self.data_iter_fn = d2l.seq_data_iter_random
        else:
            self.data_iter_fn = d2l.seq_data_iter_sequential
        self.corpus, self.vocab = d2l.load_corpus_time_machine(max_tokens)
        self.batch_size, self.num_steps = batch_size, num_steps

    def __iter__(self):
        return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)
```
最后定义一个load_data_time_machine()函数，使得能同时返回数据迭代器（采样器）和词汇表：
```python
def load_data_time_machine(batch_size, num_steps, use_random_iter=False, max_tokens=10000):
    data_iter = SeqDataLoader(batch_size, num_steps, use_random_iter, max_tokens)
    return data_iter, data_iter.Vocab
```
