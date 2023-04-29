最近在B站上跟着李沐老师学NLP，在这里把文本预处理的代码做一个小总结。

### 一. 导入文本
```python
d2l.DATA_HUB['time_machine'] = (d2l.DATA_URL + 'timemachine.txt','090b5e7e70c295757f55df93cb0a180b9691891a')

def read_book():
    with open(d2l.download('time_machine'), 'r') as f:
        lines = f.readlines()
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]

lines = read_book()
print(lines[0])
print(lines[10])
```

这里使用了Dive to Learning提供的d2l库，方便导入文本数据。

在导入文本数据后，构造一个函数来读取数据中的文本行，此处为了简化数据集，将文本中除了英文字母以外的符号全部变成空格，并且将大写字母转为小写字母。

### 二. 词元化

词元化是将一个个文本行（lines）作为输入，将文本行中的词汇拆开来变成一个个词元。词元是文本的基本单位。
```python
def tokenize(lines, token='word'):
    if (token == 'word'):
        return [line.split() for line in lines]
    elif (token == 'char'):
        return [list(line) for line in lines]
    else:
        print ('Error Token Type:' + token)

tokens = tokenize(lines)
for i in range(22):
    print(tokens[i])
```
这里构造一个tokenize函数，其输入为一个包含若干个文本行数据的列表以及一个token用作分辨词元类型。在此函数中，若token为单词（word），则使用split函数将文本行中的单词逐个拆分，然后返回一个包含若干个单词的列表；若token为字符（char），则使用list函数将文本行中的字母逐个拆分，然后返回包含若干个字母的列表；最后若输入的token无法识别则返回Error。

### 三. 构建词汇表

词元的数据类型为字符串，而深度学习模型要求的输入为数字，单纯用词元不符合模型的输入要求，需要将词元映射到从0开始的数字索引当中。首先先将所有的文本数据合并，接着对每个唯一词元进行频率统计，统计结果被称为语料库（corpus），然后对每个词元的出现频率分配一个数字索引。很少出现的词元将被删除以降低复杂性。并且对于不存在语料库中的词元或者已经删除的词元都将被映射到一个未知词元中。通常地，可以人为地增加一个列表，用于保存那些被保留的词元，例如序列开始次元表示一个句子的开始,序列结束词元表示一个句子的结束。
```python
class Vocab:
    def __init__(self, tokens=None, mini_freq=0, reserved_token=None):
        """文本词汇表"""
        if(tokens is None):
            tokens = [ ]
        if(reserved_token is None):
            reserved_token = [ ]
        counter = corpus_counter(tokens) #计算词元频率构造语料库
        self.token_freq = sorted(counter.items(), key=lambda x:x[1], reverse=True) #将词元频率按照出现频率从高到低排列
        
        self.unk, uniq_tokens = 0, ['<unk>'] + reserved_token #构造一个能够存放词元的字典
        #对于语料库中出现频率满足设定的最小频率的词元以及不在字典中的词元，逐个将这些满足条件的词元放入字典中。
        uniq_tokens += [token for token, freq in self.token_freq if freq >= mini_freq and token not in uniq_tokens] 
        self.token_to_idx = dict() #给定词元返回数字索引
        self.idx_to_token = [ ] #给定数字索引返回词元
        #将数字索引和字典中的词元一一对应
        for token in uniq_tokens:
            self.idx_to_token.append(token)
            self.token_to_idx[token] = len(self.idx_to_token) - 1
        
    def __len__(self):
        """返回储存词元字典的长度"""
        return len(self.idx_to_token) 
        
    def __getitem__(self, tokens):
        """输入一个词元，返回一个数字索引"""
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]
        
    def to_token(self, indices):
        """输入一个数字索引，返回一个词元"""
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token.get[indices]
        return [self.to_token[idx] for idx in indices]
               
def corpus_counter(tokens):
    """统计词频"""
    if (len(tokens)==0 or isinstance(tokens[0], list)): 
        """统计词元的出现频率"""
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)
```
在这部分，需要构造一个Vocab类用来处理词元和索引之间的关系和一个corpus_counter函数来统计词元的出现频率以构造语料库。首先，构造corpus_counter函数来建立语料库。对于文本数据当中的每一个唯一的词元，使用collections中的Counter()进行统计，最后返回一个词元出现频率的列表。

接着构造Vocab类，在此类中包含三大类函数：第一类函数__init__()用来定义和初始化变量。

1. 输入的变量应该是一个多个词元组成的token列表，并且设置一个mini_freq作为词元出现的最小频率用于后面过滤出现频率太少的词元以及一个用来储存保留词元的reserved_token。
2. 先使用corpus_counter函数构造一个语料库来储存词元出现频率，接着还需要对词元的出现频率按照从高到低的排序。
3. 然后声明一个unk变量，初始值为0，用来储存不在语料库中的词元和已经删除的词元。之后声明一个字典uniq_token来储存词元以及词元对应的出现频率（包括未知词元unk和保留词元reserved_token）。
4. 接着，对于语料库中出现频率满足设定的最小频率的词元以及不在字典中的词元，逐个将这些满足条件的词元放入字典中。
5. 下一步需要声明两个变量用于词元和数字索引之间的转化。
6. 最后一步，需要把词元逐个放入到idx_to_token中用于给定数字索引时返回对应词元，同时将词元对应的数字索引放入token_to_idx中用于给定词元返回数字索引。

第二类函数__len__()用来返回储存词元字典的长度。

第三类函数包含两个函数__getitem__()和to_token()。__getitem__()用来给定一个词元返回对应的数字索引，to_token()用来给定一个数字索引返回对应的词元。

### 四. 加入真实数据集

这一步，使用之前导入的时光机器数据来构造词汇表，并且打印部分高频词元。
```python
vocab = Vocab(tokens)
print(list(vocab.token_to_idx.items())[:10])
```
运行结果：



### 五. 将文本行转化为数字索引列表
```python
for i in [0, 10]:
    print('word:',tokens[i])
    print('index:',vocab[tokens[i]])
```

运行结果：



### 六. 整合所有功能
现在将之前的所有功能合并到一个函数load_corpus_time_machine()当中，此函数最终返回一个词元的索引列表corpus和一个词汇表vocabu。
```python
def load_corpus_time_machine(max_token=-1):
    lines = read_book() #导入文本数据
    tokens = tokenize(lines, 'char') #拆分文本数据转为词元
    vocabu = Vocab(tokens) #构造词汇表
    corpus = [vocabu[token] for line in tokens for token in line] #得到词元索引列表
    
    if (max_token > 0):
        corpus = corpus[:max_token] #按设置好的数量提取需要用来训练的词元
    return vocabu, corpus #返回词汇表以及数字索引列表

vocabu, corpus = load_corpus_time_machine()
len(vocabu), len(corpus)
```
需要注意的是：

1. 为了简化训练，这里使用字符（而不是单词）实现文本词元化；
2. 时光机器数据集中的每个文本行不一定是一个句子或一个段落，还可能是一个单词，因此返回的corpus仅处理为单个列表，而不是使用多词元列表构成的一个列表。