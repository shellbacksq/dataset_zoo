# 总项目

![fastModeling](../images/fastModeling.png)

## 目的：整理好一批实验项目用来学习新框架和测试

### 每份实验项目包括：
1.数据集  
包括：存放位置、大小、格式  
2.数据集描述  
包括：来源，整体描述、单个描述、直观描述  
3.实验目标  
包括：是预测还是分类等。  
4.预处理程序  
1.分为训练集、测试集的；2.batch的；3.shuffle的，4.若是文本数据有window的.



## 文本数据集
### proj1: IMDB影评倾向分类

1. 位置： /mnt/data1/imdb_full.pkl

2. 数据集描述：
  来源：http://ai.stanford.edu/~amaas/data/sentiment/  63M
  下载地址：https://s3.amazonaws.com/text-datasets/imdb_full.pkl

    整体描述：
    本数据库含有来自IMDB的25,000条影评，被标记为正面/负面两种评价。

    单个描述：
    直观描述：

3. 实验目标
  根据文本内容判断褒贬

4. 预处理程序

**使用方法**
```

from keras.datasets import imdb
(X_train, y_train), (X_test, y_test) = imdb.load_data(path="imdb.npz",
                                                      nb_words=None,
                                                      skip_top=0,
                                                      maxlen=None,
                                                      test_split=0.1)
                                                      seed=113,
                                                      start_char=1,
                                                      oov_char=2,
                                                      index_from=3)
```

参数：
path：如果你在本机上已有此数据集（位于'~/.keras/datasets/'+path），则载入。否则数据将下载到该目录下  
nb_words：整数或None，要考虑的最常见的单词数，序列中任何出现频率更低的单词将会被编码为oov_char的值。 
skip_top：整数，忽略最常出现的若干单词，这些单词将会被编码为oov_char的值  
maxlen：整数，最大序列长度，任何长度大于此值的序列将被截断  
seed：整数，用于数据重排的随机数种子  
start_char：字符，序列的起始将以该字符标记，默认为1因为0通常用作padding  
oov_char：整数，因nb_words或skip_top限制而cut掉的单词将被该字符代替  
index_from：整数，真实的单词（而不是类似于start_char的特殊占位符）将从这个下标开始  
返回值  
两个Tuple,(X_train, y_train), (X_test, y_test)，其中

X_train和X_test：序列的列表，每个序列都是词下标的列表。如果指定了nb_words，则序列中可能的最大下标为nb_words-1。如果指定了maxlen，则序列的最大可能长度为maxlen  
y_train和y_test：为序列的标签，是一个二值list





## proj2: 路透社新闻主题分类

1. 位置： /mnt/data1/reuters.pkl

2. 数据集描述：
  来源：不详 8.8M
  下载地址：https://s3.amazonaws.com/text-datasets/reuters.pkl

    整体描述：
    本数据库包含来自路透社的11,228条新闻，分为了46个主题。与IMDB库一样，每条新闻被编码为一个词下标的序列。

    单个描述：

    直观描述：

3. 实验目标
  根据文本内容分类

4. 预处理程序

**使用方法**

```
from keras.datasets import reuters
(X_train, y_train), (X_test, y_test) = reuters.load_data(path="reuters.npz",
                                                         nb_words=None,
                                                         skip_top=0,
                                                         maxlen=None,
                                                         test_split=0.2,
                                                         seed=113,
                                                         start_char=1,
                                                         oov_char=2,
                                                         index_from=3)
```


参数的含义与IMDB同名参数相同，唯一多的参数是： test_split，用于指定从原数据中分割出作为测试集的比例。该数据库支持获取用于编码序列的词下标：

word_index = reuters.get_word_index(path="reuters_word_index.json")
上面代码的返回值是一个以单词为关键字，以其下标为值的字典。例如，word_index['giraffe']的值可能为1234

**参数**
path：如果你在本机上已有此数据集（位于'~/.keras/datasets/'+path），则载入。否则数据将下载到该目录下



## 图像数据集

### proj1: mnist手写识别数据集

1. 位置： /mnt/data1/mnist

2. 数据集描述：
  来源：http://yann.lecun.com/exdb/mnist/ 12M
  下载地址：https://s3.amazonaws.com/img-datasets/mnist.pkl.gz

    整体描述
    文件 内容
    train-images-idx3-ubyte.gz 训练集图片 - 55000 张 训练图片, 5000 张 验证图片
    train-labels-idx1-ubyte.gz 训练集图片对应的数字标签
    t10k-images-idx3-ubyte.gz 测试集图片 - 10000 张 图片
    t10k-labels-idx1-ubyte.gz 测试集图片对应的数字标签

    单个描述
    每张图片由28*28个像素点组成，每个像素点的值在[0,255]之间。

    直观描述：

3. 实验目标
  根据图片数据预测该图片代表的数字。

4. 预处理程序

使用方法

```
from keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()
参数

path：如果你在本机上已有此数据集（位于'~/.keras/datasets/'+path），则载入。否则数据将下载到该目录下
返回值

两个Tuple,(X_train, y_train), (X_test, y_test)，其中

X_train和X_test：是形如（nb_samples, 28, 28）的灰度图片数据，数据类型是无符号8位整形（uint8）
y_train和y_test：是形如（nb_samples,）标签数据，标签的范围是0~9
数据库将会被下载到'~/.keras/datasets/'+path
```




## Proj2: CIFAR10/100 小图片分类数据集

1. 位置： /mnt/data1/cifar-10-python.tar.gz

2. 数据集描述：
  数据集来源：http://www.cs.toronto.edu/~kriz/cifar.html
  下载地址：
  http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz 163M
  http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz 153M

    整体描述:
    CIFAR10：该数据库具有50,000个32*32的彩色图片作为训练集，10,000个图片作为测试集。图片一共有10个类别，每个类有6000个照片。
    CIFAR100：该数据库具有50,000个32*32的彩色图片作为训练集，10,000个图片作为测试集。图片一共有100个类别，每个类别有600张图片。这100个类别又分为20个大类。

    单个描述：
    照片是三个通道的，32*32*3,其中32*32代表高度和宽度各有32个像素点，3代表有rgb三个通道。
    其中label是0-10的数字代表airplane、automobile、bird、cat、deer、dog、frog、horse、ship、truck。

    直观描述：

3. 实验目标   
  分类

4.预处理

CIFAR10  
**使用方法**

```
from keras.datasets import cifar10
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
返回值：
两个Tuple
X_train和X_test是形如（nb_samples, 3, 32, 32）的RGB三通道图像数据，数据类型是无符号8位整形（uint8）
Y_train和 Y_test是形如（nb_samples,）标签数据，标签的范围是0~9
```

CIFAR100  
**使用方法**

```
from keras.datasets import cifar100
(X_train, y_train), (X_test, y_test) = cifar100.load_data(label_mode='fine')
参数
label_mode：为‘fine’或‘coarse’之一，控制标签的精细度，‘fine’获得的标签是100个小类的标签，‘coarse’获得的标签是大类的标签
返回值
两个Tuple,(X_train, y_train), (X_test, y_test)，其中
X_train和X_test：是形如（nb_samples, 3, 32, 32）的RGB三通道图像数据，数据类型是无符号8位整形（uint8）
y_train和y_test：是形如（nb_samples,）标签数据，标签的范围是0~9

```











