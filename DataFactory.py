#encoding:utf8
#TODO:
#1. 自动判别数据文件格式，压缩文件就先解压，区别文本文件和图片文件
#2. 对于文本文件，如果是中文先切词，保存切词信息，再统计出基本信息包括：去掉常用词后的词频
#3. 序列化输出。

#模块一：获取数据 class Get_Data()
#模块二：整理数据 class Beat_Data()
#模块三：处理数据 class Process_Data()
#模块四：可视化数据 class Show_Data()

import os
import urllib
from six.moves import urllib
import numpy as np
import random


class Get_Data():
    """
    1.来源于远程主机
    2.来源于本地
    3.来源于其他
    4.保存到本地
    TODO:
    1.远程数据默认保存到本地，默认名字
    2.本地数据更改数据位置，复制或者剪切
    """
    def __init__(self,source_file_path,folder=os.getcwd(),filename=""):
        self.url=""
        self.filepath=""
        self.other=""
        self.source_file_path=source_file_path
        self.folder=folder
        self.filename=filename


    def _file_source_type(self):
        """判断数据来源"""
        if self.source_file_path.startswith("http") or \
                "^*.*.*.*$" in self.source_file_path:
            self.url=self.source_file_path
            return "url"
        self.filepath=self.source_file_path
        self.filename=self.filepath.split("/")[-1]
        return "local"

    def _download_file(self):
        """下载来自远程主机的数据"""
        if self.filename:
            self.filepath = self.folder+self.filename
        else:
            self.filepath=self.folder+"/"+self.url.split("/")[-1]
        if os.path.exists(self.filepath):
            print("Dataset ready")
        file_name, _ = urllib.request.urlretrieve(self.url, self.filepath)

    def file_path(self):
        if self._file_source_type()=="url":
            self._download_file()
        return self.filepath

    def data_info(self):
        pass

    def __repr__(self):
        self.file_path()
        return self.filepath
#
f=Get_Data(source_file_path="https://mirrors.tuna.tsinghua.edu.cn/tensorflow/releases.json")
print(f)


class Beat_Data():
    """
    1.如果是压缩文件先解压
    """
    def __init__(self,filepath):
        self.filepath=filepath

    def _extract(self,suf,filepath):
        import gzip,tarfile,zipfile,rarfile
        filename = filepath.split("/")[-1]
        if suf==".gz":
            f_name=filename.replace(".gz","")
            g_file=gzip.GzipFile(filename)
            open(f_name,"w+").write(g_file.read())
            g_file.close()
        if suf==".zip":
            zip_file = zipfile.ZipFile(filename)
            if os.path.isdir(filename + "_files"):
                pass
            else:
                os.mkdir(filename + "_files")
            for names in zip_file.namelist():
                zip_file.extract(names, filename + "_files/")
            zip_file.close()
        if suf==".tar":
            tar = tarfile.open(filename)
            names = tar.getnames()
            if os.path.isdir(filename + "_files"):
                pass
            else:
                os.mkdir(filename + "_files")
                # 由于解压后是许多文件，预先建立同名文件夹
            for name in names:
                tar.extract(name, filename + "_files/")
            tar.close()
        if suf==".zip":
            zip_file = zipfile.ZipFile(filename)
            if os.path.isdir(filename + "_files"):
                pass
            else:
                os.mkdir(filename + "_files")
            for names in zip_file.namelist():
                zip_file.extract(names, filename + "_files/")
            zip_file.close()
        if suf==".rar":
            """unrar zip file"""
            rar = rarfile.RarFile(filename)
            if os.path.isdir(filename + "_files"):
                pass
            else:
                os.mkdir(filename + "_files")
            os.chdir(filename + "_files")
            rar.extractall()
            rar.close()


    def _extractfiles(self):
        """解压文件"""
        suffixes=[".tar",".gz",".tar.gz",".bz2",".tar.bz2",".rar",\
                ".zip"]
        filename = self.filepath.split("/")[-1]
        for suf in suffixes:
            if suf in filename:
                self._extract(suf,self.filepath)
                return filename

    def _filetype(self,filepath):
        pass

    def fileinfo(self):
        pass



class Process_Data_NLP():
    """
    1.如果是中文先切词。
    """
    def __init__(self,filepath):
        self.filepath=filepath
    def split_words(self):
        pass
    def read_data(self,filepath):
        with open(filepath) as f:
            for line in f.readlines():
                for word in line.split():
                    yield word

    def build_vocab(self,word_data,vocab_size=None):
        """ 建立词汇表，TODO：最常见的N个词 """
        from collections import Counter
        import copy
        c=Counter()
        word_to_index = {}
        for word in word_data:
            c.update(word)
            if word not in word_to_index:
                index = len(word_to_index)
                word_to_index[word]=index
        if vocab_size:
            word_to_index_tmp=copy.deepcopy(word_to_index)
            for word in word_to_index_tmp.keys():
                if word not in c.most_common(vocab_size):
                    word_to_index.pop(word)
            del word_to_index_tmp
        index_to_word = dict(zip(word_to_index.values(), word_to_index.keys()))
        return word_to_index, index_to_word

    def convert_words_to_index(self,words, dictionary):
        """ Replace each word in the dataset with its index in the dictionary """
        return [dictionary[word] if word in dictionary else 0 for word in words]

    def convert_index_to_words(self,indexs, dictionary):
        return [dictionary[index] if index in dictionary else "null" for index in indexs.tolist()]

    def generate_sample(self,index_words, context_window_size):
        """ Form training pairs according to the skip-gram model. """
        for index, center in enumerate(index_words):
            context = random.randint(1, context_window_size)
            # get a random target before the center word
            for target in index_words[max(0, index - context): index]:
                yield center, target
            # get a random target after the center wrod
            for target in index_words[index + 1: index +context_window_size-context + 1]:
                yield center, target

    def get_batch(self,iterator, batch_size):
        """ Group a numerical stream into batches and yield them as Numpy arrays. """
        while True:
            center_batch = np.zeros(batch_size,dtype=np.int32)
            target_batch = np.zeros(batch_size,dtype=np.int32)
            for index in range(batch_size):
                center_batch[index], target_batch[index] = next(iterator)
            yield center_batch, target_batch


class Process_Data_IP():
    pass
class Show_Data():
    pass






def test():
    file_path = "/home/sq/learnhub/course/cs224d/cs224d/assignment2/data/ptb/ptb.train.txt"
    skip_window=5
    batch_size=10
    words = read_data(file_path)
    word_dictionary, index_dictionary = build_vocab(words)
    words = read_data(file_path)
    index_words = convert_words_to_index(words, word_dictionary)
    del words  # to save memory
    single_gen = generate_sample(index_words, skip_window)

    center_word,target_word=next(get_batch(single_gen, batch_size))
    # print(type(center_word[0]),target_word)
    print(convert_index_to_words(center_word,index_dictionary),"\n",convert_index_to_words(target_word,index_dictionary))

if __name__=="__main__":
    test()
