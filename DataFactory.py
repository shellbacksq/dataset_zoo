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
import sys
from collections import Counter
import re

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
# f=Get_Data(source_file_path="https://mirrors.tuna.tsinghua.edu.cn/tensorflow/releases.json")
# print(f)


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
    实现功能：
    1. 对中英文切词
    2. 计算词频
    3. 对词进行筛选
    4. 构建词典
    5. 加载已有数据和字典的功能
    """
    def __init__(self,filepath=""):
        pass

    @staticmethod
    def parse(line):
        subs = line.split(' ', 1)
        if 1 == len(subs):
            return subs[0], ''
        else:
            return subs[0], subs[1]

    @staticmethod
    def load(file_path):
        dids = list()
        docs = list()
        f = open(file_path, 'r')
        for line in f:
            line = line.decode('utf8')
            line = line.strip()
            if '' != line:
                did, doc = Process_Data_NLP.parse(line)
                dids.append(did)
                docs.append(doc)
        f.close()
        return dids, docs

    @staticmethod
    def split_cn_save(filepath_input,filepath_save):
        """
        保存切词后文件
        :param filepath_input:
        :param filepath_save:
        :return:
        """
        import jieba
        with open(filepath_save, "wb") as fw:
            with open(filepath_input,"rb") as fr:
                for line in fr.readlines():
                        fw.write(" ".join(jieba.cut(line)).encode("utf8"))

    @staticmethod
    def split_cn(filepath):
        """
        直接返回切完词后的句子
        :param filepath:
        :return:
        """
        import jieba
        with open(filepath,"rb") as f:
            for line in f.readlines():
                yield " ".join(jieba.cut(line))

    @staticmethod
    def read_cn_line(filepath):
        """
        直接返回句子
        :param filepath:
        :return:
        """
        return Process_Data_NLP.split_cn(filepath)

    @staticmethod
    def read_cn_word(filepath):
        """
        直接返回句子
        :param filepath:
        :return:
        """
        lines=Process_Data_NLP.split_cn(filepath)
        for line in lines:
            for word in line.split(" "):
                yield word

    @staticmethod
    def read_en_line(filepath):
        """
        直接返回句子
        :param filepath:
        :return:
        """
        with open(filepath) as f:
            for line in f.readlines():
                yield line

    @staticmethod
    def read_en_word(filepath):
        """
        返回句子中的词
        :param filepath:
        :return:
        """
        with open(filepath) as f:
            for line in f.readlines():
                for word in line.split():
                    yield word

    @staticmethod
    def _word_freq(docs):
        wdf = dict()
        for ws in docs.split("\r\n"):
            # ws = set(ws)
            for w in re.split("[\s]+",ws):
                wdf[w] = wdf.get(w, 0) + 1
        return wdf

    @staticmethod
    def _word_filter(docs,
                     exist_words_useless=True,
                    stop_list=list(),
                    min_freq=1,
                    max_freq=sys.maxsize):
        from string import punctuation
        add_punc = '，、【】“”：；（）《》‘’{}？！⑦()、%^>℃：.”“^-——=擅长于的&#@￥。．'
        all_punc = punctuation + add_punc
        stop_words = [item for item in all_punc] + [""]
        stop_words.extend(stop_list)
        if exist_words_useless:
            words_useless = set()
            # # filter with stop_words
            words_useless.update(stop_words)
            # # filter with min_freq and max_freq
            wdf = Process_Data_NLP._word_freq(docs)
            for w in wdf:
                if min_freq > wdf[w] or max_freq < wdf[w]:
                    words_useless.add(w)
        # filter with useless words
        docs = [[w for w in re.split("[\s]+",ws) if w not in stop_words] for ws in docs.split("\r\n") if len(ws)>0]
        return docs, words_useless

    @staticmethod
    def word_freq(filepath):
        with open(filepath,"rb") as f:
            return Process_Data_NLP._word_freq(f.read().decode())

    @staticmethod
    def word_filter(filepath,words_useless=None,stop_list=[],min_freq=1,max_freq=sys.maxsize):
        with open(filepath,"rb") as f:
            return Process_Data_NLP._word_filter(f.read().decode(),stop_list=stop_list)

    def build_vocab(self,word_data,vocab_size=None):
        """ 建立词汇表，TODO：最常见的N个词 """
        # word_data = list(itertools.chain.from_iterable(w))
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
    def save(self,data,outpath):
        with open(outpath,"wb") as f:
            for line in data:
                f.write(" ".join(line).encode("utf8")+"\r\n".encode("utf8"))

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




if __name__=="__main__":
    pnlp=Process_Data_NLP()
    data,_=pnlp.word_filter("F:\data\红楼梦seg.txt",stop_list=["了"])
    # # pnlp.save(data,"红楼梦filter.txt")
    import itertools
    doc=list(itertools.chain.from_iterable(data))
    vocab_dict,_=pnlp.build_vocab(doc)
    with open("红楼梦index.txt","wb") as f:
        for line in data:
            data_index=pnlp.convert_words_to_index(line,vocab_dict)
            # print(data_index)
            # c,t=pnlp.generate_sample(data_index,5)
            # print(c)
    #         print(data_index)
            f.write(" ".join([str(index) for index in data_index]).encode("utf8")+"\r\n".encode("utf8"))
    # with open("红楼梦index.txt", "rb") as f:
