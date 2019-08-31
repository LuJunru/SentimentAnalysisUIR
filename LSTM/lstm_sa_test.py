import yaml
import os
import jieba
import numpy as np
from gensim.models.word2vec import Word2Vec
from gensim.corpora.dictionary import Dictionary
from keras.preprocessing import sequence
from keras.models import model_from_yaml

import sys
sys.setrecursionlimit(1000000)

# define parameters
np.random.seed(1337)  # For Reproducibility
input_length = 100
basic_path = os.getcwd().replace("/LSTM", "")
w2v_path = basic_path + '/Data/WordEmbedding/Word60.model'


# 创建词语字典，并返回每个词语的索引，词向量，以及每个句子所对应的词语索引
def create_dictionaries(w2indx=None, combined=None):
    if (combined is not None) and (w2indx is not None):
        def parse_dataset(combined):
            """
            :intro: Words become integers
            :param combined:
            :return:
            """
            data = []
            for sentence in combined:
                new_txt = []
                for word in sentence:
                    try:
                        new_txt.append(w2indx[word])
                    except:
                        new_txt.append(0)  # freq < 10->0
                data.append(new_txt)
            return data
        n_combined = parse_dataset(combined)
        n_combined = sequence.pad_sequences(n_combined, maxlen=input_length)  # 每个句子所含词语对应的索引，所以句子中含有频数小于10的词语，索引为0
        return n_combined
    else:
        print('No data provided...')


def input_transform(string):
    words = jieba.lcut(string)
    words = np.array(words).reshape(1, -1)
    model = Word2Vec.load(w2v_path)
    gensim_dict = Dictionary()
    gensim_dict.doc2bow(model.wv.vocab.keys(), allow_update=True)
    w2indx = {v: k + 1 for k, v in gensim_dict.items()}  # 词语的索引,(k->v)=>(v->k)
    combined = create_dictionaries(w2indx, words)
    return combined


def lstm_predict(model, strings):
    data = input_transform(strings[0]).reshape(1, -1)
    for string in strings[1:]:
        data = np.vstack((data, input_transform(string).reshape(1, -1)))
    result = model.predict(data)
    return np.column_stack((result, np.array(strings).reshape(-1, 1)))


if __name__ == "__main__":
    print('loading model......')
    with open(basic_path + '/LSTM/lstm.yml', 'r') as f:
        yaml_string = yaml.load(f)
    model = model_from_yaml(yaml_string)

    print('loading weights......')
    model.load_weights(basic_path + '/LSTM/lstm.h5')
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    strings = ['酒店的环境非常好，价格也便宜，值得推荐',
               '手机质量太差了，傻逼店家，赚黑心钱，以后再也不会买了',
               "这是我看过文字写得很糟糕的书，因为买了，还是耐着性子看完了，但是总体来说不好，文字、内容、结构都不好",
               "虽说是职场指导书，但是写的有点干涩，我读一半就看不下去了！",
               "书的质量还好，但是内容实在没意思。本以为会侧重心理方面的分析，但实际上是婚外恋内容。",
               "不是太好",
               "不错不错",
               "真的一般，没什么可以学习的"]
    print(lstm_predict(model, strings))
