import jieba
import yaml
import os
import numpy as np
import pandas as pd
import multiprocessing
from gensim.models.word2vec import Word2Vec
from gensim.corpora.dictionary import Dictionary
from sklearn.model_selection import train_test_split

import keras
from keras.models import Sequential
from keras.preprocessing import sequence
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Dropout, Activation


# 设置常量
basic_path = os.getcwd().replace("LSTM", "")
data_path = basic_path + 'Data/简体30954条_3分类.csv'
w2v_path = basic_path + 'Data/WordEmbedding/Word60.model'
vocab_dim = 60
n_exposures = 1
window_size = 7
cpu_count = multiprocessing.cpu_count()
n_iterations = 5
input_length = 100
batch_size = 32
n_epoch = 3


# 读取文件
def get_file(data):
    file = pd.read_csv(data)
    label = file.iloc[:, 0]
    content = file.iloc[:, 1]
    return content, label


# 分词
def segment(text):
    return [jieba.lcut(document.replace('\n', '')) for document in text]


# 创建词语字典，并返回每个词语的索引，词向量，以及每个句子所对应的词语索引
def create_dictionaries(model=None, combined=None):
    if (combined is not None) and (model is not None):
        gensim_dict = Dictionary()
        gensim_dict.doc2bow(model.wv.vocab.keys(), allow_update=True)

        w2indx = {v: k + 1 for k, v in gensim_dict.items()}       # 词语的索引,(k->v)=>(v->k)
        w2vec = {word: model[word] for word in w2indx.keys()}   # 词语的词向量, (word->model(word))

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
        return w2indx, w2vec, n_combined
    else:
        print('No data provided...')


def load_wordvector(w2v_path):
    # 载入模型
    model = Word2Vec.load(w2v_path)
    return model


def get_data(index_dict, word_vectors, combined, y):
    n_symbols = len(index_dict) + 1  # 所有单词的索引数
    embedding_weights = np.zeros((n_symbols, vocab_dim))  # 初始化 索引为0的词语，词向量全为0
    for word, index in index_dict.items():  # 对每个词语对应其词向量
        if word in word_vectors:
            embedding_weights[index, :] = word_vectors[word]
    x_train, x_test, y_train, y_test = train_test_split(combined, y, test_size=0.2)
    y_train = keras.utils.to_categorical(y_train, num_classes=3)
    y_test = keras.utils.to_categorical(y_test, num_classes=3)
    return n_symbols, embedding_weights, x_train, y_train, x_test, y_test


# 定义网络结构
def train_lstm(n_symbols, embedding_weights, x_train, y_train, x_test, y_test):
    print('Defining a Simple Keras Model...')
    model = Sequential()  # or Graph or whatever
    model.add(Embedding(output_dim=vocab_dim,
                        input_dim=n_symbols,
                        mask_zero=True,
                        weights=[embedding_weights],
                        input_length=input_length))  # Adding Input Length
    model.add(LSTM(output_dim=30, activation='tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='softmax'))  # Dense=>全连接层,输出维度=3
    model.add(Activation('softmax'))

    print('Compiling the Model...')
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    print("Train...")
    model.fit(x_train, y_train, batch_size=batch_size, epochs=n_epoch, verbose=1)

    print("Evaluate...")
    score = model.evaluate(x_test, y_test, batch_size=batch_size)

    yaml_string = model.to_yaml()
    outfile = open(basic_path + 'LSTM/lstm.yml', 'w')
    outfile.write(yaml.dump(yaml_string, default_flow_style=True))
    model.save_weights(basic_path + 'LSTM/lstm.h5')
    print('Test score:', score)


# 训练模型，并保存
print('Loading Data...')
text, y = get_file(data_path)
print(len(text), len(y))
print('Tokenizing & Loading a Word2vec model...')
index_dict, word_vectors, combined = create_dictionaries(load_wordvector(w2v_path), segment(text))
print('Setting up Arrays for Keras Embedding Layer...')
n_symbols, embedding_weights, x_train, y_train, x_test, y_test = get_data(index_dict, word_vectors, combined, y)
print("x_train.shape and y_train.shape:")
print(x_train.shape, y_train.shape)
train_lstm(n_symbols, embedding_weights, x_train, y_train, x_test, y_test)
