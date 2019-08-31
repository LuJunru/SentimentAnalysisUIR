import yaml
import os
import pandas as pd
from keras.models import model_from_yaml
from LSTM.lstm_sa_test import lstm_predict

basic_path = os.getcwd()
print('loading model......')
with open(basic_path + '/LSTM/lstm.yml', 'r') as f:
    yaml_string = yaml.load(f)
model = model_from_yaml(yaml_string)
print('loading weights......')
model.load_weights(basic_path + '/LSTM/lstm.h5')
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


def Lstm(strings):
    return lstm_predict(model, strings)


class SentimentAnalysis:
    def __init__(self, articles):
        self.articles = articles
        self.weights = [1.0, 0.6, 0.5, 0.4, 0.3]

    def evaluate(self, article):
        evaluations = {}
        paragraphs = article.split("\n\n\n")
        for i, paragraph in enumerate(paragraphs):
            if i == 0:
                evaluations[paragraph] = self.weights[0]
            else:
                sentences = paragraph.split("\n")
                for j, sentence in enumerate(sentences):
                    if i == 1:
                        if j == 0:
                            evaluations[sentence] = self.weights[1]
                        else:
                            evaluations[sentence] = self.weights[2]
                    else:
                        if j == 0:
                            evaluations[sentence] = self.weights[3]
                        else:
                            evaluations[sentence] = self.weights[4]
        scores = pd.DataFrame(lstm_predict(model, list(evaluations.keys())))
        print(scores)
        scores["weights"] = scores.iloc[:, 3].apply(lambda x: evaluations[x])
        scores["Neural"] = scores.iloc[:, 0].astype('float') * scores["weights"]
        scores["Positive"] = scores.iloc[:, 1].astype('float') * scores["weights"]
        scores["Negtive"] = scores.iloc[:, 2].astype('float') * scores["weights"]
        return scores[["Neural", "Positive", "Negtive"]].mean(axis=0)

    def score(self, ouput_file):
        w = open(ouput_file, "w")
        for article in self.articles:
            results = self.evaluate(article)
            score = results[results == results.max()]
            w.write(str(score.index[0]) + "\n")
        w.close()


if __name__ == "__main__":
    articles = open(basic_path + "/Data/Test.txt", "r").read().split("\n\n\n\n")
    sentiment = SentimentAnalysis(articles)
    sentiment.score(basic_path + "/Data/Test_classification.txt")
