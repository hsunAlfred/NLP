from math import pi
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn import metrics
import pandas as pd
import numpy as np
import jieba
import jieba.posseg as pseg
import paddle


class nlp_model_trainging:
    def __init__(self) -> None:
        jieba.set_dictionary('dict.txt.big')
        paddle.enable_static()
        jieba.enable_paddle()

        # load stop word
        with open("stopwords.txt", encoding='utf-8') as f:
            stopwords = f.read()
        self.custom_stopwords_list = [i for i in stopwords.split('\n')]

        self.avoid_word_kind = (
            'nr', 'nz', 'PER', 'f', 'ns', 'LOC', 's', 'nt', 'ORG', 't', 'nw', 'w', 'TIME')

    def __loadCorpusAndTransform(self, corpus, HMM=True, use_paddle=True):
        # load corpus(data)
        df = pd.read_csv(corpus, on_bad_lines='skip', encoding='utf-8')

        # Feature Engineering, let label be binary
        # set feature and label
        # X = df["comment"].apply(lambda x:  " ".join(
        #     jieba.posseg.cut(str(x), use_paddle=True)))

        X = self.__featureTransform(
            df["comment"], HMM=HMM, use_paddle=use_paddle)

        # y = df["star"].apply(lambda x: 1 if x > 3 else 0)
        y = df["star"]

        df = pd.DataFrame([X, y]).T
        df.to_excel(f'test_HMM_{HMM}_paddle_{use_paddle}.xlsx', index=False)

        print(X.shape)
        print(y.shape)

        # split dataset, random_state should only set in test
        return train_test_split(X, y, random_state=1)

    def __featureTransform(self, waitTransform, HMM=True, use_paddle=True):
        trans = []
        n = 1
        for i in waitTransform:
            print(f'\r{n}/{len(waitTransform)}', end='')
            n += 1
            pc = pseg.lcut(str(i), HMM=HMM, use_paddle=use_paddle)
            temp = []
            for j in pc:
                if tuple(j)[1] not in self.avoid_word_kind:
                    temp.append(tuple(j)[0])
            trans.append(' '.join(temp))
        return np.array(trans)

    def nlp_NB(self, corpus, HMM=True, use_paddle=True):
        X_train, X_test, y_train, y_test = self.__loadCorpusAndTransform(
            corpus, HMM=HMM, use_paddle=use_paddle)

        # # BoW transform
        # max_df = 0.8  # too high prob to appear
        # min_df = 3  # too low prob to appear

        # vect = CountVectorizer(max_df=max_df,
        #                        min_df=min_df,
        #                        token_pattern=u'(?u)\\b[^\\d\\W]\\w+\\b',
        #                        stop_words=frozenset(self.custom_stopwords_list))

        # # when develop, it helps us to makesure stop word work
        # # counts = pd.DataFrame(vect.fit_transform(X_train).toarray(),
        # #                       columns=vect.get_feature_names_out())

        # nb = MultinomialNB()
        # pipe = make_pipeline(vect, nb)

        # pipe.fit(X_train, y_train)

        # cv = cross_val_score(pipe, X_train, y_train,
        #                      cv=10, scoring='accuracy').mean()

        # y_pred = pipe.predict(X_test)

        # tempX = ('10點半點的餐，12點還沒下單，這速度也是醉了。',
        #          '味道還可以', '德國香腸沒有太好吃', '焗土豆和檸檬雞還是推薦的。')
        # tex = self.__featureTransform(tempX)
        # print(tex)
        # te = pipe.predict(tex)
        # print(te)

        # accuracy_score = metrics.accuracy_score(y_test, y_pred)

        # confusion_matrix = metrics.confusion_matrix(y_test, y_pred)

        # return pipe, cv, accuracy_score, confusion_matrix
        return 1, 2, 3, 4


if __name__ == "__main__":
    pipe, cv, accuracy_score, confusion_matrix = nlp_model_trainging().nlp_NB(
        corpus='comment_zh_tw.csv', HMM=True, use_paddle=True)
    pipe, cv, accuracy_score, confusion_matrix = nlp_model_trainging().nlp_NB(
        corpus='comment_zh_tw.csv', HMM=True, use_paddle=False)
    pipe, cv, accuracy_score, confusion_matrix = nlp_model_trainging().nlp_NB(
        corpus='comment_zh_tw.csv', HMM=False, use_paddle=True)
    pipe, cv, accuracy_score, confusion_matrix = nlp_model_trainging().nlp_NB(
        corpus='comment_zh_tw.csv', HMM=False, use_paddle=False)
    # print(pipe)
    # print(cv)
    # print(accuracy_score)
    # print(confusion_matrix)
