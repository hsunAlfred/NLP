from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn import metrics
import pandas as pd
import jieba


class nlp_model_trainging:
    def __init__(self) -> None:
        # load stop word
        with open("stopwords.txt", encoding='utf-8') as f:
            stopwords = f.read()
        self.custom_stopwords_list = [i for i in stopwords.split('\n')]

    def __loadCorpusAndTransform(self, corpus='comment_zh_tw.csv'):
        # load corpus(data)
        df = pd.read_csv(corpus, on_bad_lines='skip', encoding='utf-8')

        # Feature Engineering, let label be binary
        # set feature and label
        X = df["comment"].apply(lambda x:  " ".join(jieba.cut(str(x))))
        y = df["star"].apply(lambda x: 1 if x > 3 else 0)

        # split dataset, random_state should only set in test
        return train_test_split(X, y, random_state=1)

    def nlp_NB(self):
        X_train, X_test, y_train, y_test = self.__loadCorpusAndTransform()

        # BoW transform
        max_df = 0.8  # too high prob to appear
        min_df = 3  # too low prob to appear

        vect = CountVectorizer(max_df=max_df,
                               min_df=min_df,
                               token_pattern=u'(?u)\\b[^\\d\\W]\\w+\\b',
                               stop_words=frozenset(self.custom_stopwords_list))

        # when develop, it helps us to makesure stop word work
        # counts = pd.DataFrame(vect.fit_transform(X_train).toarray(),
        #                       columns=vect.get_feature_names_out())

        nb = MultinomialNB()
        pipe = make_pipeline(vect, nb)

        pipe.fit(X_train, y_train)

        cv = cross_val_score(pipe, X_train, y_train,
                             cv=10, scoring='accuracy').mean()

        y_pred = pipe.predict(X_test)
        print(pipe.predict(
            ['10點半點的餐，12點還沒下單，這速度也是醉了。味道還可以，德國香腸沒有太好吃，焗土豆和檸檬雞還是推薦的。']))

        accuracy_score = metrics.accuracy_score(y_test, y_pred)

        confusion_matrix = metrics.confusion_matrix(y_test, y_pred)

        return pipe, cv, accuracy_score, confusion_matrix


if __name__ == "__main__":
    nlp_model_trainging().nlp_NB()
