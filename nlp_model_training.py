from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn import metrics
import pandas as pd
from joblib import dump
from nlp_model_frame import nlp_model_frame


class nlp_model_trainging(nlp_model_frame):
    def __init__(self) -> None:
        super().__init__()

    def loadCorpusAndTransform(self, corpus, HMM=True, use_paddle=True):
        # load corpus(data)
        df = pd.read_csv(corpus, on_bad_lines='skip', encoding='utf-8')

        # Feature Engineering, let label be binary
        # set feature and label
        # X = df["comment"].apply(lambda x:  " ".join(
        #     jieba.posseg.cut(str(x), use_paddle=True)))

        X = self.featureTransform(
            df["comment"], HMM=HMM, use_paddle=use_paddle)

        # y = df["star"].apply(lambda x: 1 if x > 3 else 0)
        y = df["star"]

        df = pd.DataFrame([X, y], index=["X", 'y']).T.dropna()

        X = df["X"]
        y = df['y'].astype('category')

        print(X.shape)
        print(y.shape)

        # split dataset, random_state should only set in test
        return train_test_split(X, y)

    def nlp_NB(self, corpus, HMM=True, use_paddle=True):
        X_train, X_test, y_train, y_test = self.loadCorpusAndTransform(
            corpus, HMM=HMM, use_paddle=use_paddle)

        # BoW transform
        max_df = 0.8  # too high prob to appear
        min_df = 3  # too low prob to appear

        vect = CountVectorizer(max_df=max_df,
                               min_df=min_df,
                               token_pattern=u'(?u)\\b[^\\d\\W]\\w+\\b',
                               stop_words=frozenset(self.custom_stopwords_list))

        # counts = pd.DataFrame(vect.fit_transform(X_train).toarray(),
        #                       columns=vect.get_feature_names_out())

        nb = MultinomialNB()
        pipe = make_pipeline(vect, nb)

        pipe.fit(X_train, y_train)

        cv = cross_val_score(pipe, X_train, y_train,
                             cv=10, scoring='accuracy').mean()

        y_pred = pipe.predict(X_test)

        accuracy_score = metrics.accuracy_score(y_test, y_pred)

        confusion_matrix = metrics.confusion_matrix(y_test, y_pred)

        return pipe, cv, accuracy_score, confusion_matrix


if __name__ == "__main__":
    nmt = nlp_model_trainging()
    res = ''
    for h, u in [(True, True), (True, False), (False, True), (False, False)]:
        pipe, cv, accuracy_score, confusion_matrix = nmt.nlp_NB(
            corpus='comment_zh_tw.csv', HMM=h, use_paddle=u)

        dump(pipe, f'./selfModel/nlp_NB_HMM_{h}_paddle_{u}.joblib')

        res += f'\n\nHMM_{h}_paddle_{u}'
        res += (
            f'\ncross value:{cv:.3f}\naccuracy score:{accuracy_score:.3f}\nconfusion matrix\n{confusion_matrix}')

    print(res)
    with open('nlp_NB_score.txt', 'w') as f:
        f.write(res)
