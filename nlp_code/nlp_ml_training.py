from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
# from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import pandas as pd
from joblib import dump
from nlp_frame import nlp_frame
# from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer


class nlp_model_training(nlp_frame):
    def __init__(self) -> None:
        super().__init__()
        self.vect = HashingVectorizer(n_features=2**10)

    def __loadCorpusAndTransform(self, corpus: str, HMM: bool, use_paddle: bool):
        # load corpus(data)
        df = pd.read_csv(corpus, on_bad_lines='skip', encoding='utf-8')

        # Feature Engineering(feature to seg, label to category)

        X = self.seg(
            df["comment"], HMM=HMM, use_paddle=use_paddle)

        y = df["star"]

        df = pd.DataFrame([X, y], index=["X", 'y']).T.dropna()

        # BoW transform
        # -----------------------------------
        X = self.vect.fit_transform(df["X"]).toarray()
        # transform to dataframe
        # X = pd.DataFrame(
        #     X, columns=self.vect.get_feature_names_out())
        y = df['y'].astype('category')

        print(X.shape)
        print(y.shape)

        # split dataset, random_state should only set in test
        return train_test_split(X, y, train_size=0.8)

    def training(self, corpus: str, model: str, HMM: bool, use_paddle: bool):
        X_train, X_test, y_train, y_test = self.__loadCorpusAndTransform(
            corpus, HMM=HMM, use_paddle=use_paddle)

        m = None
        if model == "NB":
            m = GaussianNB()
        elif model == "RF":
            m = RandomForestClassifier(max_depth=2, random_state=0)

        m.fit(X_train, y_train)

        cv = cross_val_score(m, X_train, y_train,
                             cv=5, scoring='accuracy').mean()

        y_pred = m.predict(X_test)

        accuracy_score = metrics.accuracy_score(y_test, y_pred)

        confusion_matrix = metrics.confusion_matrix(y_test, y_pred)

        return m, cv, accuracy_score, confusion_matrix


def nb_call():
    res = ''
    params = {
        "corpus": 'comment_zh_tw.csv',
        "model": "NB",
    }
    for h, u in [(True, True), (True, False), (False, True), (False, False)]:
        nmt = nlp_model_training()

        params["HMM"] = h
        params["use_paddle"] = u
        nb, cv, accuracy_score, confusion_matrix = nmt.training(**params)

        dump(
            nb, f'nlpModel_{params["model"]}/nlp_{params["model"]}_HMM_{h}_paddle_{u}.joblib')
        dump(
            nmt.vect, f'nlpModel_{params["model"]}/nlp_vect_HMM_{h}_paddle_{u}.vect')

        res += f'\n\nHMM_{h}_paddle_{u}'
        res += (
            f'\ncross value:{cv:.3f}\naccuracy score:{accuracy_score:.3f}\nconfusion matrix\n{confusion_matrix}')

    print(res)
    with open('nlp_NB_score.txt', 'w') as f:
        f.write(res)


if __name__ == "__main__":
    nb_call()

    # rf_call()
