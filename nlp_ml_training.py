from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
# from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import pandas as pd
from joblib import dump
from nlp_frame import nlp_frame


class nlp_model_training(nlp_frame):
    def __init__(self) -> None:
        super().__init__()

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
        X = self.__bowTransform(df["X"]).toarray()
        # transform to dataframe
        # X = pd.DataFrame(
        #     X, columns=self.vect.get_feature_names_out())
        y = df['y'].astype('category')

        print(X.shape)
        print(y.shape)

        # split dataset, random_state should only set in test
        return train_test_split(X, y, train_size=0.8)

    def __bowTransform(self, source):  # -----------------------------------
        return self.vect.fit_transform(source)

    def __w2vTransform(self, source):
        pass

    def nlp_NB(self, corpus: str, HMM: bool, use_paddle: bool):
        X_train, X_test, y_train, y_test = self.__loadCorpusAndTransform(
            corpus, HMM=HMM, use_paddle=use_paddle)

        nb = GaussianNB()

        nb.fit(X_train, y_train)

        cv = cross_val_score(nb, X_train, y_train,
                             cv=5, scoring='accuracy').mean()

        y_pred = nb.predict(X_test)

        accuracy_score = metrics.accuracy_score(y_test, y_pred)

        confusion_matrix = metrics.confusion_matrix(y_test, y_pred)

        return nb, cv, accuracy_score, confusion_matrix

    def nlp_RF(self, corpus: str, HMM: bool, use_paddle: bool):
        X_train, X_test, y_train, y_test = self.__loadCorpusAndTransform(
            corpus, HMM=HMM, use_paddle=use_paddle)

        rf = RandomForestClassifier(max_depth=2, random_state=0)
        rf.fit(X_train, y_train)

        cv = cross_val_score(rf, X_train, y_train,
                             cv=5, scoring='accuracy').mean()

        y_pred = rf.predict(X_test)

        accuracy_score = metrics.accuracy_score(y_test, y_pred)

        confusion_matrix = metrics.confusion_matrix(y_test, y_pred)

        return rf, cv, accuracy_score, confusion_matrix


def nb_call():
    res = ''
    for h, u in [(True, True), (True, False), (False, True), (False, False)]:
        nmt = nlp_model_training()
        nb, cv, accuracy_score, confusion_matrix = nmt.nlp_NB(
            corpus='comment_zh_tw.csv', HMM=h, use_paddle=u)

        dump(nb, f'nlpModel_NB/nlp_NB_HMM_{h}_paddle_{u}.joblib')
        dump(nmt.vect, f'nlpModel_NB/nlp_vect_HMM_{h}_paddle_{u}.vect')

        res += f'\n\nHMM_{h}_paddle_{u}'
        res += (
            f'\ncross value:{cv:.3f}\naccuracy score:{accuracy_score:.3f}\nconfusion matrix\n{confusion_matrix}')

    print(res)
    with open('nlp_NB_score.txt', 'w') as f:
        f.write(res)


def rf_call():
    res = ''
    for h, u in [(True, True), (True, False), (False, True), (False, False)]:
        nmt = nlp_model_training()
        rf, cv, accuracy_score, confusion_matrix = nmt.nlp_RF(
            corpus='comment_zh_tw.csv', HMM=h, use_paddle=u)

        dump(rf, f'nlpModel_RF/nlp_RF_HMM_{h}_paddle_{u}.joblib')
        dump(nmt.vect, f'nlpModel_RF/nlp_vect_HMM_{h}_paddle_{u}.vect')

        res += f'\n\nHMM_{h}_paddle_{u}'
        res += (
            f'\ncross value:{cv:.3f}\naccuracy score:{accuracy_score:.3f}\nconfusion matrix\n{confusion_matrix}')

    print(res)
    with open('nlp_RF_score.txt', 'w') as f:
        f.write(res)


if __name__ == "__main__":
    nb_call()

    # rf_call()
