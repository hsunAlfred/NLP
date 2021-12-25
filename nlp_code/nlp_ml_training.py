from numpy import uint
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
# from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
#import pandas as pd
from joblib import dump
from nlp_frame import nlp_frame
from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.feature_extraction.text import HashingVectorizer
import time


class nlp_model_training(nlp_frame):
    def __init__(self, vectParams: dict, segParams: dict, modelSelect: str, modelParams: dict) -> None:
        '''"vectParams":dict
        {
            "analyzer":str "word" | "char" | "char_wb",
            "max_df":float [0.0, 1.0],
            "min_df":float [0.0, 1.0],
            "binary":bool
        }

        "segParams":dict
        {
            "corpus":str,
            "HMM":bool,
            "use_paddle":bool
        }

        "modelSelect":str "NB" | "RF",
        "modelParams":dict
        {
            NB use, no Fool Proof, makesure what modelSelect you set
            "alpha":float, default=1.0. Additive (Laplace/Lidstone) smoothing parameter (0 for no smoothing).
            "fit_prior":bool, default=True,Whether to learn class prior probabilities or not. If false, a uniform prior will be used.

            RF use, no Fool Proof, makesure what modelSelect you set
            "n_estimators":int, default=100
            "criterion":str "gini" | "entropy" default=”gini”
            "min_samples_split":int or float, default=2
            "min_samples_leaf"int or float, default=1
            "max_features":str "auto", "sqrt", "log2"
            "bootstrap":bool, default=True
            "oob_scorebool": bool, default=False, Only available if bootstrap=True.
            "class_weight":default=None, {“balanced”, “balanced_subsample”} [{0: 1, 1: 1}, {0: 1, 1: 5}, {0: 1, 1: 1}, {0: 1, 1: 1}]
        }
        '''
        super().__init__()

        self.vect = CountVectorizer(**vectParams)

        self.segParams = segParams
        self.modelSelect = modelSelect
        self.modelParams = modelParams

        # self.vect = HashingVectorizer(n_features=2**n)

    def __loadCorpusAndSplit(self, corpus: str, HMM: bool, use_paddle: bool):
        df = self.loadCorpus(corpus, HMM, use_paddle)

        # BoW transform
        # -----------------------------------
        X = self.vect.fit_transform(df["X"]).toarray()
        # transform to dataframe
        # X = pd.DataFrame(
        #     X, columns=self.vect.get_feature_names_out())
        y = df['y'].astype('category')

        print(f"\n{X.shape}\n{y.shape}")

        # split dataset, random_state should only set in test
        X_v, X_test, y_v, y_test = \
            train_test_split(X, y, train_size=0.8, stratify=y)

        X_train, X_vaild, y_train, y_vaild = \
            train_test_split(X_v, y_v,
                             train_size=0.75, stratify=y_v)

        return X_train, X_test, y_train, y_test, X_vaild, y_vaild

    def training(self):
        X_train, X_test, y_train, y_test, X_vaild, y_vaild = \
            self.__loadCorpusAndSplit(**self.segParams)

        print(f'train dataset shape:{X_train.shape} {y_train.shape}')
        print(f'vaild dataset shape:{X_vaild.shape} {y_vaild.shape}')
        print(f'test  dataset shape:{X_test.shape} {y_test.shape}')

        model = None
        if self.modelSelect == "NB":
            # model = GaussianNB()
            model = MultinomialNB(**self.modelParams)
        elif self.modelSelect == "RF":
            model = RandomForestClassifier(**self.modelParams)

        model.fit(X_train, y_train)

        # cv = cross_val_score(model, X_train, y_train,
        #                      cv=5, scoring='accuracy').mean()
        # ----------------------------------------------------------------
        y_train_pred = model.predict(X_train)
        confusion_matrix_train = metrics.confusion_matrix(
            y_train, y_train_pred)
        accuracy_score_train = metrics.accuracy_score(y_train, y_train_pred)

        y_vaild_pred = model.predict(X_vaild)
        confusion_matrix_vaild = metrics.confusion_matrix(
            y_vaild, y_vaild_pred)
        accuracy_score_vaild = metrics.accuracy_score(y_vaild, y_vaild_pred)

        y_pred = model.predict(X_test)
        accuracy_score_test = metrics.accuracy_score(y_test, y_pred)
        confusion_matrix_test = metrics.confusion_matrix(y_test, y_pred)

        return (model, accuracy_score_train, confusion_matrix_train,
                accuracy_score_vaild, confusion_matrix_vaild, accuracy_score_test, confusion_matrix_test)


def ml_call(vectParams, segParams, modelSelect, modelParams, resultTimestamp):
    nmt = nlp_model_training(vectParams, segParams, modelSelect, modelParams)
    model, accuracy_score_train, confusion_matrix_train, accuracy_score_vaild, confusion_matrix_vaild,\
        accuracy_score_test, confusion_matrix_test = nmt.training()

    dump(
        model, f'nlpModel_{modelSelect}/{resultTimestamp}.joblib')
    dump(
        nmt.vect, f'nlpModel_{modelSelect}/vect_{resultTimestamp}.vect')

    temp = f"\n{resultTimestamp}\n{vectParams}\n{segParams}\n{modelSelect}\n{modelParams}\n" +\
        "-------------------------------------------------------------\n"
    with open(f"./info/{modelSelect}_parameters.txt", mode="a", encoding="utf-8") as f:
        f.write(temp)
    print(temp)

    temp = f"\n{resultTimestamp}\n" +\
        f"accuracy score train:{accuracy_score_train:.3f}\nconfusion matrix train\n{confusion_matrix_train}\n" +\
        f"accuracy score vaild:{accuracy_score_vaild:.3f}\nconfusion matrix vaild\n{confusion_matrix_vaild}\n" +\
        f"accuracy score test:{accuracy_score_test:.3f}\nconfusion matrix test\n{confusion_matrix_test}\n" +\
        "-------------------------------------------------------------\n"

    with open(f"./info/{modelSelect}_modelScore.txt", 'a', encoding="utf-8") as f:
        f.write(temp)
    print(temp)


if __name__ == "__main__":
    # max_df min_df -> float in range [0.0, 1.0]
    start = time.time()
    nnNB = 1
    for corp in ("./corpus_words/corpus_new.xlsx", "./corpus_words/corpus.xlsx"):
        for ste in range(1, 11):
            for ma in range(ste, 11):
                for mi in range(ma-ste, ma+1):
                    for h, u in ((True, True), (True, False), (False, True), (False, False)):
                        for b in (True, False):
                            for fp in (True, False):
                                for a in ("word", "char", "char_wb"):
                                    try:
                                        print(
                                            f'***************iteration:{nnNB}***************')
                                        resultTimestamp = f"{time.time()}"

                                        vectParams = {
                                            "analyzer": a,
                                            "max_df": ma/10,
                                            "min_df": mi/10,
                                            "binary": b
                                        }

                                        segParams = {
                                            "corpus": corp,
                                            "HMM": h,
                                            "use_paddle": u
                                        }

                                        modelSelect = "NB"

                                        # alpha:Additive (Laplace/Lidstone) smoothing parameter(0 for no smoothing).
                                        # "fit_prior": bool, default = True Whether to learn class prior probabilities or not. If false, a uniform prior will be used.
                                        modelParams = {
                                            "alpha": 1.0,
                                            "fit_prior": fp
                                        }

                                        ml_call(vectParams, segParams,
                                                modelSelect, modelParams, resultTimestamp)
                                    except Exception as e:
                                        temp = f'\n{nnNB}\n{e}\n{resultTimestamp}\n{vectParams}\n{segParams}\n{modelSelect}\n{modelParams}'
                                        with open('./info/err.txt', 'a', encoding='utf-8') as f:
                                            f.write(temp)
                                        print(temp)
                                    nnNB += 1

    end1 = time.time()

    # max_df min_df -> float in range [0.0, 1.0]
    nnRF = 1
    for corp in ("./corpus_words/corpus_new.xlsx", "./corpus_words/corpus.xlsx"):
        for ste in range(1, 11):
            for ma in range(ste, 11):
                for mi in range(ma-ste, ma+1):
                    for h, u in ((True, True), (True, False), (False, True), (False, False)):
                        for b in (True, False):
                            for a in ("word", "char", "char_wb"):
                                for n_estimators in range(5, 101, 5):
                                    for criterion in ("gini", "entropy"):
                                        for min_samples_split in range(2, 10):
                                            for min_samples_leafint in range(1, 10):
                                                for max_features in ("auto", "sqrt", "log2"):
                                                    for bootstrap, oob_scorebool, in ((True, True), (True, False), (False, False)):
                                                        for max_samples in range(1, 11):
                                                            for class_weight in (
                                                                    {
                                                                        1: 7.5,
                                                                        2: 9.5,
                                                                        3: 2
                                                                    },
                                                                    None, 'balanced', 'balanced_subsample'):
                                                                try:
                                                                    print(
                                                                        f'***************iteration:{nnRF}***************')
                                                                    resultTimestamp = f"{time.time()}"

                                                                    vectParams = {
                                                                        "analyzer": a,
                                                                        "max_df": ma/10,
                                                                        "min_df": mi/10,
                                                                        "binary": b
                                                                    }

                                                                    segParams = {
                                                                        "corpus": corp,
                                                                        "HMM": h,
                                                                        "use_paddle": u
                                                                    }

                                                                    modelSelect = "RF"

                                                                    # alpha:Additive (Laplace/Lidstone) smoothing parameter(0 for no smoothing).
                                                                    # "fit_prior": bool, default = True Whether to learn class prior probabilities or not. If false, a uniform prior will be used.
                                                                    modelParams = {
                                                                        "n_estimators": n_estimators,
                                                                        "criterion": criterion,
                                                                        "min_samples_split": min_samples_split,
                                                                        "min_samples_leaf": min_samples_leafint,
                                                                        "max_features": max_features,
                                                                        "bootstrap": bootstrap,
                                                                        "oob_score": oob_scorebool,
                                                                        "max_samples": max_samples/10,
                                                                        "class_weight": class_weight
                                                                    }

                                                                    ml_call(vectParams, segParams,
                                                                            modelSelect, modelParams, resultTimestamp)
                                                                except Exception as e:
                                                                    temp = f'\n{nnRF}\n{e}\n{resultTimestamp}\n{vectParams}\n{segParams}\n{modelSelect}\n{modelParams}'
                                                                    with open('./info/err.txt', 'a', encoding='utf-8') as f:
                                                                        f.write(
                                                                            temp)
                                                                    print(temp)
                                                                nnRF += 1
    end2 = time.time()

    print(f'NB time:{(end1-start):.3f}')
    print(f'NB iternations:{nnNB}')

    print(f'RF time:{(end2-end1):.3f}')
    print(f'RF iternations:{nnRF}')
