from numpy import uint
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
# from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import pandas as pd
from joblib import dump
from nlp_frame import nlp_frame
from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.feature_extraction.text import HashingVectorizer
import time
import pathlib


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

    def __loadCorpusAndTransform(self, corpus: str, HMM: bool, use_paddle: bool):
        corpusTarget = corpus.split('/')[2].split('.')[0]

        # load corpus(data)
        # df = pd.read_csv(corpus, on_bad_lines='skip', encoding='utf-8')
        if pathlib.Path(f'./corpus_words/seg_{corpusTarget}_{HMM}_{use_paddle}.xlsx').exists():
            df = pd.read_excel(
                f'./corpus_words/seg_{corpusTarget}_{HMM}_{use_paddle}.xlsx', usecols=['X', 'y']).dropna()
        else:
            df = pd.read_excel(corpus)

            # Feature Engineering(feature to seg, label to category)

            X = self.seg(
                df["comment"], HMM=HMM, use_paddle=use_paddle)

            # y = df["star"]
            y = df["rate"].apply(nlp_frame.toThreeClass)

            df = pd.DataFrame([X, y], index=["X", 'y']).T.dropna()

            df.to_excel(
                f'./corpus_words/seg_{corpusTarget}_{HMM}_{use_paddle}.xlsx', index=False)

            df = pd.read_excel(
                f'./corpus_words/seg_{corpusTarget}_{HMM}_{use_paddle}.xlsx', usecols=['X', 'y']).dropna()
        # BoW transform
        # -----------------------------------
        X = self.vect.fit_transform(df["X"]).toarray()
        # transform to dataframe
        # X = pd.DataFrame(
        #     X, columns=self.vect.get_feature_names_out())
        y = df['y'].astype('category')

        print(f"\n{X.shape}\n{y.shape}")

        # split dataset, random_state should only set in test
        return train_test_split(X, y, train_size=0.8)

    def training(self):
        X_train, X_test, y_train, y_test = self.__loadCorpusAndTransform(
            **self.segParams)

        print(f'train dataset shape:{X_train.shape} {y_train.shape}')
        print(f'test  dataset shape:{X_test.shape} {y_test.shape}')

        model = None
        if self.modelSelect == "NB":
            # model = GaussianNB()
            model = MultinomialNB(**self.modelParams)
        elif self.modelSelect == "RF":
            model = RandomForestClassifier(**self.modelParams)

        model.fit(X_train, y_train)

        cv = cross_val_score(model, X_train, y_train,
                             cv=5, scoring='accuracy').mean()

        y_pred = model.predict(X_test)

        accuracy_score = metrics.accuracy_score(y_test, y_pred)

        confusion_matrix = metrics.confusion_matrix(y_test, y_pred)

        return model, cv, accuracy_score, confusion_matrix


def ml_call(vectParams, segParams, modelSelect, modelParams, resultTimestamp):
    nmt = nlp_model_training(vectParams, segParams, modelSelect, modelParams)
    model, cv, accuracy_score, confusion_matrix = nmt.training()

    dump(
        model, f'nlpModel_{modelSelect}/{resultTimestamp}.joblib')
    dump(
        nmt.vect, f'nlpModel_{modelSelect}/vect_{resultTimestamp}.vect')

    temp = f"\n{resultTimestamp}\n{vectParams}\n{segParams}\n{modelSelect}\n{modelParams}\n" +\
        "-------------------------------------------------------------\n"
    with open(f"./info/{modelSelect}_parameters.txt", mode="a", encoding="utf-8") as f:
        f.write(temp)
    print(temp)

    temp = f"\n{resultTimestamp}\ncross value:{cv:.3f}\naccuracy score:{accuracy_score:.3f}\n" +\
        f"confusion matrix\n{confusion_matrix}\n" +\
        "-------------------------------------------------------------\n"

    with open(f"./info/{modelSelect}_modelScore.txt", 'a', encoding="utf-8") as f:
        f.write(temp)
    print(temp)


if __name__ == "__main__":
    # max_df min_df -> float in range [0.0, 1.0]
    # nn = 1
    # for corp in ("./corpus_words/corpus_new.xlsx", "./corpus_words/corpus.xlsx"):
    #     for ste in range(1, 11):
    #         for ma in range(ste, 11):
    #             for mi in range(ma-ste, ma+1):
    #                 for h, u in ((True, True), (True, False), (False, True), (False, False)):
    #                     for b in (True, False):
    #                         for fp in (True, False):
    #                             for a in ("word", "char", "char_wb"):
    #                                 try:
    #                                     print(
    #                                         f'***************iteration:{nn}***************')
    #                                     resultTimestamp = f"{time.time()}"

    #                                     vectParams = {
    #                                         "analyzer": a,
    #                                         "max_df": ma/10,
    #                                         "min_df": mi/10,
    #                                         "binary": b
    #                                     }

    #                                     segParams = {
    #                                         "corpus": corp,
    #                                         "HMM": h,
    #                                         "use_paddle": u
    #                                     }

    #                                     modelSelect = "NB"

    #                                     # alpha:Additive (Laplace/Lidstone) smoothing parameter(0 for no smoothing).
    #                                     # "fit_prior": bool, default = True Whether to learn class prior probabilities or not. If false, a uniform prior will be used.
    #                                     modelParams = {
    #                                         "alpha": 1.0,
    #                                         "fit_prior": fp
    #                                     }

    #                                     ml_call(vectParams, segParams,
    #                                             modelSelect, modelParams, resultTimestamp)
    #                                 except Exception as e:
    #                                     temp = f'\n{nn}\n{e}\n{resultTimestamp}\n{vectParams}\n{segParams}\n{modelSelect}\n{modelParams}'
    #                                     with open('./info/err.txt', 'a', encoding='utf-8') as f:
    #                                         f.write(temp)
    #                                     print(temp)
    #                                 nn += 1

    # max_df min_df -> float in range [0.0, 1.0]
    nn = 1
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
                                                                    {1: 14, 2: 15,
                                                                     3: 9, 4: 3, 5: 2},
                                                                    None, 'balanced', 'balanced_subsample'):
                                                                try:
                                                                    print(
                                                                        f'***************iteration:{nn}***************')
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
                                                                    temp = f'\n{nn}\n{e}\n{resultTimestamp}\n{vectParams}\n{segParams}\n{modelSelect}\n{modelParams}'
                                                                    with open('./info/err.txt', 'a', encoding='utf-8') as f:
                                                                        f.write(
                                                                            temp)
                                                                    print(temp)
                                                                nn += 1
