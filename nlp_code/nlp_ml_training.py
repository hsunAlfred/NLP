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
from xgboost import XGBClassifier

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
            
            XG use, https://xgboost.readthedocs.io/en/stable/python/python_api.html#module-xgboost.sklearn
            n_estimators: int,#總共迭代的次數，即決策樹的個數。預設值為100
            max_depth: int,#樹的最大深度，默認值為6
            booster: str,#Specify which booster to use: gbtree, gblinear or dart.
            learning_rate: float,#學習速率，預設0.3
            gamma: float,#懲罰項係數，指定節點分裂所需的最小損失函數下降值
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
        elif self.modelSelect == "XG":
            model = XGBClassifier(**self.modelParams)

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


def ml_call(vectParams, segParams, modelSelect, modelParams, resultTimestamp, dumpModel):
    nmt = nlp_model_training(vectParams, segParams, modelSelect, modelParams)
    model, accuracy_score_train, confusion_matrix_train, accuracy_score_vaild, confusion_matrix_vaild,\
        accuracy_score_test, confusion_matrix_test = nmt.training()

    if dumpModel == True:
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
    start = time.time()

    # try:
    #     resultTimestamp = f"{time.time()}"

    #     vectParams = {
    #         "analyzer": "char_wb",
    #         "max_df": 0.8,
    #         "min_df": 0.0,
    #         "binary": False
    #     }

    #     segParams = {
    #         "corpus": "./corpus_words/corpus_new.xlsx",
    #         "HMM": True,
    #         "use_paddle": False
    #     }

    #     modelSelect = "NB"

    #     # alpha:Additive (Laplace/Lidstone) smoothing parameter(0 for no smoothing).
    #     # "fit_prior": bool, default = True Whether to learn class prior probabilities or not. If false, a uniform prior will be used.
    #     modelParams = {
    #         "alpha": 1.0,
    #         "fit_prior": True
    #     }

    #     ml_call(vectParams, segParams,
    #             modelSelect, modelParams, resultTimestamp)
    # except Exception as e:
    #     temp = f'\n{e}\n{resultTimestamp}\n{vectParams}\n{segParams}\n{modelSelect}\n{modelParams}'
    #     with open('./info/err.txt', 'a', encoding='utf-8') as f:
    #         f.write(temp)
    #     print(temp)

    # end1 = time.time()

    # print(f'NB time:{(end1-start):.3f}')
    # max_df min_df -> float in range [0.0, 1.0]
    # start = time.time()
    # nnNB = 1
    # for ma in range(1, 11):
    #     for h, u in ((True, True), (True, False)):
    #         for b in (True, False):
    #             for a in ("char", "char_wb"):
    #                 try:
    #                     print(
    #                         f'***************iteration:{nnNB}***************')
    #                     resultTimestamp = f"{time.time()}"

    #                     vectParams = {
    #                         "analyzer": a,
    #                         "max_df": ma/10,
    #                         "min_df": 0.0,
    #                         "binary": b
    #                     }

    #                     segParams = {
    #                         "corpus": "./corpus_words/corpus_new2.xlsx",
    #                         "HMM": h,
    #                         "use_paddle": u
    #                     }

    #                     modelSelect = "NB"

    #                     # alpha:Additive (Laplace/Lidstone) smoothing parameter(0 for no smoothing).
    #                     # "fit_prior": bool, default = True Whether to learn class prior probabilities or not. If false, a uniform prior will be used.
    #                     modelParams = {
    #                         "alpha": 1.0,
    #                         "fit_prior": True
    #                     }

    #                     ml_call(vectParams, segParams,
    #                             modelSelect, modelParams, resultTimestamp, dumpModel=False)
    #                 except Exception as e:
    #                     temp = f'\n{nnNB}\n{e}\n{resultTimestamp}\n{vectParams}\n{segParams}\n{modelSelect}\n{modelParams}'
    #                     with open('./info/err.txt', 'a', encoding='utf-8') as f:
    #                         f.write(temp)
    #                     print(temp)
    #                 nnNB += 1

    # end1 = time.time()

    # max_df min_df -> float in range [0.0, 1.0]
    # nnRF = 1
    # for n_estimators in range(60, 201, 5):
    #     for criterion in ("gini", "entropy"):
    #         for min_samples_split in range(5, 10):
    #             for min_samples_leaf in range(1, 5):
    #                 for max_features in ("auto", "sqrt"):
    #                     for max_samples in range(3, 10):
    #                         for class_weight in (
    #                                 {
    #                                     1: 12,
    #                                     2: 16,
    #                                     3: 9,
    #                                     4: 3,
    #                                     5: 2
    #                                 },
    #                                 'balanced', 'balanced_subsample'):
    #                             try:
    #                                 print(
    #                                     f'***************iteration:{nnRF}***************')
    #                                 resultTimestamp = f"{time.time()}"

    #                                 vectParams = {
    #                                     "analyzer": "char_wb",
    #                                     "max_df": 0.8,
    #                                     "min_df": 0.0,
    #                                     "binary": False
    #                                 }

    #                                 segParams = {
    #                                     "corpus": "./corpus_words/corpus_new2.xlsx",
    #                                     "HMM": True,
    #                                     "use_paddle": False
    #                                 }

    #                                 modelSelect = "RF"

    #                                 # alpha:Additive (Laplace/Lidstone) smoothing parameter(0 for no smoothing).
    #                                 # "fit_prior": bool, default = True Whether to learn class prior probabilities or not. If false, a uniform prior will be used.
    #                                 modelParams = {
    #                                     "n_estimators": n_estimators,
    #                                     "criterion": criterion,
    #                                     "min_samples_split": min_samples_split,
    #                                     "min_samples_leaf": min_samples_leaf,
    #                                     "max_features": max_features,
    #                                     "max_samples": max_samples/10,
    #                                     "class_weight": class_weight
    #                                 }

    #                                 ml_call(vectParams, segParams,
    #                                         modelSelect, modelParams, resultTimestamp, dumpModel=False)
    #                             except Exception as e:
    #                                 temp = f'\n{nnRF}\n{e}\n{resultTimestamp}\n{vectParams}\n{segParams}\n{modelSelect}\n{modelParams}'
    #                                 with open('./info/err.txt', 'a', encoding='utf-8') as f:
    #                                     f.write(
    #                                         temp)
    #                                 print(temp)
    #                             nnRF += 1
    # end2 = time.time()

    # print(f'NB time:{(end1-start):.3f}')
    # print(f'NB iternations:{nnNB}')

    # print(f'RF time:{(end2-end1):.3f}')
    # print(f'RF iternations:{nnRF}')

#     try:
#         resultTimestamp = f"{time.time()}"

#         vectParams = {
#             "analyzer": "char_wb",
#             "max_df": 0.8,
#             "min_df": 0.0,
#             "binary": False
#         }

#         segParams = {
#             "corpus": "./corpus_words/corpus_new.xlsx",
#             "HMM": True,
#             "use_paddle": False
#         }

#         modelSelect = "RF"

#         # alpha:Additive (Laplace/Lidstone) smoothing parameter(0 for no smoothing).
#         # "fit_prior": bool, default = True Whether to learn class prior probabilities or not. If false, a uniform prior will be used.
#         modelParams = {
#             "n_estimators": 65,
#             "criterion": 'entropy',
#             "min_samples_split": 9,
#             "min_samples_leaf": 1,
#             "max_features": 'log2',
#             "max_samples": 0.9,
#             "class_weight": 'balanced_subsample'
#         }

#         ml_call(vectParams, segParams,
#                 modelSelect, modelParams, resultTimestamp, dumpModel=True)
    try:
        resultTimestamp = f"{time.time()}"

        vectParams = {
            "analyzer": "char_wb",
            "max_df": 0.8,
            "min_df": 0.0,
            "binary": False
        }

        segParams = {
            "corpus": "./corpus_words/corpus_new.xlsx",
            "HMM": True,
            "use_paddle": False
        }

        modelSelect = "XG"

        # alpha:Additive (Laplace/Lidstone) smoothing parameter(0 for no smoothing).
        # "fit_prior": bool, default = True Whether to learn class prior probabilities or not. If false, a uniform prior will be used.
        modelParams = {
            "n_estimators":100,
            "max_depth": 10,
            "booster": "dart",#Specify which booster to use: gbtree, gblinear or dart.
            "learning_rate": 0.3,#學習速率，預設0.3。
            "gpu_id" :0
        }

        ml_call(vectParams, segParams,
                modelSelect, modelParams, resultTimestamp, dumpModel=False)
    except Exception as e:
        temp = f'\n{e}\n{resultTimestamp}\n{vectParams}\n{segParams}\n{modelSelect}\n{modelParams}'
        with open('./info/err.txt', 'a', encoding='utf-8') as f:
            f.write(
                temp)
        print(temp)
