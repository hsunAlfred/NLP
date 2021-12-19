from joblib import load
from nlp_frame import nlp_frame


class nlp_ml_predict(nlp_frame):
    def __init__(self) -> None:
        super().__init__()

    def predict(self, modelPath: str, vectPath: str, predictList: list, h: bool, u: bool):
        model = load(modelPath)
        vect = load(vectPath)

        tex = self.seg(predictList, h, u)
        tex = vect.fit_transform(tex).toarray()

        res = model.predict(tex)

        return tex, res


if __name__ == "__main__":
    nmp = nlp_ml_predict()
    params = {
        "modelPath": 'nlpModel_NB/nlp_NB_HMM_True_paddle_False.joblib',
        "vectPath": 'nlpModel_NB/nlp_vect_HMM_True_paddle_True.vect',
        "predictList": ["青椒好難吃", "番茄很棒"],
        "h": False,
        "u": False
    }

    tex, res = nmp.predict(**params)
    print(params['predictList'], res)
