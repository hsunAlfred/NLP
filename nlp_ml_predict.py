from joblib import load
from nlp_frame import nlp_frame


class nlp_ml_predict(nlp_frame):
    def __init__(self) -> None:
        super().__init__()

    def nlp_NB(self, modelPath: str, vectPath: str, predictList: list, h, u):
        model = load(modelPath)
        vect = load(vectPath)

        tex = self.seg(predictList, h, u)
        tex = vect.transform(tex)

        res = model.predict(tex)

        return tex, res


if __name__ == "__main__":
    nmp = nlp_ml_predict()
    params = {
        "modelPath": 'nlpModel_NB/(Best)nlp_NB_HMM_True_paddle_False.joblib',
        "vectPath": 'nlpModel_NB/nlp_vect_HMM_True_paddle_False.vect',
        "predictList": ["青椒好難吃", "番茄很棒"],
        "h": False,
        "u": False
    }

    tex, res = nmp.nlp_NB(**params)
    print(params['predictList'], res)
