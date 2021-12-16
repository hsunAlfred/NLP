from joblib import load
from nlp_model_frame import nlp_model_frame


class nlp_model_predict(nlp_model_frame):
    def __init__(self) -> None:
        super().__init__()

    def nlp_NB(self, modelPath: str, predictList: list, h, u):
        model = load(modelPath)

        tex = self.featureTransform(predictList, h, u)

        res = model.predict(tex)

        return tex, res


if __name__ == "__main__":
    nmp = nlp_model_predict()
    params = {
        "modelPath": 'selfModel/(Best)nlp_NB_HMM_False_paddle_False.joblib',
        "predictList": ["青椒好難吃", "番茄很棒"],
        "h": False,
        "u": False
    }

    tex, res = nmp.nlp_NB(**params)
    print(tex, res)
