import tensorflow as tf
from nlp_frame import nlp_frame


class nlp_dl_predict(nlp_frame):
    def __init__(self) -> None:
        super().__init__()

    def nlp_Bert(self, modelPath: str, predictList: list, h, u):
        model = tf.keras.models.load_model(modelPath)

        tx = tf.constant(tuple(self.seg(predictList, h, u)))
        tx = tf.transpose([tx])
        logits = model.predict(tx)
        pred = logits.argmax(-1).tolist()
        return logits, pred


if __name__ == "__main__":
    ndp = nlp_dl_predict()
    params = {
        "modelPath": "nlpModel_BERT/nlp_Bert.h5",
        "predictList": ["青椒好難吃", "番茄很棒"],
        "h": True,
        "u": False
    }

    logits, pred = ndp.nlp_Bert(**params)
    print(logits)
    print(pred)
