import tensorflow as tf
import tensorflow_hub as hub
from nlp_frame import nlp_frame
import pandas as pd
import time
from sklearn.model_selection import train_test_split


class nlp_dl_training(nlp_frame):
    def __init__(self) -> None:
        super().__init__()

    def nlp_Bert_Build(self, corpus, HMM: bool, use_paddle: bool, epochs: int):
        # load corpus(data)
        df = pd.read_csv(corpus, on_bad_lines='skip', encoding='utf-8')

        # Feature Engineering(feature to seg, label to category)
        X = self.seg(
            df["comment"], HMM=HMM, use_paddle=use_paddle)

        y = df["star"].apply(lambda x: int(x)-1)

        df = pd.DataFrame([X, y], index=["X", 'y']).T.dropna()

        # 計算類別數量
        nclasses = len(list(set(df["y"])))

        # split dataset
        X_train, X_test, y_train, y_test = train_test_split(
            df["X"], df["y"], test_size=0.2)
        print(X_train.shape, y_train.shape)
        print(X_test.shape, y_test.shape)

        # transform to tensor
        # x.shape, y.shape must to be (n, 1) (n, classes)
        tX_train, ty_train = self.transTensor(
            X_train, y_train, nclasses)

        tX_test, ty_test = self.transTensor(
            X_test, y_test, nclasses)

        bert = hub.KerasLayer(
            'nlpModel_BERT/albert_large',
            trainable=True,
            output_key='pooled_output'
        )

        inputs = tf.keras.layers.Input(shape=(None,), dtype=tf.string)
        m = bert(inputs)
        m = tf.keras.layers.Masking()(m)
        outputs = tf.keras.layers.Dense(nclasses, activation='softmax')(m)

        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(loss='categorical_crossentropy')
        model.fit(tX_train, ty_train, epochs=epochs, batch_size=100)

        tf.keras.utils.plot_model(model, to_file='model.png')

        evaluate_loss = model.evaluate(
            tX_test, ty_test, batch_size=100, verbose=2)
        logits = model.predict(tX_test)
        pred = logits.argmax()

        return model, nclasses, evaluate_loss, logits, pred, ty_test

    def nlp_Bert_Predict(self, model, nclasses, predictList, HMM, use_paddle):
        predictList = self.seg(
            predictList, HMM=HMM, use_paddle=use_paddle)
        tx_pred, ty = self.transTensor(
            predictList, [0, 4], nclasses)
        logits = model.predict(tx_pred)
        pred = logits.argmax()

        return logits, pred, ty


if __name__ == "__main__":
    start = time.time()
    ndt = nlp_dl_training()
    params = {
        "corpus": 'comment_zh_tw.csv',
        "HMM": True,
        "use_paddle": False,
        "epochs": 1
    }
    model, nclasses, evaluate_loss, logits, pred, ty_test = \
        ndt.nlp_Bert_Build(**params)
    print(f'evaluate\n{evaluate_loss}\n')

    params = {
        "model": model,
        "nclasses": nclasses,
        "predictList": ["青椒難吃", "番茄好吃"],
        "HMM": True,
        "use_paddle": False
    }
    logits, pred, ty = ndt.nlp_Bert_Predict(**params)
    print(f'logits\n{logits}\n')
    print(f'predict\n{pred}\n')

    end = time.time()-start
    print(f'{end:.3f}')
