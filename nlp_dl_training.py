import tensorflow as tf
import tensorflow_hub as hub
from nlp_frame import nlp_frame
import pandas as pd
import time


class nlp_dl_training(nlp_frame):
    def __init__(self) -> None:
        super().__init__()

    def nlp_Bert(self, corpus, HMM=True, use_paddle=True):
        # load corpus(data)
        df = pd.read_csv(corpus, on_bad_lines='skip', encoding='utf-8')

        # Feature Engineering(feature to seg, label to category)
        X = self.seg(
            df["comment"], HMM=HMM, use_paddle=use_paddle)

        y = df["star"].apply(lambda x: int(x)-1)

        df = pd.DataFrame([X, y], index=["X", 'y']).T.dropna()

        tx = tf.constant([tuple(df['X'])])
        tx = tf.transpose(tx)

        # 計算類別數量
        nclasses = len(list(set(df['y'])))
        # 轉換為類別變數
        ty = tf.constant(tf.keras.utils.to_categorical(
            df['y'], nclasses))
        print(tx.shape, ty.shape)

        bert = hub.KerasLayer(
            'nlpModel_BERT/albert_large',
            trainable=True,
            output_key='pooled_output'
        )

        inputs = tf.keras.layers.Input(shape=(None,), dtype=tf.string)
        m = bert(inputs)
        outputs = tf.keras.layers.Dense(nclasses, activation='softmax')(m)

        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(loss='categorical_crossentropy')
        model.fit(tx, ty, epochs=100, batch_size=100)

        tf.keras.utils.plot_model(model, to_file='model.png')

        try:
            predictList = ["青椒好難吃", "番茄很棒"]
            tx = tf.constant(tuple(self.seg(predictList, h, u)))
            tx = tf.transpose([tx])
            logits = model.predict()
            pred = logits.argmax(-1).tolist()
            print(pred)
        except:
            pass

        res = model.evaluate(tx, ty, batch_size=100, verbose=2)
        print(res)

        model.save('nlpModel_BERT/nlp_Bert.h5')


if __name__ == "__main__":
    start = time.time()
    ndt = nlp_dl_training()
    h, u = True, False
    ndt.nlp_Bert(corpus='comment_zh_tw.csv', HMM=h, use_paddle=u)
    end = time.time()-start
    print(f'{end:.3f}')
