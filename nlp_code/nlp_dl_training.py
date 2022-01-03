from sklearn.model_selection import train_test_split
import time
import pandas as pd
from nlp_frame import nlp_frame
import tensorflow_hub as hub
import tensorflow as tf
import os
from matplotlib import image as mpimg
from matplotlib import pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


class nlp_dl_training(nlp_frame):
    def __init__(self) -> None:
        super().__init__()
        self.X_tokenizer = None
        self.max_seq_len = None

    def __transTensor(self, nclasses, X_train, y_train=None,
                      X_vaild=None, y_vaild=None, X_test=None, y_test=None):

        # ------------------train data------------------------------
        X_temp = tf.constant([tuple(X_train)])
        tX_train = tf.transpose(X_temp)

        try:
            if y_train.empty:
                print(tX_train.shape)
                return tX_train
        except:
            if y_train == None:
                print(tX_train.shape)
                return tX_train

        # 轉換為類別變數
        ty_train = tf.constant(tf.keras.utils.to_categorical(
            y_train, nclasses))

        print(f'train dataset shape: {tX_train.shape} {ty_train.shape}')

        # ------------------vaild data------------------------------
        X_temp = tf.constant([tuple(X_vaild)])
        tX_vaild = tf.transpose(X_temp)

        # 轉換為類別變數
        ty_vaild = tf.constant(tf.keras.utils.to_categorical(
            y_vaild, nclasses))

        print(f'vaild dataset shape: {tX_vaild.shape} {ty_vaild.shape}')

        # ------------------test data------------------------------
        X_temp = tf.constant([tuple(X_test)])
        tX_test = tf.transpose(X_temp)

        # 轉換為類別變數
        ty_test = tf.constant(tf.keras.utils.to_categorical(
            y_test, nclasses))

        print(f'test dataset shape: {tX_test.shape} {ty_test.shape}')

        return tX_train, ty_train, tX_vaild, ty_vaild, tX_test, ty_test

    def __transTensor2(self, nclasses, X_train, y_train=None,
                       X_vaild=None, y_vaild=None, X_test=None, y_test=None):

        try:
            if X_vaild.empty:
                corpus = pd.concat([X_train])
        except:
            if X_vaild == None:
                corpus = pd.concat([X_train])
        else:
            corpus = pd.concat([X_train, X_vaild, X_test])

        # ------------------train data------------------------------
        if self.X_tokenizer == None:
            self.X_tokenizer = tf.keras \
                .preprocessing \
                .text \
                .Tokenizer(num_words=10000)
            self.X_tokenizer.fit_on_texts(corpus)

        X_train_token = self.X_tokenizer.texts_to_sequences(X_train)
        if self.max_seq_len == None:
            self.max_seq_len = max([len(seq) for seq in X_train_token])
        tX_train = tf.keras.preprocessing \
            .sequence \
            .pad_sequences(X_train_token, maxlen=self.max_seq_len)

        try:
            if X_vaild.empty:
                print(tX_train.shape)
                return tX_train
        except:
            if X_vaild == None:
                print(tX_train.shape)
                return tX_train

        # 轉換為類別變數
        ty_train = tf.constant(tf.keras.utils.to_categorical(
            y_train, nclasses))

        print(tX_train.shape, ty_train.shape)

        # ------------------vaild data------------------------------
        X_vaild_token = self.X_tokenizer.texts_to_sequences(X_vaild)
        tX_vaild = tf.keras.preprocessing \
            .sequence \
            .pad_sequences(X_vaild_token, maxlen=self.max_seq_len)

        # 轉換為類別變數
        ty_vaild = tf.constant(tf.keras.utils.to_categorical(
            y_vaild, nclasses))

        print(tX_vaild.shape, ty_vaild.shape)

        # ------------------test data------------------------------
        X_test_token = self.X_tokenizer.texts_to_sequences(X_test)
        tX_test = tf.keras.preprocessing \
            .sequence \
            .pad_sequences(X_test_token, maxlen=self.max_seq_len)

        # 轉換為類別變數
        ty_test = tf.constant(tf.keras.utils.to_categorical(
            y_test, nclasses))

        print(ty_test.shape, y_test.shape)

        return tX_train, ty_train, tX_vaild, ty_vaild, tX_test, ty_test

    def __threeClasses(self, y):
        if y >= 4:
            return 2
        elif y == 3:
            return 1
        else:
            return 0

    def __loadCorpusAndSplit(self, corpus: str, HMM: bool, use_paddle: bool):
        df = self.loadCorpus(corpus, HMM, use_paddle)

        X = df["X"]
        # y = df['y'].apply(lambda tem: tem-1)
        y = df['y'].apply(self.__threeClasses)

        # 計算類別數量
        nclasses = len(list(set(y)))

        print(f"\n{X.shape}\n{y.shape}")

        # split dataset, random_state should only set in test
        X_v, X_test, y_v, y_test = \
            train_test_split(X, y, train_size=0.8, stratify=y)

        X_train, X_vaild, y_train, y_vaild = \
            train_test_split(X_v, y_v,
                             train_size=0.75, stratify=y_v)

        return X_train, y_train, X_vaild, y_vaild, X_test, y_test, nclasses

    def __modelScore(self, model, tX_train, y_train, tX_vaild, y_vaild, tX_test, y_test):
        pred = {}
        logits = model.predict(tX_train)
        pred["train"] = (logits.argmax(-1).tolist(), y_train)

        logits = model.predict(tX_vaild)
        pred["vaild"] = (logits.argmax(-1).tolist(), y_vaild)

        logits = model.predict(tX_test)
        pred["test"] = (logits.argmax(-1).tolist(), y_test)

        return pred

    def nlp_LSTM_Build(self, corpus, HMM: bool, use_paddle: bool, epochs: int):
        X_train, y_train, X_vaild, y_vaild, X_test, y_test, nclasses = \
            self.__loadCorpusAndSplit(corpus, HMM, use_paddle)

        # transform to tensor
        # x.shape, y.shape must to be (n, 1) (n, classes)
        tX_train, ty_train, tX_vaild, ty_vaild, tX_test, ty_test = self.__transTensor2(
            nclasses, X_train, y_train, X_vaild, y_vaild, X_test, y_test)

        # 輸入層
        top_input = tf.keras.Input(
            shape=(self.max_seq_len,),
            dtype='int32'
        )

        # 詞嵌入層
        # 經過詞嵌入層的轉換，兩個新聞標題都變成一個詞向量的序列
        # 而每個詞向量的維度為 256
        embedding_layer = tf.keras.layers.Embedding(10000, 256)
        top_embedded = embedding_layer(top_input)

        # LSTM 層
        # 兩個新聞標題經過此層後
        # 為一個 128 維度向量
        shared_lstm = tf.keras.layers.LSTM(128)
        top_output = shared_lstm(top_embedded)

        # 全連接層搭配 Softmax Activation
        # 可以回傳 5 個成對標題，屬於各類別的可能機率
        dense = tf.keras.layers.Dense(units=nclasses, activation='softmax')
        predictions = dense(top_output)

        model = tf.keras.Model(inputs=top_input, outputs=predictions)
        model.compile(
            optimizer='rmsprop',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        history = model.fit(
            x=tX_train,
            y=ty_train,
            batch_size=1000,
            epochs=epochs,
            validation_data=(tX_vaild, ty_vaild),
            shuffle=True
        )

        tf.keras.utils.plot_model(model, to_file='./info/model.png')

        pred = self.__modelScore(
            model, tX_train, y_train, tX_vaild, y_vaild, tX_test, y_test)

        return model, nclasses, pred, history

    def nlp_LSTM_Predict(self, model, nclasses, predictList, HMM, use_paddle):
        predictList = self.seg(
            predictList, HMM=HMM, use_paddle=use_paddle)

        par = {
            "nclasses": nclasses,
            "X_train": pd.Series(predictList),
            "y_train": None,
            "X_test": None,
            "y_test": None
        }

        tx_pred = self.__transTensor2(**par)
        logits = model.predict(tx_pred)
        pred = logits.argmax(-1).tolist()

        return logits, pred

    def nlp_Bert_Build(self, corpus, epochs: int):
        df = pd.read_excel(corpus)

        # Feature Engineering(feature to seg, label to category)

        X = df["comment"]

        y = df['y'].apply(self.__threeClasses)

        # 計算類別數量
        nclasses = len(list(set(y)))

        df = pd.DataFrame([X, y], index=["X", 'y']).T.dropna()

        # split dataset, random_state should only set in test
        X_v, X_test, y_v, y_test = \
            train_test_split(X, y, train_size=0.8, stratify=y)

        X_train, X_vaild, y_train, y_vaild = \
            train_test_split(X_v, y_v,
                             train_size=0.75, stratify=y_v)

        # transform to tensor
        # x.shape, y.shape must to be (n, 1) (n, classes)
        tX_train, ty_train, tX_vaild, ty_vaild, tX_test, ty_test = self.__transTensor(
            nclasses, X_train, y_train, X_vaild, y_vaild, X_test, y_test)

        # 輸入層
        inputs_bert = tf.keras.layers.Input(
            shape=(None,),
            dtype=tf.string
        )

        # bert layer
        bert = hub.KerasLayer(
            'corpus_words/albert_large',
            trainable=True,
            output_key='pooled_output'
        )
        m = bert(inputs_bert)

        # 全連接層搭配 Softmax Activation
        # 可以回傳 5 個成對標題，屬於各類別的可能機率
        m = tf.keras.layers.Masking()(m)
        outputs = tf.keras.layers.Dense(nclasses, activation='softmax')(m)

        model = tf.keras.Model(inputs=inputs_bert, outputs=outputs)
        model.compile(
            optimizer='rmsprop',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        history = model.fit(
            x=tX_train,
            y=ty_train,
            batch_size=1000,
            epochs=epochs,
            validation_data=(tX_vaild, ty_vaild),
            shuffle=True
        )

        # tf.keras.utils.plot_model(model, to_file='model.png')

        pred = self.__modelScore(
            model, tX_train, y_train, tX_vaild, y_vaild, tX_test, y_test)

        return model, nclasses, pred, history

    def nlp_Bert_Predict(self, model, nclasses, predictList, HMM, use_paddle):
        predictList = self.seg(
            predictList, HMM=HMM, use_paddle=use_paddle)

        par = {
            "nclasses": nclasses,
            "X_train": predictList,
            "y_train": None,
            "X_test": None,
            "y_test": None
        }

        tx_pred = self.__transTensor(**par)
        logits = model.predict(tx_pred)
        pred = logits.argmax(-1).tolist()

        return logits, pred


def main(mo="BERT"):
    start = time.time()

    ndt = nlp_dl_training()

    if mo == "BERT":
        params = {
            "corpus": './corpus_words/corpus_new.xlsx',
            "epochs": 200
        }

        model, nclasses, pred, history = ndt.nlp_Bert_Build(**params)
    elif mo == "LSTM":
        params = {
            "corpus": './corpus_words/corpus_new.xlsx',
            "HMM": True,
            "use_paddle": False,
            "epochs": 2
        }
        model, nclasses, pred, history = ndt.nlp_LSTM_Build(**params)

    temp = {
        "accuracy": history.history['accuracy'],
        "val_accuracy": history.history['val_accuracy'],
        "loss": history.history['loss'],
        "val_loss": history.history['val_loss']
    }
    df = pd.DataFrame(temp)
    df.to_csv(f'./info/{mo}_acc_los.csv')

    res = str(model.summary())
    for k, v in pred.items():
        res += f'{k}\nconfusion matrix\n{tf.math.confusion_matrix(labels = v[1], predictions = v[0])}'

    with open(f'info/{mo}_result.txt', 'w') as f:
        f.write(res)

    # params = {
    #     "model": model,
    #     "nclasses": nclasses,
    #     "predictList": ["青椒難吃", "番茄好吃"],
    #     "HMM": True,
    #     "use_paddle": False
    # }
    # logits, pred = ndt.nlp_LSTM_Predict(**params)
    # logits, pred = ndt.nlp_Bert_Predict(**params)
    # print(f'logits\n{logits}\n')
    # print(f'predict\n{pred}\n')
    end = time.time()-start
    print(f'{end:.3f}')


if __name__ == "__main__":
    main()
