import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
import tensorflow_hub as hub
from nlp_frame import nlp_frame
import pandas as pd
import time
from sklearn.model_selection import train_test_split


class nlp_dl_training(nlp_frame):
    def __init__(self) -> None:
        super().__init__()
        self.X_tokenizer = None
        self.max_seq_len = None

    def __transTensor(self, nclasses, X_train, y_train = None, 
                                X_vaild = None, y_vaild=None, X_test=None, y_test=None):
        
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

        print(f'test dataset shape: {tX_train.shape} {ty_train.shape}')

        return tX_train, ty_train, tX_vaild, ty_vaild, tX_test, ty_test

    def __transTensor2(self, nclasses, X_train, y_train = None, 
                                X_vaild = None, y_vaild=None, X_test=None, y_test=None):
        
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

        return self.max_seq_len, tX_train, ty_train, tX_vaild, ty_vaild, tX_test, ty_test

    def __loadCorpusAndSplit(self, corpus: str, HMM: bool, use_paddle: bool):
        df = self.loadCorpus(corpus, HMM, use_paddle)

        X = df["X"]
        y = df['y'].apply(lambda tem:tem-1)

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

    def nlp_LSTM_Build(self, corpus, HMM: bool, use_paddle: bool, epochs: int):
        X_train, y_train, X_vaild, y_vaild, X_test, y_test, nclasses = \
            self.__loadCorpusAndSplit(corpus, HMM, use_paddle)

        # transform to tensor
        # x.shape, y.shape must to be (n, 1) (n, classes)
        max_seq_len, tX_train, ty_train, tX_vaild, ty_vaild, tX_test, ty_test = self.__transTensor2(
            nclasses, X_train, y_train, X_vaild, y_vaild, X_test, y_test)

        # 輸入層
        top_input = tf.keras.Input(
                shape=(max_seq_len,), 
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
        dense =  tf.keras.layers.Dense(units=nclasses, activation='softmax')
        predictions = dense(top_output)

        model = tf.keras.Model(inputs=top_input, outputs=predictions)
        model.compile(
            optimizer='rmsprop',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        model.fit(
            x=tX_train, 
            y=ty_train,
            batch_size=1000,
            epochs=epochs,
            validation_data=(tX_vaild, ty_vaild),
            shuffle=True
        )
        
        # tf.keras.utils.plot_model(model, to_file='model.png')

        evaluate_loss = model.evaluate(
            tX_test, ty_test, batch_size=1000, verbose=2)
        logits = model.predict(tX_test)
        pred = logits.argmax(-1).tolist()

        return model, nclasses, evaluate_loss, logits, pred, ty_test

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

    def nlp_Bert_Build(self, corpus, HMM: bool, use_paddle: bool, epochs: int):
        X_train, y_train, X_vaild, y_vaild, X_test, y_test, nclasses = \
            self.__loadCorpusAndSplit(corpus, HMM, use_paddle)

        # transform to tensor
        # x.shape, y.shape must to be (n, 1) (n, classes)
        tX_train, ty_train, tX_vaild, ty_vaild, tX_test, ty_test = self.__transTensor(
            nclasses, X_train, y_train, X_vaild, y_vaild, X_test, y_test)

        bert = hub.KerasLayer(
            'corpus_words/albert_large',
            trainable=True,
            output_key='pooled_output'
        )

        inputs = tf.keras.layers.Input(shape=(None,), dtype=tf.string)
        m = bert(inputs)
        m = tf.keras.layers.Masking()(m)
        outputs = tf.keras.layers.Dense(nclasses, activation='softmax')(m)

        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(loss='categorical_crossentropy')
        model.fit(tX_train, ty_train, epochs=epochs, batch_size=1000, 
            validation_data = (tX_vaild, ty_vaild))
        
        # tf.keras.utils.plot_model(model, to_file='model.png')

        evaluate_loss = model.evaluate(
            tX_test, ty_test, batch_size=1000, verbose=2)
        logits = model.predict(tX_test)
        pred = logits.argmax(-1).tolist()

        return model, nclasses, evaluate_loss, logits, pred, ty_test

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


if __name__ == "__main__":
    start = time.time()
    ndt = nlp_dl_training()
    params = {
        "corpus": './corpus_words/corpus_new.xlsx',
        "HMM": True,
        "use_paddle": False,
        "epochs": 3000
    }
    # model, nclasses, evaluate_loss, logits, pred, ty_test =\
    #     ndt.nlp_Bert_Build(**params)
    model, nclasses, evaluate_loss, logits, pred, ty_test =\
        ndt.nlp_LSTM_Build(**params)
    print(f'evaluate\n{evaluate_loss}\n')

    params = {
        "model": model,
        "nclasses": nclasses,
        "predictList": ["青椒難吃", "番茄好吃"],
        "HMM": True,
        "use_paddle": False
    }
    logits, pred = ndt.nlp_LSTM_Predict(**params)
    # logits, pred = ndt.nlp_Bert_Predict(**params)
    print(f'logits\n{logits}\n')
    print(f'predict\n{pred}\n')

    end = time.time()-start
    print(f'{end:.3f}')
