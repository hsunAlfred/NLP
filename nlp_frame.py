import jieba
import jieba.posseg as pseg
import numpy as np
import paddle
# from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer


class nlp_frame:
    def __init__(self) -> None:
        # load stop word
        with open("stopwords.txt", encoding='utf-8') as f:
            stopwords = f.read()
        self.custom_stopwords_list = [i for i in stopwords.split('\n')]

        # self.avoid_word_kind = (
        #     'nr', 'PER', 'ns', 'LOC', 's', 'nt', 'ORG', 'nw', 'w', 'TIME')
        #self.avoid_word_kind = ('w')
        self.avoid_word_kind = ()

        self.vect = HashingVectorizer(n_features=2**15)

    def seg(self, waitTransform, HMM=True, use_paddle=True):
        jieba.set_dictionary('dict.txt.big')
        if use_paddle:
            paddle.enable_static()
            jieba.enable_paddle()
        trans = []
        n = 1
        for i in waitTransform:
            print(f'\r{n}/{len(waitTransform)}', end='')
            n += 1
            pc = pseg.lcut(str(i), HMM=HMM, use_paddle=use_paddle)
            temp = []
            for j in pc:
                if tuple(j)[1] not in self.avoid_word_kind:
                    if tuple(j)[0] not in self.custom_stopwords_list:
                        temp.append(tuple(j)[0])
            trans.append(' '.join(temp))

        return np.array(trans)

    def transTensor(self, x, y, nclasses):
        import tensorflow as tf
        tx = tf.constant([tuple(x)])
        tx = tf.transpose(tx)

        # 轉換為類別變數
        ty = tf.constant(tf.keras.utils.to_categorical(
            y, nclasses))
        print(tx.shape, ty.shape)
        return tx, ty
