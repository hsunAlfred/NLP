import jieba
import jieba.posseg as pseg
import numpy as np
import paddle
import pathlib
import pandas as pd

class nlp_frame:
    def __init__(self) -> None:
        # load stop word
        with open("./corpus_words/stopwords.txt", encoding='utf-8') as f:
            stopwords = f.read()
        self.custom_stopwords_list = [i for i in stopwords.split('\n')]

        # self.avoid_word_kind = (
        #     'nr', 'PER', 'ns', 'LOC', 's', 'nt', 'ORG', 'nw', 'w', 'TIME')
        #self.avoid_word_kind = ('w')
        self.avoid_word_kind = ()

    def seg(self, waitTransform, HMM=True, use_paddle=True):
        jieba.set_dictionary('./corpus_words/dict.txt.big')
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
    
    def loadCorpus(self, corpus: str, HMM: bool, use_paddle: bool):
        corpusTarget = corpus.split('/')[-1].split('.')[0]

        # load corpus(data)
        # df = pd.read_csv(corpus, on_bad_lines='skip', encoding='utf-8')
        if pathlib.Path(f'./corpus_words/seg_{corpusTarget}_{HMM}_{use_paddle}.xlsx').exists():
            df = pd.read_excel(
                f'./corpus_words/seg_{corpusTarget}_{HMM}_{use_paddle}.xlsx', usecols=['X', 'y']).dropna()
        else:
            df = pd.read_excel(corpus)

            # Feature Engineering(feature to seg, label to category)

            X = self.seg(
                df["comment"], HMM=HMM, use_paddle=use_paddle)

            # y = df["star"]
            y = df["rate"]  # .apply(nlp_frame.toThreeClass)

            df = pd.DataFrame([X, y], index=["X", 'y']).T.dropna()

            df.to_excel(
                f'./corpus_words/seg_{corpusTarget}_{HMM}_{use_paddle}.xlsx', index=False)

            df = pd.read_excel(
                f'./corpus_words/seg_{corpusTarget}_{HMM}_{use_paddle}.xlsx', usecols=['X', 'y']).dropna()
        
        return df

    @staticmethod
    def toThreeClass(x):
        if x > + 4:
            return 3
        elif x == 3:
            return 2
        else:
            return 1
