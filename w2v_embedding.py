import tensorflow as tf
import math
from gensim import models as GenModels
from gensim.models import word2vec
import numpy as np
import logging
import jieba

class W2VEmbedding(object):
    filename = "w2vmodel/wiki_chs.model"

    EMBEDDING_SIZE = 250
    MIN_COUNT = 5

    def jieba_word(self):
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        # jieba custom setting.
        # jieba.set_dictionary('jieba_dict/dict.txt.big')
        # list = jieba.lcut("这款战车当时被赋予「Kirovets-1（Кировец-1）」的代号")
        # print()
        # load stopwords set
        stopword_set = set()
        # with open('jieba_dict/stopwords.txt','r', encoding='utf-8') as stopwords:
        #     for stopword in stopwords:
        #         stopword_set.add(stopword.strip('\n'))
        output = open('../files/w2v_files/wiki_seg_all.txt', 'w', encoding='utf-8')
        with open('../files/w2v_files/wikichs_all.txt', 'r', encoding='utf-8') as content:
            for texts_num, line in enumerate(content):
                line = line.strip('\n')
                words = jieba.cut(line, cut_all=False)
                for word in words:
                    if word not in stopword_set:
                        output.write(word + ' ')
                output.write('\n')

                if (texts_num + 1) % 10000 == 0:
                    logging.info("已完成前 %d 行的斷詞" % (texts_num + 1))
        output.close()

    def train_gensim(self,load):
        if load:
            # 模型讀取方式
            model = word2vec.Word2Vec.load("../w2vmodel/word2vec.model")
            logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
            sentences = word2vec.LineSentence("../语料/UNKNOWN_WORD.txt")
            # 保存模型，供日後使用
            model.build_vocab(sentences, update=True)
            model.save("../w2vsave/word2vec2.model")

            # 模型讀取方式
            # model = word2vec.Word2Vec.load("your_model_name")
        else:
            logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
            sentences = word2vec.LineSentence("../files/w2v_files/wiki_seg_all.txt")
            model = word2vec.Word2Vec(sentences, size=self.EMBEDDING_SIZE, min_count=self.MIN_COUNT)
            # 保存模型，供日後使用
            model.save("../w2vsave/word2vec.model")

    def toembedding(self):
        genmodel = GenModels.Word2Vec.load(self.filename)
        index2words = {}
        wffile = []
        embedding_matrix = np.zeros((len(genmodel.wv.vocab), self.EMBEDDING_SIZE),dtype=np.float32)
        for i in range(len(genmodel.wv.vocab)):
            embedding_vector = genmodel.wv[genmodel.wv.index2word[i]]
            if embedding_vector is not None:
                index2words[i] = genmodel.wv.index2word[i]
                wffile.append(str(i) + "\t" + genmodel.wv.index2word[i] + "\n")
                embedding_matrix[i] = embedding_vector
            else:
                print("wrong")
        print(embedding_matrix.shape)
        with open("embeddings/index2words.txt", 'w', encoding='utf-8') as wf:
            wf.writelines(wffile)
        np.save("embeddings/wordembedding.npy",embedding_matrix)

    def load_idx_dict(self):
        idx2word = {}
        word2idx = {}
        with open("embeddings/index2words.txt", 'r', encoding='utf-8') as rf:
            lines = rf.readlines()
            for line in lines:
                line = line.strip()
                line = line.split("\t")
                if len(line)!=2:
                    print("wrong")
                idx2word[int(line[0])] = line[1]
                word2idx[line[1]] = int(line[0])
                # print(line[1])
        return idx2word,word2idx

    def test(self):
        idx2word, word2idx = w2v_test.load_idx_dict()
        print(len(idx2word))
        print(len(word2idx))
        print(word2idx["，"])
        embedding = np.load("embeddings/wordembedding.npy")
        print(embedding.shape)
        genwo = GenModels.Word2Vec.load("w2vmodel/wiki_chs.model").wv["保时捷"]
        print(genwo.shape)
        embeddingwo = embedding[word2idx["奥迪"], :]
        print(embeddingwo.shape)
        if (genwo == embeddingwo).all():
            print("yes")
        else:
            print("no")
    def test2(self):
        idx2word, word2idx = w2v_test.load_idx_dict()
        embedding = np.load("embeddings/wordembedding.npy")
        print(idx2word[0])
        print(idx2word[1])
        print(idx2word[2])

if __name__ == "__main__":
    w2v_test = W2VEmbedding()
    # w2v_test.toembedding()
    # idx2word, word2idx = w2v_test.load_idx_dict()
    # print(word2idx["我"])
    # print(idx2word[232])
    w2v_test.test2()
