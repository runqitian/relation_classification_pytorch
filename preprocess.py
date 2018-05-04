
from __future__ import print_function
import numpy as np
import gzip
import os
import sys
if (sys.version_info > (3, 0)):
    import pickle as pkl
else: #Python 2.7 imports
    import cPickle as pkl
import jieba
from jieba import posseg as pseg
import codecs
from pyltp import Parser


class PreTrain(object):
    """
        22类关系
    """
    relationsMapping = {"locaA": 0, "locAa": 1,
                        "med-ill": 2, "ill-med": 3,
                        "clsaA": 4, "clsAa": 5,
                        "w-c": 6, "c-w": 7,
                        "cs-ef": 8, "ef-cs": 9,
                        "pfs-name": 10,"name-pfs": 11,
                        "org": 12,"prdc": 13,
                        "discribe": 14, "is": 15,
                        "pdc-pdt": 16,"pdt-pdc": 17,
                        "user-tool": 18,"tool-user": 19,
                        "same": 20,
                        'other': 21
                        }

    distanceMapping = {'PADDING': 0, 'LowerMin': 1, 'GreaterMax': 2}
    minDistance = -99
    maxDistance = 99

    maxSentenceLen = 100
    max_distance = 204

    parser = Parser()# 初始化实例
    def __init__(self):
        self.parser.load("LTP/parser.model")  # 加载模型
        self.embedding = np.load("embeddings/wordembedding.npy")
        self.idx2word, self.word2idx, self.vocab = self.load_idx_dict()
        for dis in range(self.minDistance, self.maxDistance + 1):
            self.distanceMapping[str(dis)] = len(self.distanceMapping)
    def load_idx_dict(self):
        idx2word = {}
        word2idx = {}
        vocab = []
        with open("embeddings/index2words.txt", 'r', encoding='utf-8') as rf:
            lines = rf.readlines()
            for line in lines:
                line = line.strip()
                line = line.split("\t")
                if len(line) != 2:
                    print("wrong")
                idx2word[int(line[0])] = line[1]
                word2idx[line[1]] = int(line[0])
                vocab.append(line[1])
                # print(line[1])
        return idx2word, word2idx, vocab


    def sentence_w2v(self,pos1,pos2,sentence):
        # print(sentence)
        pos1 = int(pos1)
        pos2 = int(pos2)
        sdp = np.zeros(self.maxSentenceLen, dtype=np.float32)
        tokenidxs = np.zeros(self.maxSentenceLen)
        positionValues1 = np.zeros(self.maxSentenceLen)
        positionValues2 = np.zeros(self.maxSentenceLen)
        tokens = str(sentence).split(" ")
        words = tokens.copy()
        flags = []
        slen = len(tokens)
        for idx in range(0,slen):
            sdp[idx] = 0.3
            distance1 = idx - int(pos1)
            distance2 = idx - int(pos2)
            if distance1 in self.distanceMapping:
                positionValues1[idx] = self.distanceMapping[distance1]
            elif distance1 <= self.minDistance:
                positionValues1[idx] = self.distanceMapping['LowerMin']
            else:
                positionValues1[idx] = self.distanceMapping['GreaterMax']

            if distance2 in self.distanceMapping:
                positionValues2[idx] = self.distanceMapping[distance2]
            elif distance2 <= self.minDistance:
                positionValues2[idx] = self.distanceMapping['LowerMin']
            else:
                positionValues2[idx] = self.distanceMapping['GreaterMax']

            if idx == pos1 or idx == pos2:
                flags.append("kej")
            else:
                flags.append(pseg.lcut(tokens[idx])[0].flag)

            if not self.vocab.__contains__(tokens[idx]):
                temp = jieba.lcut(tokens[idx])
                tokens[idx] = temp[len(temp) - 1]
                if not self.vocab.__contains__(tokens[idx]):
                    # print(str(idx) + " " + str(tokens))
                    # print(tokens[idx])
                    tokens[idx] = 'PADDING_WORD'
            tokenidxs[idx] = self.word2idx[tokens[idx]]
            # print(tokens[idx] + ":" + str(tokenidxs[idx]))
        # print(tokenidxs)
        arcs = self.parser.parse(words, flags)  # 句法分析
        # print("\t".join("%d:%s" % (arc.head, arc.relation) for arc in arcs))
        # for i in range(len(words)):
        #     print(str(i + 1) + " " + words[i] + " " + flags[i] + " " + str(arcs[i].head) + ":" + arcs[i].relation)
        iter_idx = pos1
        while True:
            if arcs[iter_idx].relation != "HED":
                sdp[iter_idx] = 0.8
                iter_idx = (arcs[iter_idx].head - 1)
            else:
                sdp[iter_idx] = 0.8
                break
        iter_idx = pos2
        while True:
            if arcs[iter_idx].relation != "HED":
                sdp[iter_idx] = 0.8
                iter_idx = (arcs[iter_idx].head - 1)
            else:
                sdp[iter_idx] = 0.8
                break

        # for i in range(len(words)):
        #     print(str(i + 1) + " " + words[i] + " " + flags[i] + " " + str(arcs[i].head) + ":" + arcs[i].relation + " " + str(sdp[i]))
        return tokenidxs,positionValues1,positionValues2,sdp

    def kejline_w2v(self,pos1,pos2,tokens):
        # print(sentence)
        sdp = np.zeros(self.maxSentenceLen, dtype=np.float32)
        tokenidxs = np.zeros(self.maxSentenceLen)
        positionValues1 = np.zeros(self.maxSentenceLen)
        positionValues2 = np.zeros(self.maxSentenceLen)
        words = tokens.copy()
        flags = []
        slen = len(tokens)
        for idx in range(0,slen):
            sdp[idx] = 0.3
            distance1 = str(idx - pos1)
            distance2 = str(idx - pos2)
            if distance1 in self.distanceMapping:
                positionValues1[idx] = self.distanceMapping[distance1]
            elif int(distance1) <= self.minDistance:
                positionValues1[idx] = self.distanceMapping['LowerMin']
            else:
                positionValues1[idx] = self.distanceMapping['GreaterMax']

            if distance2 in self.distanceMapping:
                positionValues2[idx] = self.distanceMapping[distance2]
            elif int(distance2) <= self.minDistance:
                positionValues2[idx] = self.distanceMapping['LowerMin']
            else:
                positionValues2[idx] = self.distanceMapping['GreaterMax']

            if idx == pos1 or idx == pos2:
                flags.append("kej")
            else:
                flags.append(pseg.lcut(tokens[idx])[0].flag)

            if not self.vocab.__contains__(tokens[idx]):
                temp = jieba.lcut(tokens[idx])
                tokens[idx] = temp[len(temp) - 1]
                if not self.vocab.__contains__(tokens[idx]):
                    # print(str(idx) + " " + str(tokens))
                    # print(tokens[idx])
                    tokens[idx] = 'PADDING_WORD'
            tokenidxs[idx] = self.word2idx[tokens[idx]]
            # print(tokens[idx] + ":" + str(tokenidxs[idx]))
        # print(tokenidxs)
        arcs = self.parser.parse(words, flags)  # 句法分析
        # print("\t".join("%d:%s" % (arc.head, arc.relation) for arc in arcs))
        # for i in range(len(words)):
        #     print(str(i + 1) + " " + words[i] + " " + flags[i] + " " + str(arcs[i].head) + ":" + arcs[i].relation)
        iter_idx = pos1
        while True:
            if arcs[iter_idx].relation != "HED":
                sdp[iter_idx] = 0.8
                iter_idx = (arcs[iter_idx].head - 1)
            else:
                sdp[iter_idx] = 0.8
                break
        iter_idx = pos2
        while True:
            if arcs[iter_idx].relation != "HED":
                sdp[iter_idx] = 0.8
                iter_idx = (arcs[iter_idx].head - 1)
            else:
                sdp[iter_idx] = 0.8
                break

        # for i in range(len(words)):
        #     print(str(i + 1) + " " + words[i] + " " + flags[i] + " " + str(arcs[i].head) + ":" + arcs[i].relation + " " + str(sdp[i]))
        return tokenidxs,positionValues1,positionValues2,sdp


    def process_one_input(self,input):
        temps = str(input).split("\t")
        relation = temps[0]
        pos1 = temps[1]
        pos2 = temps[2]
        sentence = temps[3].strip()
        tokenidxs,positionValues1,positionValues2,sdp = self.sentence_w2v(pos1,pos2,sentence)
        return self.relationsMapping[relation],tokenidxs,positionValues1,positionValues2,sdp

    def process_file(self,file,saveFlag = True,savepath = '../pkl/sem-relations.pkl.gz'):
        relationidxs = []
        positionMatrix1 = []
        positionMatrix2 = []
        tokenMatrix = []
        sdpMatrix = []
        with codecs.open(file,"r","utf8") as rd:
            lines = rd.readlines()
            # self.maxSentenceLen = self.get_max_sentence_len(lines)
            for line in lines:
                #检查长度
                if len(line.split("\t")[3].split(" "))>self.maxSentenceLen:
                    print("超过长度")
                    continue
                relationidx, tokenidxs, positionValues1, positionValues2,sdp = self.process_one_input(line)
                relationidxs.append(relationidx)
                positionMatrix1.append(positionValues1)
                positionMatrix2.append(positionValues2)
                tokenMatrix.append(tokenidxs)
                sdpMatrix.append(sdp)
        relationidxs = np.asarray(relationidxs, dtype='int32')
        positionMatrix1 = np.asarray(positionMatrix1, dtype='int32')
        positionMatrix2 = np.asarray(positionMatrix2, dtype='int32')
        tokenMatrix = np.asarray(tokenMatrix, dtype='int32')
        sdpMatrix = np.asarray(sdpMatrix, dtype='float32')
        if saveFlag:
            self.save_pkl(relationidxs, positionMatrix1, positionMatrix2, tokenMatrix, sdpMatrix, savepath)
        return relationidxs,positionMatrix1,positionMatrix2,tokenMatrix, sdpMatrix

    def save_pkl(self,relationidxs, positionMatrix1, positionMatrix2, tokenMatrix,sdpMatrix,save_path):
        data = {'relationidxs': relationidxs, 'positionMatrix1': positionMatrix1,
                'positionMatrix2': positionMatrix2,'tokenMatrix': tokenMatrix,
                "sdpMatrix": sdpMatrix}
        f = gzip.open(save_path, 'wb')
        pkl.dump(data, f)
        f.close()


    def process_one(self,line):
        # self.maxSentenceLen = 78
        if len(line.split("\t")[3].split(" ")) > self.maxSentenceLen:
            print("超过长度")
            return None
        relationidx, tokenidxs, positionValues1, positionValues2 = self.process_one_input(line)
        relationidx = np.asarray(relationidx, dtype='int32')
        positionMatrix1 = np.asarray(positionValues1, dtype='int32')
        positionMatrix2 = np.asarray(positionValues2, dtype='int32')
        tokenMatrix = np.asarray(tokenidxs, dtype='int32')
        tokenMatrix = tokenMatrix.reshape((1, self.maxSentenceLen))
        positionMatrix1 = positionMatrix1.reshape((1, self.maxSentenceLen))
        positionMatrix2 = positionMatrix2.reshape((1, self.maxSentenceLen))
        return relationidx, positionMatrix1, positionMatrix2, tokenMatrix

# pre = PreTrain()
# pre.process_file("files/test.txt",True,'pkl/test.pkl.gz')
# pre.sentence_w2v(2,4,"入宫 为 魏孝文帝 和 文明太后 治过 病 ， 多有 疗效")