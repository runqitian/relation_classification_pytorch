import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

import numpy as np
from collections import OrderedDict

from data_tools import DataTools

import re
import jieba
import jieba.posseg as pseg

from preprocess import PreTrain

import codecs
import gzip
import pickle as pkl

pt_ent = re.compile("<Entity>(.*?)</Entity>")

MAX_SENTENCE_LEN = 100

POS_DIM = 125
POS_EMBD_LEN = 204

WORD_DIM = 250

FINAL_DIM = 660

CLASSES_NUM = 15
CLASS_DIM = 20

learning_rate = 0.001
multiple = 0.01


# relation2idx = {
#     "locaA": 0, "locAa": 1,
#     "med-ill": 2, "ill-med": 3,
#     "clsaA": 4, "clsAa": 5,
#     "w-c": 6, "c-w": 7,
#     "cs-ef": 8, "ef-cs": 9,
#     "pfs-name": 10,"name-pfs": 11,
#     "way-obj": 12,"obj-way": 13,
#     "org-ent": 14, "ent-org": 15,
#     "pdc-pdt": 16,"pdt-pdc": 17,
#     "user-tool": 18,"tool-user": 19,
#     "same": 20,
#     'other': 21
# }
#
# idx2relation = {
#     0:"locaA", 1:"locAa",
#     2:"med-ill", 3:"ill-med",
#     4:"clsaA", 5:"clsAa",
#     6:"w-c", 7:"c-w",
#     8:"cs-ef", 9:"ef-cs",
#     10:"pfs-name",11:"name-pfs",
#     12:"way-obj", 13:"obj-way",
#     14:"org-ent", 15:"ent-org",
#     16:"pdc-pdt", 17:"pdt-pdc",
#     18:"user-tool", 19:"tool-user",
#     20:"same",
#     21:"other"
# }

relation2idx = {
    "Position": 0, "Position1": 1,
    "Cause": 2, "Cause1": 3,
    "Medicine": 4, "Medicine1": 5,
    "Identity": 6, "Identity1": 7,
    "From": 8, "From1": 9,
    "Part": 10,"Part1": 11,
    "Describe": 12,"Describe1": 13,
    'Other': 14
}

idx2relation = {
    0: "Position0", 1: "Position1",
    2: "Cause0", 3: "Cause1",
    4: "Medicine0", 5: "Medicine1",
    6: "Identity0", 7: "Identity1",
    8: "From0", 9: "From1",
    10: "Part0", 11: "Part1",
    12: "Describe0", 13: "Describe1",
    14: 'Other'
}

class TrainDataset(Dataset):
    """
        训练数据集
    """
    def __init__(self):
        dt = DataTools()
        self.ys, self.tokenMatrix, self.positionMatrix1, self.positionMatrix2, self.sdpMatrix = dt.get_data("./pkl/final_train.pkl.gz")
        print(self.ys.shape[0])
        # print(self.sdpMatrix)
        # print(self.sdpMatrix[0,:])
    def __getitem__(self, item):
        return self.ys[item], torch.LongTensor(self.tokenMatrix[item,:]),\
               torch.LongTensor(self.positionMatrix1[item,:]),torch.LongTensor(self.positionMatrix2[item,:]),self.sdpMatrix[item,:]
    def __len__(self):
        return self.ys.shape[0]


class TestDataset(Dataset):
    """
            测试数据集
        """
    def __init__(self):
        dt = DataTools()
        self.ys, self.tokenMatrix, self.positionMatrix1, self.positionMatrix2, self.sdpMatrix = dt.get_data("./pkl/final_test.pkl.gz")
        print(self.ys.shape[0])
        # print(self.sdpMatrix)
        # print(self.sdpMatrix[0,:])
    def __getitem__(self, item):
        return self.ys[item], torch.LongTensor(self.tokenMatrix[item,:]),\
               torch.LongTensor(self.positionMatrix1[item,:]),torch.LongTensor(self.positionMatrix2[item,:]),self.sdpMatrix[item,:]
    def __len__(self):
        return self.ys.shape[0]


class Net(nn.Module):
    """
        pytorch CNN模型
    """
    def __init__(self):
        filter_num = 100

        super(Net, self).__init__()
        print("初始化Net模型")
        wordembedding = np.load("embeddings/wordembedding.npy")
        self.word_embeds = nn.Embedding(wordembedding.shape[0], WORD_DIM)
        pretrained_weight = np.array(wordembedding)
        self.word_embeds.weight.data.copy_(torch.from_numpy(pretrained_weight))

        self.pos1_embeds = nn.Embedding(POS_EMBD_LEN, POS_DIM)

        self.pos2_embeds = nn.Embedding(POS_EMBD_LEN, POS_DIM)

        #   sdp
        # self.U = nn.Parameter(torch.randn(WORD_DIM,CLASS_DIM),requires_grad=True)
        # self.class_matrix = nn.Parameter(torch.randn(CLASS_DIM,CLASSES_NUM))
        # self.M = nn.Parameter(torch.randn(MAX_SENTENCE_LEN,CLASSES_NUM))

        self.conv1_1 = torch.nn.Sequential(
            torch.nn.Conv2d(250, filter_num, kernel_size=(3, 1), padding=(0, 0)),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d((1, 2)))

        self.conv1_3 = torch.nn.Sequential(
            torch.nn.Conv2d(250, filter_num, kernel_size=(3, 3), padding=(0, 1)),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d((1, 2)))

        self.conv1_5 = torch.nn.Sequential(
            torch.nn.Conv2d(250, filter_num, kernel_size=(3, 5), padding=(0, 2)),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d((1, 2)))
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(300, 200, kernel_size=(1, 1), padding=(0, 0)),
            torch.nn.Conv2d(200, 300, kernel_size=(1, 3), padding=(0, 1)),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d((1, 50)))
        # self.conv3 = torch.nn.Sequential(
        #     torch.nn.Conv2d(300, 400, kernel_size=(1, 5), padding=(0, 2)),
        #     torch.nn.ReLU(),
        #     torch.nn.MaxPool2d((1, 96),stride=1))

        self.dense1 = torch.nn.Sequential(
            torch.nn.Dropout(0.25),
            torch.nn.Linear(300,100),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.25),
            torch.nn.Linear(100,50),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.25),
            torch.nn.Linear(50, CLASSES_NUM)
        )




    def forward(self, tokenMatrix, pos1Matrix, pos2Matrix,sdpMatrix):
        word_input = self.word_embeds(tokenMatrix)
        pos1_input = self.pos1_embeds(pos1Matrix)
        pos2_input = self.pos2_embeds(pos2Matrix)

        # sdp 生成
        # G = torch.matmul(word_input.view(-1,WORD_DIM), self.U)
        # G = torch.matmul(G,self.class_matrix)
        # G = G.view(-1,MAX_SENTENCE_LEN, CLASSES_NUM)
        # alpha = torch.matmul(sdpMatrix,self.M)
        # alpha = alpha.unsqueeze(1)
        # alpha = torch.matmul(alpha, G.transpose(1,2))
        # alpha = torch.add(sdpMatrix,multiple,torch.squeeze(alpha,1))

        alpha = sdpMatrix
        alpha_dn = None
        for i in range(alpha.shape[0]):
            dni = torch.diag(alpha[i, :])
            dni = torch.unsqueeze(dni, 0)
            if i == 0:
                alpha_dn = dni
            else:
                alpha_dn = torch.cat([alpha_dn, dni], 0)
        weighted_data = torch.matmul(alpha_dn,word_input)
        pos_input = torch.cat((pos1_input,pos2_input),dim=2)
        pos_input = torch.unsqueeze(pos_input,2)
        word_input = torch.unsqueeze(word_input,2)
        weighted_data = torch.unsqueeze(weighted_data,2)
        final_input = torch.cat((word_input,pos_input,weighted_data),dim=2)
        # final_input = torch.cat((word_input, pos1_input, pos2_input,weighted_data), dim=2)
        final_input_trans = torch.transpose(final_input, 1, 3)
        # print(final_input_trans.shape)

        conv1_1_out = self.conv1_1(final_input_trans)
        conv1_3_out = self.conv1_3(final_input_trans)
        conv1_5_out = self.conv1_5(final_input_trans)
        conv1_out = torch.cat((conv1_1_out,conv1_3_out,conv1_5_out),dim=1)
        # print(conv1_out.shape)
        conv2_out = self.conv2(conv1_out)
        # print(conv2_out.shape)
        # conv3_out = self.conv3(conv2_out)
        # print(conv3_out.shape)
        # print(conv1_out.shape)
        conv2_out_sq = torch.squeeze(conv2_out,2)
        conv2_out_sq = torch.squeeze(conv2_out_sq, 2)
        # print(conv1_out_sq.shape)
        output = self.dense1(conv2_out_sq)

        return output
class Entrance(object):
    def __init__(self,load):
        if load:
            self.model = torch.load("pkl/kejmodel_att_h_23.torch")
        else:
            self.model = Net()

        print(self.model)
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam([{"params": self.model.pos1_embeds.parameters()},
                                {"params": self.model.pos2_embeds.parameters()},
                                {"params": self.model.conv1_1.parameters()},
                                {"params": self.model.conv1_3.parameters()},
                                {"params": self.model.conv1_5.parameters()},
                                {"params": self.model.conv2.parameters()},
                                # {"params": model.conv3.parameters()},
                                {"params": self.model.dense1.parameters()}
                                # {"params": model.U},
                                # {"params": model.class_matrix},
                                # {"params": model.M}
                                ],lr = learning_rate)
        self.train_dataset = TrainDataset()
        self.test_dataset = TestDataset()
        self.pre = PreTrain()

    def train(self, epoch, data_loader):
        self.model.train()
        train_acc = 0
        train_loss = 0
        for batch_idx, (ys, tokenMatrix, pos1Matrix, pos2Matrix, sdpMatrix) in enumerate(data_loader):
            ys = Variable(ys.long())
            tokenMatrix= Variable(torch.LongTensor(tokenMatrix))
            pos1Matrix = Variable(torch.LongTensor(pos1Matrix))
            pos2Matrix=  Variable(torch.LongTensor(pos2Matrix))
            sdpMatrix =  Variable(sdpMatrix)
            output = self.model(tokenMatrix, pos1Matrix, pos2Matrix,sdpMatrix)
            loss = self.loss_function(output,ys)
            train_loss += loss.data[0]
            pred = torch.max(output, 1)[1]
            train_correct = (pred == ys).sum()
            train_acc += train_correct.data[0]
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        print('Train Loss: {:.6f}, Acc: {:.6f}'.format(train_loss / (len(
            self.train_dataset)), train_acc / (len(self.train_dataset))))

    def test(self, data_loader):
        self.model.eval()
        eval_loss = 0.
        eval_acc = 0.
        for batch_idx, (ys, tokenMatrix, pos1Matrix, pos2Matrix, sdpMatrix) in enumerate(data_loader):
            ys = Variable(ys.long())
            tokenMatrix = Variable(torch.LongTensor(tokenMatrix))
            pos1Matrix = Variable(torch.LongTensor(pos1Matrix))
            pos2Matrix = Variable(torch.LongTensor(pos2Matrix))
            sdpMatrix = Variable(sdpMatrix)
            output = self.model(tokenMatrix, pos1Matrix, pos2Matrix,sdpMatrix)
            loss = self.loss_function(output, ys)
            eval_loss += loss.data[0]
            pred = torch.max(output, 1)[1]
            num_correct = (pred ==ys).sum()
            eval_acc += num_correct.data[0]
        print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(
            self.test_dataset)), eval_acc / (len(self.test_dataset))))

    def test_train(self, data_loader):
        self.model.eval()
        eval_loss = 0.
        eval_acc = 0.
        for batch_idx, (ys, tokenMatrix, pos1Matrix, pos2Matrix, sdpMatrix) in enumerate(data_loader):
            ys = Variable(ys.long())
            tokenMatrix = Variable(torch.LongTensor(tokenMatrix))
            pos1Matrix = Variable(torch.LongTensor(pos1Matrix))
            pos2Matrix = Variable(torch.LongTensor(pos2Matrix))
            sdpMatrix = Variable(sdpMatrix)
            output = self.model(tokenMatrix, pos1Matrix, pos2Matrix,sdpMatrix)
            loss = self.loss_function(output, ys)
            eval_loss += loss.data[0]
            pred = torch.max(output, 1)[1]
            num_correct = (pred ==ys).sum()
            eval_acc += num_correct.data[0]
        print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(
            self.train_dataset)), eval_acc / (len(self.train_dataset))))

    def entrance(self):
        train_loader = DataLoader(dataset=self.train_dataset,
                                                 batch_size = 64,
                                                 shuffle=True)
        test_loader = DataLoader(dataset=self.test_dataset,
                                                 batch_size = 64,
                                                 shuffle=False)

        # for i in range(5):
        #     print("epoch: " + str(i))
        #     self.train(i, train_loader)
        #     self.test(test_loader)
        self.test_train(train_loader)
        self.test(test_loader)

        torch.save(self.model,"pkl/kejmodel_att_h_23.torch")

    def freq_tag(self, word):
        freq = jieba.get_FREQ(word)
        tag = ""
        if freq is not None:
            tag = pseg.lcut(word,HMM=False)[0].flag
        return freq,tag

    def recover_dict(self, word,freq,tag):
        if freq is None:
            jieba.del_word(word)
        else:
            jieba.add_word(word, freq=freq, tag=tag)

    def get_entity(self, line):
        line = str(line)
        temps = re.findall(pt_ent, line)
        entity1 = temps[0]
        entity2 = temps[1]
        return entity1, entity2

    def test_direct(self, line):
        linetemp = line
        line = str(line)
        entity1, entity2 = self.get_entity(line)
        line = line.replace("<entity>", "")
        line = line.replace("</entity>", "")
        freq1, tag1 = self.freq_tag(entity1)
        jieba.add_word(entity1, freq=1000000, tag="kej")
        freq2, tag2 = self.freq_tag(entity2)
        jieba.add_word(entity2, freq=1000000, tag="kej")
        words = jieba.lcut(line)
        # print(words)
        self.recover_dict(entity1, freq1, tag1)
        self.recover_dict(entity2, freq2, tag2)
        seq1 = []
        seq2 = []
        for i in range(len(words)):
            if words[i] == entity1:
                seq1.append(i)
            if words[i] == entity2:
                seq2.append(i)
        tokenidxs, positionValues1, positionValues2, sdp = self.pre.kejline_w2v(seq1[0],seq2[0],words)
        self.model.eval()
        tokenMatrix = Variable(torch.LongTensor(tokenidxs))
        pos1Matrix = Variable(torch.LongTensor(positionValues1))
        pos2Matrix = Variable(torch.LongTensor(positionValues2))
        sdpMatrix = Variable(torch.FloatTensor(sdp))
        tokenMatrix = torch.unsqueeze(tokenMatrix,0)
        pos1Matrix = torch.unsqueeze(pos1Matrix,0)
        pos2Matrix = torch.unsqueeze(pos2Matrix,0)
        sdpMatrix = torch.unsqueeze(sdpMatrix,0)
        output = self.model(tokenMatrix, pos1Matrix, pos2Matrix, sdpMatrix)
        pred = torch.max(output, 1)[1]
        relation = idx2relation[int(pred)]
        return relation, relation + "\t" + linetemp

    def save_pkl(self,relationidxs, positionMatrix1, positionMatrix2, tokenMatrix,sdpMatrix,save_path):
        data = {'relationidxs': relationidxs, 'positionMatrix1': positionMatrix1,
                'positionMatrix2': positionMatrix2,'tokenMatrix': tokenMatrix,
                "sdpMatrix": sdpMatrix}
        f = gzip.open(save_path, 'wb')
        pkl.dump(data, f)
        f.close()

    def train_direct(self, file, saveFlag):
        relationidxs = []
        positionMatrix1 = []
        positionMatrix2 = []
        tokenMatrix = []
        sdpMatrix = []
        with codecs.open(file, "r") as rf:
            lines = rf.readlines()
            print(len(lines))
            line_i = 0
            for inputline in lines:
                print(line_i)
                line_i+=1
                two = inputline.split("\t")
                relation = two[0]
                relationidx = relation2idx[relation]
                line = two[1]
                line = line.replace("<NR>", "")
                line = line.replace("<Position>", "")
                linetemp = line
                line = str(line)
                entity1, entity2 = self.get_entity(line)
                line = line.replace("<Entity>", "")
                line = line.replace("</Entity>", "")

                freq1, tag1 = self.freq_tag(entity1)
                jieba.add_word(entity1, freq=1000000, tag="kej")
                freq2, tag2 = self.freq_tag(entity2)
                jieba.add_word(entity2, freq=1000000, tag="kej")
                words = jieba.lcut(line)
                if len(words)>100:
                    continue
                # print(words)
                self.recover_dict(entity1, freq1, tag1)
                self.recover_dict(entity2, freq2, tag2)
                seq1 = []
                seq2 = []
                for i in range(len(words)):
                    if words[i] == entity1:
                        seq1.append(i)
                    if words[i] == entity2:
                        seq2.append(i)
                if len(seq1) == 0  or len(seq2) == 0:
                    continue
                tokenidxs, positionValues1, positionValues2, sdp = self.pre.kejline_w2v(seq1[0],seq2[0],words)
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
        print(relationidxs.shape[0])
        if saveFlag:
            self.save_pkl(relationidxs, positionMatrix1, positionMatrix2, tokenMatrix, sdpMatrix, "kej_test.pkl.gz")
        return relationidxs,positionMatrix1,positionMatrix2,tokenMatrix, sdpMatrix

sentence = "Medicine	结论:<Entity>特比萘芬软膏</Entity>治疗浅部真菌病较<Entity>咪康唑软膏</Entity>疗程短、疗效好"
entrance = Entrance(True)
entrance.entrance()
# entrance.train_direct("./keywords/examples_2000.txt",True)

# outputs = []
# with codecs.open("keywords/test2.txt","r") as rf:由<entity>海平面上升</entity>引起的<entity>地下水位上升</entity>会使
#     lines = rf.readlines()
#     print(len(lines))
#     i = 0
#     for line in lines:
#         if i%100==0:
#             print(i/100)
#         i+=1
#         line = line.strip()
#         relation, output = entrance.test_direct(line)
#         if relation!="other":
#             outputs.append(output)
#             print(output)
# with codecs.open("./迭代结果.txt","w") as wf:
#     wf.writelines(outputs)