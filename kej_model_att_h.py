import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

import numpy as np
from collections import OrderedDict

from data_tools import DataTools

MAX_SENTENCE_LEN = 100

POS_DIM = 125
POS_EMBD_LEN = 204

WORD_DIM = 250

FINAL_DIM = 660

CLASSES_NUM = 11
CLASS_DIM = 11

learning_rate = 0.0001
multiple = 0.01

class TrainDataset(Dataset):
    """
        训练数据集
    """
    def __init__(self):
        dt = DataTools()
        self.ys, self.tokenMatrix, self.positionMatrix1, self.positionMatrix2, self.sdpMatrix = dt.get_data("pkl/train.pkl.gz")
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
        self.ys, self.tokenMatrix, self.positionMatrix1, self.positionMatrix2, self.sdpMatrix = dt.get_data("pkl/test.pkl.gz")
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
        wordembedding = np.load("w2vmodel/wordembedding.npy")
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
            torch.nn.Linear(50, 11)
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

# model = Net()
model = torch.load("pkl/kejmodel_att_h.torch")
print(model)


loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam([{"params": model.pos1_embeds.parameters()},
                        {"params": model.pos2_embeds.parameters()},
                        {"params": model.conv1_1.parameters()},
                        {"params": model.conv1_3.parameters()},
                        {"params": model.conv1_5.parameters()},
                        {"params": model.conv2.parameters()},
                        # {"params": model.conv3.parameters()},
                        {"params": model.dense1.parameters()}
                        # {"params": model.U},
                        # {"params": model.class_matrix},
                        # {"params": model.M}
                        ],lr = learning_rate)

def train(epoch,data_loader):
    model.train()
    train_acc = 0
    train_loss = 0
    for batch_idx, (ys, tokenMatrix, pos1Matrix, pos2Matrix, sdpMatrix) in enumerate(data_loader):
        ys = Variable(ys.long())
        tokenMatrix= Variable(torch.LongTensor(tokenMatrix))
        pos1Matrix = Variable(torch.LongTensor(pos1Matrix))
        pos2Matrix=  Variable(torch.LongTensor(pos2Matrix))
        sdpMatrix =  Variable(sdpMatrix)
        output = model(tokenMatrix, pos1Matrix, pos2Matrix,sdpMatrix)
        loss = loss_function(output,ys)
        train_loss += loss.data[0]
        pred = torch.max(output, 1)[1]
        train_correct = (pred == ys).sum()
        train_acc += train_correct.data[0]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('Train Loss: {:.6f}, Acc: {:.6f}'.format(train_loss / (len(
        train_dataset)), train_acc / (len(train_dataset))))


def test(data_loader):
    model.eval()
    eval_loss = 0.
    eval_acc = 0.
    for batch_idx, (ys, tokenMatrix, pos1Matrix, pos2Matrix, sdpMatrix) in enumerate(data_loader):
        ys = Variable(ys.long())
        tokenMatrix = Variable(torch.LongTensor(tokenMatrix))
        pos1Matrix = Variable(torch.LongTensor(pos1Matrix))
        pos2Matrix = Variable(torch.LongTensor(pos2Matrix))
        sdpMatrix = Variable(sdpMatrix)
        output = model(tokenMatrix, pos1Matrix, pos2Matrix,sdpMatrix)
        loss = loss_function(output, ys)
        eval_loss += loss.data[0]
        pred = torch.max(output, 1)[1]
        num_correct = (pred ==ys).sum()
        eval_acc += num_correct.data[0]
    print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(
        test_dataset)), eval_acc / (len(test_dataset))))

train_dataset = TrainDataset()
test_dataset = TestDataset()
train_loader = DataLoader(dataset=train_dataset,
                                         batch_size = 64,
                                         shuffle=True)
test_loader = DataLoader(dataset=test_dataset,
                                         batch_size = 64,
                                         shuffle=False)

for i in range(5):
    print("epoch: " + str(i))
    train(i, train_loader)
    test(test_loader)

torch.save(model,"pkl/kejmodel_att_h.torch")