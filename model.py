'''
This is a Pytorch implementation of fake news classifier based on sequence models.
It contains:
1. GRU, LSTM and RNN based models to classifying whether a news snippet is fake or not
2. Data loader class

Requires Pytorch version 0.4.0 and above.
'''
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class FakeNewsModelGRU(nn.Module):
    def __init__(self,inputSize,embeddingSize,hiddenSize):
        super(FakeNewsModelGRU,self).__init__()
        self.embedding = nn.Embedding(inputSize,embeddingSize)
        self.rnn = nn.GRU(embeddingSize,hiddenSize)
        self.linear = nn.Linear(hiddenSize,1)
        self.initWeights()
    def initWeights(self):
        nn.init.kaiming_uniform_(self.rnn.weight_ih_l0)
        nn.init.kaiming_uniform_(self.rnn.weight_hh_l0)
        nn.init.kaiming_uniform_(self.linear.weight)
    def forward(self,input,train):
        emb = self.embedding(input).permute(1,0,2)
        output,hidden = self.rnn(emb)
        if train:
            return self.linear(hidden.view(1,-1))
        else:
            return torch.sigmoid(self.linear(hidden.view(1,-1)))

class FakeNewsModelLSTM(nn.Module):
    def __init__(self,inputSize,embeddingSize,hiddenSize):
        super(FakeNewsModelLSTM,self).__init__()
        self.embedding = nn.Embedding(inputSize,embeddingSize)
        self.rnn = nn.LSTM(embeddingSize,hiddenSize)
        self.linear = nn.Linear(2*hiddenSize,1)
        self.initWeights()
    def initWeights(self):
        nn.init.kaiming_uniform_(self.rnn.weight_ih_l0)
        nn.init.kaiming_uniform_(self.rnn.weight_hh_l0)
        nn.init.kaiming_uniform_(self.linear.weight)
    def forward(self,input,train):
        emb = self.embedding(input).permute(1,0,2)
        output,hidden = self.rnn(emb)
        if train:
            return self.linear(torch.cat(hidden,dim=0).view(1,-1))
        else:
            return torch.sigmoid(self.linear(torch.cat(hidden,dim=0).view(1,-1)))

class FakeNewsModelRNN(nn.Module):
    def __init__(self,inputSize,embeddingSize,hiddenSize):
        super(FakeNewsModelRNN,self).__init__()
        self.embedding = nn.Embedding(inputSize,embeddingSize)
        self.rnn = nn.RNN(embeddingSize,hiddenSize)
        self.linear = nn.Linear(hiddenSize,1)
        self.initWeights()
    def initWeights(self):
        nn.init.kaiming_uniform_(self.rnn.weight_ih_l0)
        nn.init.kaiming_uniform_(self.rnn.weight_hh_l0)
        nn.init.kaiming_uniform_(self.linear.weight)
    def forward(self,input,train):
        emb = self.embedding(input).permute(1,0,2)
        output,hidden = self.rnn(emb)
        if train:
            return self.linear(hidden.view(1,-1))
        else:
            return torch.sigmoid(self.linear(hidden.view(1,-1)))

class loadData(Dataset):
    def __init__(self,train,validate):
        data = pd.read_csv(train)
        vals = {False:0,True:1}
        self.trainLabel = data[~data['Label'].isna()].Label.map(vals).values
        self.trainData = data[~data['Label'].isna()].Statement.values

        vals = {'FALSE':0,'TRUE':1}
        data = pd.read_csv(validate)
        self.validateLabel = data[~data['Label'].isna()].Label.map(vals).values
        self.validateData = data[~data['Label'].isna()].Statement.values

        vocab = []
        for i in self.trainData:
            vocab += i.lower().split()
        for i in self.validateData:
            vocab += i.lower().split()

        vocab = list(set(vocab))
        self.vocabSize = len(vocab)
        self.ix2wrd={i:word for i,word in enumerate(vocab)}
        self.wrd2ix={word:i for i,word in enumerate(vocab)}

        self.isTrain=True

    def vocabularySize(self):
        return self.vocabSize

    def __len__(self):
        if self.isTrain:
            return len(self.trainData)
        else:
            return len(self.validateData)

    def __getitem__(self,idx):
        if self.isTrain:
            news = torch.from_numpy(np.array([self.wrd2ix[word] for word in self.trainData[idx].lower().split()])).long()
            classification = torch.FloatTensor([self.trainLabel[idx]])
        else:
            news = torch.from_numpy(np.array([self.wrd2ix[word] for word in self.validateData[idx].lower().split()])).long()
            classification = torch.FloatTensor([self.validateLabel[idx]])

        return news,classification
