'''
This is the main script to train sequence models to detect fake news and it is in Pytorch.
It trains a model specified to use one of the sequence models viz RNN, GRU or LSTM and calculates the accuracy with respect to a 
validation set.
For more info run:

python trainer --help

Requires Pytorch version 0.4.0 and above.
'''

import torch
from model import loadData
from model import FakeNewsModelRNN as rnn
from model import FakeNewsModelGRU as gru
from model import FakeNewsModelLSTM as lstm
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import time

def str2bool(string):
    if string in 'True':
        print('running on GPU')
        return True
    elif string in 'False':
        print('running on CPU')
        return False

parser = argparse.ArgumentParser()
parser.add_argument('--mbsize',type=int,default=1,help='Specifies mini-batch size for training')
parser.add_argument('--epoch',type=int,default=10,help='Number of epochs to run')
parser.add_argument('--pe',type=int,default=1,help='Prints epoch loss after every pe number of epochs')
parser.add_argument('--numworkers',type=int,default=4,help='Number of workers to be spawned to load data by dataloader')
parser.add_argument('--lr',type=float,default=0.01,help='Learning rate')
parser.add_argument('--train',type=str,required=True,help='Path to training data')
parser.add_argument('--type',type=str,required=True,help='Type of sequence model to be used (rnn, lstm or gru)')
parser.add_argument('--validate',type=str,required=True,help='Path to validation data')
parser.add_argument('--modelName',type=str,default='mymodel.pth',help='Name under which trained model will be saved')
parser.add_argument('--cuda',type=str2bool,default='True',help='Set it to True to use GPU else False')
args = parser.parse_args()

elapsed = -time.time()

if args.cuda and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

data = loadData(args.train,args.validate)

if args.cuda:
    trainloader = DataLoader(data,batch_size=args.mbsize,shuffle=True,num_workers=args.numworkers,pin_memory=True)
else:
    trainloader = DataLoader(data,batch_size=args.mbsize,shuffle=True,num_workers=args.numworkers)

vocabSize = data.vocabularySize()
embeddingSize = 300
hiddenSize = 100
momentum = 0.9

if args.type in 'rnn':
    print('RNN model')
    model = rnn(vocabSize,embeddingSize,hiddenSize).to(device)
elif args.type in 'gru':
    print('GRU model')
    model = gru(vocabSize,embeddingSize,hiddenSize).to(device)
elif args.type in 'lstm':
    print('LSTM model')
    model = lstm(vocabSize,embeddingSize,hiddenSize).to(device)
else:
    print('Invalid entry for model type. Should be one of rnn, lstm or gru')
    assert False

criterion = nn.BCEWithLogitsLoss().to(device)
optimizer = optim.SGD(model.parameters(),lr=args.lr,momentum=momentum,nesterov=True)
#optimizer = optim.Adam(model.parameters(),lr=args.lr)

totalLoss  = 0
print('training started')

for epoch in range(1,args.epoch+1):
    for i,dat in enumerate(trainloader):
        news = dat[0].to(device)
        label = dat[1].to(device)
        optimizer.zero_grad()
        pred = model.forward(news,True)
        loss = criterion(pred,label)
        loss.backward()
#        nn.utils.clip_grad_value_(model.parameters(),25)
        optimizer.step()
        totalLoss += loss/news.size(1)
    if epoch % args.pe == 0:
        print('Epoch {} loss is {}'.format(epoch,totalLoss/(args.mbsize*args.pe)))
        totalLoss = 0

elapsed += time.time()
print('training ended. Time elapsed is {}'.format(elapsed))
torch.save({'model':model.state_dict(),'ix2wrd':data.ix2wrd,'wrd2ix':data.wrd2ix},args.modelName)
print('model saved')

data.isTrain=False

if args.cuda:
    model = model.to(torch.device('cpu'))

validateloader = DataLoader(data,batch_size=1,shuffle=True)

correct = 0
print('validation started')
for i,dat in enumerate(validateloader):
    news = dat[0]
    label = dat[1]
    pred = model.forward(news,False).item()
    if pred >= 0.5 :
        pred = 1
    else:
        pred = 0

    if pred == label:
        correct += 1
print('validation accuracy = {}'.format((correct/len(validateloader))*100))
