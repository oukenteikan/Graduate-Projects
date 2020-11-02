import os
import math
import argparse
import torch
from torch import optim
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_
from torch.nn import functional as F
from model import Encoder, Decoder, Seq2Seq
from utils import *
import random
import math
import logging

format_str = f'%(asctime)s %(filename)s [%(lineno)d] : %(message)s'
logging.basicConfig(format=format_str, datefmt="%d-%M-%Y %H:%M:%S", level=logging.DEBUG)
logger = logging.getLogger(__name__)

def translate(model, test_iter, vocab_size, lang):
    model.eval()
    pad = 0
    total_loss = 0
    for b, batch in enumerate(test_iter):
        src = batch["src"]
        trg = batch["trg"]
        src = Variable(src.data.cuda(), volatile=True)
        trg = Variable(trg.data.cuda(), volatile=True)
        print_tensor(lang,src.cpu())
        print_tensor(lang,trg.cpu())
        output = model(src, trg, teacher_forcing_ratio=0.0)
        loss = F.nll_loss(output[1:].view(-1, vocab_size),
                               trg[1:].contiguous().view(-1),
                               ignore_index=pad)
        total_loss += loss.item()
    return total_loss / len(val_iter)

logger.info("preparing dataset...")
lang, train_iter, val_iter, test_iter = load_data(1)
vocab_size = len(lang.index2word)

hidden_size = 512
embed_size = 512

encoder = Encoder(vocab_size, embed_size, hidden_size, n_layers=2, dropout=0.2)
decoder = Decoder(embed_size, hidden_size, vocab_size, n_layers=2, dropout=0.2)
seq2seq = Seq2Seq(encoder, decoder)
seq2seq.load_state_dict(torch.load("./save/seq2seq_{}.pt".format(24)))
seq2seq = seq2seq.cuda()
    
def printSortedIdx(lang,probs):
    probs = list(map(lambda x:math.exp(x), probs))
    total = sum(probs)
    sortedProbs = sorted(range(len(probs)), key=lambda k: probs[k], reverse=True)
    print(lang.indice2sentence(sortedProbs[:5]))

def getMaxProbIdx(lang,probs):
    printSortedIdx(lang,probs)
    probs = list(map(lambda x:math.exp(x), probs))
    total = sum(probs)
    print(total, max(probs), probs.index(max(probs)))
    choice = random.random() * total
    #print(choice)
    idx, cur = 0,0
    for i,prob in enumerate(probs):
        cur += prob
        if cur >= choice:
            idx = i
            break
    return idx

seq2seq.eval()
pad = 0
total_loss = 0
for b, batch in enumerate(test_iter):
    src = torch.from_numpy(batch["src"])
    trg = torch.from_numpy(batch["trg"])
    src = src.cuda()
    trg = src.cuda()
    src = Variable(src.data.cuda(), volatile=True)
    trg = Variable(trg.data.cuda(), volatile=True)
    outputs = seq2seq(src, trg, teacher_forcing_ratio=0.0)
    outputs = outputs.squeeze(1)
    #top1 = outputs.data.max(1)[1]
    outputs = outputs.data.detach().cpu().numpy().tolist()
    outputs = map(lambda probs: getMaxProbIdx(lang,probs),outputs[1:])
    sentence = lang.indice2sentence(outputs)
    #print(top1)
    print(sentence)

