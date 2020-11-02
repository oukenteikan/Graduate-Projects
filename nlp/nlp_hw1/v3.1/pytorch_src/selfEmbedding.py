import torch
import torch.nn as nn
import random
from torch import optim
import torch.nn.functional as F


import re
import unicodedata
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

SOS_token = 0
EOS_token = 1

class Lang:
    def __init__(self):
        self.word2index = {}
        self.word2count = {}
        self.index2word = ["SOS","EOS"]
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word.append(word)
            self.n_words += 1
        else:
            self.word2count[word] += 1


# Turn a Unicode string to plain ASCII, thanks to
# https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" ", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s.strip()


def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def tensorFromIndexes(indexes):
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def prepareData(path):
    dataframe = pd.read_csv(path)
    storyStringCorpus = []
    for storyString in dataframe.values.tolist():
        normalizedStoryString = [storyString[0]]
        for line in storyString[1:]:
            normalizedStoryString.append(normalizeString(line))
        storyStringCorpus.append(normalizedStoryString)

    lang = Lang()
    for storyString in storyStringCorpus:
        for line in storyString[1:]:
            lang.addSentence(line)

    storyIndexesCorpus = []
    for storyString in storyStringCorpus:
        storyIndexes = [storyString[0]]
        for line in storyString[1:]:
            storyIndexes.append(indexesFromSentence(lang, line))
        storyIndexesCorpus.append(storyIndexes)

    return (lang, storyIndexesCorpus)


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def getEmbedding(self, input):
        return self.embedding(input);

    def getSentenceEmbedding(self, input, hidden): #input example [1,2,3,4,5]
        input_lenth = len(input)
        output = self.embedding(torch.tensor(input, dtype=torch.long, device=device)).view(input_lenth,1,self.hidden_size)
        output, hidden = self.gru(output, hidden)
        return hidden.view(-1)

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class DecoderRNN(nn.Module):
    def __init__(self, output_size, hidden_size):
        super(DecoderRNN, self).__init__()
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)


    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output, hidden = self.gru(embedded, hidden)
        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden

    def getTopOneWord(self, input):
        topv, topi = self.out(input.view(1,-1)).topk(1)
        return topi.squeeze().detach()


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):
    encoder_hidden = encoder.initHidden()
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)
    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)

    decoder_input = torch.tensor([SOS_token], dtype=torch.long, device=device)
    decoder_hidden = encoder_hidden

    for di in range(target_length):
        decoder_output, decoder_hidden = decoder(
            decoder_input, decoder_hidden)
        topv, topi = decoder_output.topk(1)
        decoder_input = topi.squeeze().detach()  # detach from history as input
        loss += criterion(decoder_output, target_tensor[di])

    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()
    return loss.item() / target_length


import time
import math


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def trainIters(corpus ,encoder, decoder, n_iters, print_every=1000, plot_every=100, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    corpus_size = len(corpus)
    for iter in range(1, n_iters + 1):
        #print(iter)
        idx = (iter-1)%corpus_size
        if(idx==0):
            random.shuffle(corpus)
        training_story = corpus[idx][1:]

        input_list = []
        for i in range(1,6):
            # print("\t" + str(i))
            input_list = training_story[i]
            target_list = input_list
            input_tensor = tensorFromIndexes(input_list)
            target_tensor = tensorFromIndexes(target_list)

            loss = train(input_tensor, target_tensor, encoder,
                         decoder, encoder_optimizer, decoder_optimizer, criterion)
            print_loss_total += loss
            plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))
        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0
    showPlot(plot_losses)

def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)


def getSentenceEmbedding(encoder, decoder, corpus, lang):
    allSentenceEmbeddings = {}
    for idx,story in enumerate(corpus):
        #print(idx)
        sentenceEmbeddings = []
        for line in story[1:]:
            lineEmbedding = encoder.getSentenceEmbedding(line, encoder.initHidden())
            lineMostLikeWordIndex = decoder.getTopOneWord(lineEmbedding)
            lineMostLikeWord = lang.index2word[lineMostLikeWordIndex]
            sentenceEmbeddings.append(lineMostLikeWord)
        allSentenceEmbeddings[story[0]] = sentenceEmbeddings
    return  allSentenceEmbeddings


lang,storyIndexesCorpus = prepareData("storyData/stories.csv")
hidden_size = 256
encoder1 = EncoderRNN(lang.n_words, hidden_size).to(device)
decoder1 = DecoderRNN(lang.n_words, hidden_size).to(device)

trainIters(storyIndexesCorpus,encoder1, decoder1, 200000, print_every=500)

torch.save(encoder1, 'storyData/encoder.pkl')
torch.save(decoder1, 'storyData/decoder.pkl')

with open("storyData/sentenceEmbeddings", "wb") as file:
    pickle.dump(getSentenceEmbedding(encoder1,decoder1,storyIndexesCorpus,lang), file)

with open("storyData/Index2Word", "wb") as file:
    pickle.dump(lang.index2word, file)

wordEmbeddings = torch.tensor([i for i in range(lang.n_words)], dtype=torch.long, device=device)
wordEmbeddings = encoder1.getEmbedding(wordEmbeddings).tolist()

with open("storyData/wordEmbeddings", "wb") as file:
    pickle.dump(wordEmbeddings, file)
