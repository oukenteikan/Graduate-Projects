###############################################################################
# Language Modeling on Penn Tree Bank
#
# This file generates new sentences sampled from the language model
#
###############################################################################

# -*- coding: utf-8 -*-

import argparse
import sys
import time
import numpy, math
import torch
import torch.nn as nn
from numbers import Number
from utils import batchify, get_batch, repackage_hidden
from ir_baseline import IRetriver

import data

parser = argparse.ArgumentParser(description='PyTorch Language Model')

# Model parameters.
#parser.add_argument('--train-data', type=str, default='data/penn/train.txt',
#                    help='location of the training data corpus. Used for the rescore_story function')
parser.add_argument('--vocab', type=str, default='./all_vocab.pkl',
                    help='path to a pickle of the vocab used in training the model')
parser.add_argument('--keywords', type=str, default='',
                    help='location of the file for validation keywords')
parser.add_argument('--conditional-data', type=str, default='/home/charlie/nlp/output/roc_key.test',
                    help='location of the file that contains the content that the generation conditions on')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (LSTM, QRNN)')
parser.add_argument('--checkpoint', type=str, default='/home/charlie/nlp/output/stage2.pt',
                    help='model checkpoint to use')
parser.add_argument('--task', type=str, default='cond_generate',
                    choices=['generate', 'cond_generate', 'shannon_game', 'rescore_ending', 'rescore_story', 'scoring'],
                    help='specify the generation task')
parser.add_argument('--outf', type=str, default='/home/charlie/nlp/output/roc.test',
                    help='output file for generated text')
parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                    help='batch size')
parser.add_argument('--sents', type=int, default='10000',
                    help='number of sentences to generate')
parser.add_argument('--words', type=int, default='1000',
                    help='number of words to generate')
parser.add_argument('--story-length', type=int, default='5',
                    help='length of story to generate')
parser.add_argument('--storyline-length', type=int, default='5',
                    help='length of storyline to generate')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true', default=True,
                    help='use CUDA')
parser.add_argument('--dedup', action='store_true',
                    help='de-duplication')
parser.add_argument('--print-cond-data', action='store_true', default=True,
                    help='whether to print the prompt on which conditionally generated text is conditioned')
parser.add_argument('--temperature', type=float, default=0.5,
                    help='temperature - higher will increase diversity')
parser.add_argument('--log-interval', type=int, default=100,
                    help='reporting interval')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

if args.temperature < 1e-3:
    parser.error("--temperature has to be greater or equal 1e-3")

with open(args.checkpoint, 'rb') as f:
    model, criterion, optimizer = torch.load(f, map_location=lambda storage, loc: storage)
model.eval()
if args.model == 'QRNN': model.reset()
if args.cuda: model.cuda()

corpus = data.Corpus(applyDict=True, vocab_path=args.vocab)
ntokens = len(corpus.dictionary)
print('ntokens: ', ntokens)
eot_id = corpus.dictionary.word2idx['<EOT>']
eol_id = corpus.dictionary.word2idx['<EOL>']
eos_id = corpus.dictionary.word2idx['<eos>']
delimiter_id = corpus.dictionary.word2idx['#']
if not args.dedup:
    start_id = corpus.dictionary.word2idx['</s>']
    period_id = corpus.dictionary.word2idx['.']
print('eot id: {} | eol id: {} | eos id: {} | delimiter id: {}'.format(eot_id, eol_id, eos_id, delimiter_id))

test_batch_size, story_length, storyline_length = args.batch_size, args.story_length, args.storyline_length
hidden = model.init_hidden(test_batch_size)
input = torch.rand(1, 1).mul(ntokens).long()
if args.cuda: input.data = input.data.cuda()

with torch.no_grad():
    with open(args.outf, 'w') as outf:
        if args.task == 'cond_generate':
            data = corpus.tokenize(args.conditional_data, applyDict=True).tolist()
            nsent = 0
            while nsent < args.sents:
                try:
                    idx = data.index(eos_id)  # the only thing that breaks the while loop is if there are no more eos_ids
                except:
                    break
                word_set = set()
                period_count = 0
                for i in range(args.words):
                    if i < idx:
                        word_idx = data[i]
                    else:
                        output = model.decoder(output)
                        assert output.shape[1] == len(corpus.dictionary)
                        word_weights = output.squeeze().data.div(args.temperature).exp().cpu()
                        last_word_idx = word_idx
                        if args.dedup:
                            if len(word_set) == storyline_length:
                                word_idx = eol_id
                                if last_word_idx == eol_id:
                                    word_idx = eos_id
                            else:
                                word_idx = -1
                                while word_idx < 0:
                                    samples = torch.multinomial(word_weights, storyline_length)
                                    for sample in samples:
                                        sample = sample.item()
                                        if sample in [eot_id, eol_id, eos_id, delimiter_id]:
                                            continue
                                        if sample not in word_set:
                                            word_idx = sample
                                            word_set.add(word_idx)
                                            break
                                    if word_idx < 0:
                                        print("Duplicate All!!!")
                        else:
                            if last_word_idx in [period_id, eol_id]:
                                word_idx = start_id
                                if period_count == story_length:
                                    word_idx = eos_id
                            else:
                                word_idx = -1
                                while word_idx < 0:
                                    samples = torch.multinomial(word_weights, story_length)
                                    for sample in samples:
                                        sample = sample.item()
                                        if sample not in [eot_id, eol_id, eos_id, start_id]:
                                            word_idx = samples[0].item()
                                            break
                            if word_idx == period_id: period_count += 1
                        word = corpus.dictionary.idx2word[word_idx]
                    if word_idx == eos_id :
                        outf.write('\n')
                        break
                    word = corpus.dictionary.idx2word[word_idx]
                    outf.write(word + ' ')
                    input.data.fill_(word_idx)
                    output, hidden = model(input, hidden)
                data = data[idx+1:]  # start after the previous idx id
                print('| Generated {} sentences'.format(nsent+1))
                nsent += 1
            outf.flush()
        else:
            print("No longer support other task!")
