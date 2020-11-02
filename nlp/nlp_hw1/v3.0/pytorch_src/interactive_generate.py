###############################################################################
# Language Modeling on Penn Tree Bank
#
# This file generates new sentences sampled from the language model
#
###############################################################################

import argparse
import sys
import os
import _pickle as pickle

import numpy, math
import torch
import torch.nn as nn
from numbers import Number
from utils import batchify, get_batch, repackage_hidden, load_pickle
from ir_baseline import IRetriver


def load_model(path, cuda):
    with open(path, 'rb') as f:
        #model = torch.load(f, map_location=lambda storage, loc: storage)
        model = torch.load(f, map_location=lambda storage, loc: storage)
    model.eval()
    if not hasattr(model, 'tie_weights'):
        model.tie_weights = True
    if cuda:
        model.cuda()
    else:
        model.cpu()
    return model


def to_tensor(vocab, tokens, cuda):
    ids = torch.LongTensor(len(tokens))
    for i, token in enumerate(tokens):
        ids[i] = vocab.word2idx.get(token, 0)
    if cuda:
        ids = ids.cuda()
    return ids

def generate(model, vocab, prefix, eos_id, max_len, dedup, cuda, temperature):
    prefix = to_tensor(vocab, prefix, cuda).tolist()
    cond_length = len(prefix)
    hidden = model.init_hidden(1)
    tokens = []
    input = torch.rand(1, 1).mul(len(vocab)).long()
    if cuda:
       input.data = input.data.cuda()
    exist_word = set()
    for i in range(max_len):
        if i < cond_length:
            word_idx = prefix[i]
        else:
            word_weights = output.squeeze().data.div(temperature)
            word_weights = (word_weights - torch.max(word_weights)).exp()#.cpu()
            samples = torch.multinomial(word_weights, 5)
            if dedup:
                for word_idx in samples:
                    if word_idx not in exist_word:
                        break
                exist_word.add(word_idx)
            else:
                word_idx = samples[0]
        input.data.fill_(word_idx)
        output, hidden = model(input, hidden)
        if word_idx == eos_id:
            break
        word = vocab.idx2word[word_idx]
        tokens.append(word)
    return tokens

def compute_similarity(model, user_word):
    all_word_embeddings = model.encoder.weight
    vocab_size, emb_size = all_word_embeddings.size()

    #user_word = Variable(user_word)
    user_word_embedding = model.encoder(user_word).repeat(vocab_size, 1)

    sim_score = nn.CosineSimilarity(dim=1)
    scores = sim_score(user_word_embedding, all_word_embeddings)

    return scores


def constrained_generate(model, vocab, ref, prefix, eos_id, max_len, dedup, cuda, temperature, K=1, extendable=False):
    prefix = to_tensor(vocab, prefix, cuda).tolist()
    ref = to_tensor(vocab, ref, cuda)
    cond_length = len(prefix)
    hidden = model.init_hidden(1)
    tokens = []
    input = torch.rand(1, 1).mul(len(vocab)).long()
    if cuda:
       input.data = input.data.cuda()
    exist_word = set()
    vocab_size = len(vocab)
    mask = torch.zeros(vocab_size)
    for i in range(max_len):
        #print i
        if i < cond_length:
            #print('prefix:', vocab.idx2word[prefix[i]])
            word_idx = prefix[i]
        else:
            j = i - cond_length
            if j < len(ref):
                user_word = ref[j:j+1]
                word_similarity_weights = compute_similarity(model, user_word) #+ 1.
                _, topk_word_inds = torch.topk(word_similarity_weights, K)
                topk_word_inds = topk_word_inds.data#.cpu()
                mask.zero_()
                mask[topk_word_inds] = 1.
            else:
                mask.fill_(1.)

            # output: (seq_len, batch_size, vocab_size) where seq_len = batch_size = 1
            word_weights = output.squeeze().data.div(temperature)
            word_weights = (word_weights - torch.max(word_weights)).exp()#.cpu()
            #word_weights = word_weights * mask

            samples = torch.multinomial(word_weights, 5)
            if dedup:
                for word_idx in samples:
                    if word_idx not in exist_word:
                        break
                exist_word.add(word_idx)
            else:
                word_idx = samples[0]

            if j == len(ref) and not extendable:
                word_idx = eos_id

            #print('sampled word:', vocab.idx2word[word_idx])

        input.data.fill_(word_idx)
        output, hidden = model(input, hidden)
        word = vocab.idx2word[word_idx]
        tokens.append(word)
        if word_idx == eos_id:
            break

    return tokens

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Language Model')

    # Model parameters.
    parser.add_argument('--keyword-vocab', type=str, required=True, default='models/keyword_dict.pkl')
    parser.add_argument('--story-vocab', type=str, required=True, default='models/story_dict.pkl')
    parser.add_argument('--keyword-model', type=str, required=True)
    parser.add_argument('--story-model', type=str, required=True)
    parser.add_argument('--titles', type=str, required=True)
    parser.add_argument('--words', type=int, default='1000',
                        help='number of words to generate')
    parser.add_argument('--dedup', action='store_true',
                        help='de-duplication')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='temperature - higher will increase diversity')
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

    kw_model = load_model(args.keyword_model, args.cuda)
    st_model = load_model(args.story_model, args.cuda)
    kw_vocab = load_pickle(args.keyword_vocab)
    st_vocab = load_pickle(args.story_vocab)
    kw_vocab_size = len(kw_vocab)
    st_vocab_size = len(st_vocab)
    kw_eos_id = kw_vocab.word2idx['<eos>']
    st_eos_id = st_vocab.word2idx['<eos>']

    with open(args.titles, 'r') as fin:
        titles = fin.readlines()
        title_id = 0
        while True:
            title = titles[title_id % len(titles)].split()
            print('Title:', title_id, ' '.join(title[:-1]))

            print('Model generated keywords:')
            # title ends with EOT
            tokens = generate(kw_model, kw_vocab, title, kw_vocab.word2idx['<EOL>'], 15, True, args.cuda, args.temperature)
            print(' '.join(tokens))

            keywords = input('Keywords: ')
            keywords = keywords.split()

            print('Suggested keywords: ')
            tokens = constrained_generate(kw_model, kw_vocab, keywords, title, kw_vocab.word2idx['<EOL>'], 15, True, args.cuda, 0.15)
            print(' '.join(tokens))

            prefix = tokens
            tokens = generate(st_model, st_vocab, prefix, st_eos_id, args.words, args.dedup, args.cuda, args.temperature)
            print(' '.join(tokens))

            title_id = input('Title ID: ')
            title_id = int(title_id)
