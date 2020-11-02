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
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np
import logging

format_str = f'%(asctime)s %(filename)s [%(lineno)d] : %(message)s'
logging.basicConfig(format=format_str, datefmt="%d-%M-%Y %H:%M:%S", level=logging.DEBUG)
logger = logging.getLogger(__name__)

def parse_arguments():
    p = argparse.ArgumentParser(description='Hyperparams')
    p.add_argument('--epochs', type=int, default=100, help='number of epochs for train')
    p.add_argument('--batch_size', type=int, default=32, help='number of epochs for train')
    p.add_argument('--lr', type=float, default=0.01, help='initial learning rate')
    p.add_argument('--grad_clip', type=float, default=10.0, help='in case of gradient explosion')
    p.add_argument('--hidden_size', type=int, default=1024, help='hidden size')
    p.add_argument('--embed_size', type=int, default=512, help='embed size')
    p.add_argument('--encoder_layer', type=int, default=3, help='encoder layer')
    p.add_argument('--decoder_layer', type=int, default=3, help='decoder layer')
    p.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
    p.add_argument('--display', type=int, default=100, help='dropout rate')
    return p.parse_args()

def evaluate(model, val_iter, vocab_size, lang):
    model.eval()
    pad = 0
    total_loss = 0
    for b, batch in enumerate(val_iter):
        src = torch.from_numpy(batch["src"])
        trg = torch.from_numpy(batch["trg"])
        src, trg = src.cuda(), trg.cuda()
        output = model(src, trg, teacher_forcing_ratio=0.0)
        loss = F.nll_loss(output[1:].view(-1, vocab_size),
                               trg[1:].contiguous().view(-1),
                               ignore_index=pad)
        total_loss += loss.item()
    return total_loss / len(val_iter)


def train(e, model, optimizer, train_iter, vocab_size, grad_clip, lang, args):
    model.train()
    total_loss = 0
    pad = lang.word2index["<pad>"]
    for b, batch in enumerate(train_iter):
        src = torch.from_numpy(batch["src"])
        trg = torch.from_numpy(batch["trg"])
        #print_tensor(lang,src)
        #print_tensor(lang,trg)
        src, trg = src.cuda(), trg.cuda()
        optimizer.zero_grad()
        output = model(src, trg)
        loss = F.nll_loss(output[1:].view(-1, vocab_size),
                               trg[1:].contiguous().view(-1),
                               ignore_index=pad)
        loss.backward()
        clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item()

        if b % args.display == 0 and b != 0:
            total_loss = total_loss / args.display
            print("[%d][loss:%5.2f][pp:%5.2f]" %
                  (b, total_loss, math.exp(total_loss)))
            total_loss = 0


def main():


    args = parse_arguments()
    logger.info(args)

    assert torch.cuda.is_available()

    logger.info("preparing dataset...")
    lang, train_iter, val_iter, test_iter = load_data(args.batch_size)
    vocab_size = len(lang.index2word)
    logger.info("[lang_vocab]:%d" % (vocab_size))

    logger.info("Instantiating models...")
    encoder = Encoder(vocab_size, args.embed_size, args.hidden_size, n_layers = args.decoder_layer, dropout=args.dropout)
    decoder = Decoder(args.embed_size, args.hidden_size, vocab_size, n_layers = args.encoder_layer, dropout=args.dropout)
    seq2seq = Seq2Seq(encoder, decoder).cuda()
    optimizer = optim.Adam(seq2seq.parameters(), lr=args.lr)
    logger.info(seq2seq)

    best_val_loss = None
    for e in range(1, args.epochs+1):
        train(e, seq2seq, optimizer, train_iter,
              vocab_size, args.grad_clip, lang, args)
        val_loss = evaluate(seq2seq, val_iter, vocab_size, lang)
        print("[Epoch:%d] val_loss:%5.3f | val_pp:%5.2fS"
              % (e, val_loss, math.exp(val_loss)))

        if best_val_loss==None or True:
            logger.info("[!] saving model...")
            if not os.path.isdir("save"):
                os.makedirs("save")
            torch.save(seq2seq.state_dict(), './save/seq2seq_%d.pt' % (e))
            best_val_loss = val_loss
    
    test_loss = evaluate(seq2seq, test_iter, vocab_size, lang)
    print("[TEST] test_loss:%5.3f | test_pp:%5.2fS" % (test_loss, math.exp(test_loss)))


"""
lang, train_iter, val_iter, test_iter = load_data()
for i_batch, sample_batched in enumerate(train_iter):
    print(i_batch, sample_batched['src'].shape, sample_batched['trg'].shape,
          sample_batched['src'], sample_batched['trg'])
    if i_batch == 3:
        break
"""

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt as e:
        logger.error("STOP")
        logger.error(e)

