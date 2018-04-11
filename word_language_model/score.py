#!/usr/bin/env python3

# coding: utf-8
import argparse
import time
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import pdb

import data
import model

# parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM Language Model')
# parser.add_argument('--data', type=str, default='./data/wikitext-2',
#                     help='location of the data corpus')
# args = parser.parse_args()

class Score:
    def __init__(self, data_source):
        self.data = data.Corpus(data_source)
        self.criterion = nn.CrossEntropyLoss()
        # pass

    def repackage_hidden(self, h):
        """Wraps hidden states in new Variables, to detach them from their history."""
        if type(h) == Variable:
            return Variable(h.data)
        else:
            return tuple(self.repackage_hidden(v) for v in h)

    def load_model(self, filestring):
        return torch.load(open(filestring, 'rb'))

    def get_batch(self, source, i, evaluation=False):
        # seq_len = min(args.bptt, len(source) - 1 - i)
        seq_len = len(source) - 1 - i
        data = Variable(source[i:i+seq_len], volatile=evaluation)
        target = Variable(source[i+1:i+1+seq_len].view(-1))
        return data, target

    def batchify(self, data, bsz):
        # Work out how cleanly we can divide the dataset into bsz parts.
        nbatch = data.size(0) // bsz
        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        data = data.narrow(0, 0, nbatch * bsz)
        # Evenly divide the data across the bsz batches.
        data = data.view(bsz, -1).t().contiguous()
        # if args.cuda:
        data = data.cuda()
        return data

    def score_sent(self, string, model):
        losses = []
        # manually adding back test data for testing
        testcorpus = self.data.tokenize_line(string)
        data_source = self.batchify(testcorpus, 1)
        # Turn on evaluation mode which disables dropout.
        model.eval()
        total_loss = 0
        # ntokens = len(corpus.dictionary)
        # DLK hack time:
        # pdb.set_trace()
        # File "/fs/project/white.1240/king/examples/word_language_model/score.py", line 65, in score_sent
        # ntokens = int(str(model.decoder).split(',')[1].split('=')[1][0:-1])
        # IndexError: list index out of range
        try:
            ntokens = int(
                str(
                    model.decoder
                ).split(',')[1].split('=')[1][0:-1]
            )
        except:
            ntokens = int(
                str(
                    model.decoder
                ).split(" ")[3][0:-1]
            )
        # hidden = self.model.init_hidden(self.eval_batch_size)
        hidden = model.init_hidden(1)
        # for i in range(0, data_source.size(0) - 1, args.bptt):
        for i in range(0, data_source.size(0) - 1, len(string)):
            data, targets = self.get_batch(data_source, i, evaluation=True)
            # print("data\n", data)
            # print("targets\n", targets)
            output, hidden = model(data, hidden)
            output_flat = output.view(-1, ntokens)
            total_loss += len(data) * self.criterion(output_flat, targets).data
            hidden = self.repackage_hidden(hidden)
        losses.append(total_loss)
        # pdb.set_trace()
        return losses # total_loss[0] / len(data_source)