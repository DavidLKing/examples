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

import score


class Realize:
    def __init__(self, data_loc, model_loc):
        self.score = score.Score(data_loc)
        self.model = self.score.load_model(model_loc)

    # def lookup(self, string):
    #     return self.score.data.dictionary.word2idx[string]
    #
    # def vectorize(self, seq):
    #     ids = torch.LongTensor(tokens)
    #     words = seq.split() + ['<eos>']
    #     token = 0
    #     for w in words:
    #         ids[token] = self.lookup(w)
    #         token += 1
    #     return ids

    def score(self, seq):
        return self.score.score_sent(seq, self.model)

    def main(self):
        test = 't h i s @ a @ v e r y @ f i n e @ t e s t'
        testscore = lambda x: self.score.score_sent(x, self.model)
        testscore(test)
        pdb.set_trace()

if __name__ == '__main__':
    r = Realize()
    r.main()