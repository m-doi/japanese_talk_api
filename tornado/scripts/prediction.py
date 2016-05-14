# -*- coding: utf-8 -*-
import time
import math
import sys
import argparse
import os
import cPickle as pickle

import numpy as np
from chainer import cuda, Variable, FunctionSet
import chainer.functions as F
from CharRNN import CharRNN, make_initial_state


# Paasで動かしやすいように、MecabではなくIgoを使うようにした
# import MeCab
# mt = MeCab.Tagger('-Ochasen')

from igo.Tagger import Tagger
dicDir = os.path.join(os.path.dirname(__file__), "../dict/dict4igo")

import logging
app_log = logging.getLogger("tornado")


tagger = Tagger(dicDir)

def prediction(args, vocab="", model=""):

    output = ""
    np.random.seed(args.seed)

    # load vocabulary
    if vocab == "":
        vocab = pickle.load(open(args.vocabulary, 'rb'))
    ivocab = {}
    for c, i in vocab.items():
        ivocab[i] = c

    # load model
    if model == "":
        model = pickle.load(open(args.model, 'rb'))
    n_units = model.embed.W.data.shape[1]

    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
        model.to_gpu()

    # initialize generator
    state = make_initial_state(n_units, batchsize=1, train=False)
    if args.gpu >= 0:
        for key, value in state.items():
            value.data = cuda.to_gpu(value.data)

    prev_char = np.array([0], dtype=np.int32)
    if args.gpu >= 0:
        prev_char = cuda.to_gpu(prev_char)
    if len(args.primetext) > 0:
        words = []
        results = tagger.parse(args.primetext)
        for result in results:
            words.append(result.surface)
        # Mecabを使う場合はこちら
        # result = mt.parseToNode(args.primetext.encode('utf-8'))
        # while result:
        #     words.append(unicode(result.surface, 'utf-8'))
        #     result = result.next

        for word in words:
            if (word in vocab):
                val = vocab[word]
            else:
                val = 0
            prev_char = np.ones((1,)).astype(np.int32) * val
            if args.gpu >= 0:
                prev_char = cuda.to_gpu(prev_char)

            state, prob = model.predict(prev_char, state)

    for i in xrange(args.length):
        state, prob = model.predict(prev_char, state)

        if args.sample > 0:
            probability = cuda.to_cpu(prob.data)[0].astype(np.float64)
            probability /= np.sum(probability)
            index = np.random.choice(range(len(probability)), p=probability)
        else:
            index = np.argmax(cuda.to_cpu(prob.data))
        output += ivocab[index]

        prev_char = np.array([index], dtype=np.int32)
        if args.gpu >= 0:
            prev_char = cuda.to_gpu(prev_char)

    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model',      type=str,   required=True)
    parser.add_argument('--vocabulary', type=str,   required=True)

    parser.add_argument('--seed',       type=int,   default=123)
    parser.add_argument('--sample',     type=int,   default=1)
    parser.add_argument('--primetext',  type=str,   default='')
    parser.add_argument('--length',     type=int,   default=2000)
    parser.add_argument('--gpu',        type=int,   default=-1)

    args = parser.parse_args()
    prediction(args)
