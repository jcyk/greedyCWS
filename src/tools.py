# -*- coding: UTF-8 -*-
from collections import defaultdict

import numpy as np
import random
import gensim
import re

def initCemb(ndims,train_file,pre_trained,thr = 5.):
    f = open(train_file)
    train_vocab = defaultdict(float)
    for line in f.readlines():
        sent = unicode(line.decode('utf8')).split()
        for word in sent:
            for character in word:
                train_vocab[character]+=1
    f.close()
    character_vecs = {}
    for character in train_vocab:
        if train_vocab[character]< thr:
            continue
        character_vecs[character] = np.random.uniform(-0.5/ndims,0.5/ndims,ndims)
    if pre_trained is not None:
        pre_trained = gensim.models.Word2Vec.load(pre_trained)
        pre_trained_vocab = set([ unicode(w.decode('utf8')) for w in pre_trained.vocab.keys()])
        for character in pre_trained_vocab:
            character_vecs[character] = pre_trained[character.encode('utf8')]
    Cemb = np.zeros(shape=(len(character_vecs)+1,ndims))
    idx = 1
    character_idx_map = dict()
    for character in character_vecs:
        Cemb[idx] = character_vecs[character]
        character_idx_map[character] = idx
        idx+=1
    return Cemb,character_idx_map

def SMEB(lens):
    idxs = []
    for len in lens:
        for i in xrange(len-1):
            idxs.append(0)
        idxs.append(len)
    return idxs

def prepareData(character_idx_map,path,test=False):
    seqs,wlenss,idxss = [],[],[]
    f = open(path)
    for line in f.readlines():
        sent = unicode(line.decode('utf8')).split()
        Left = 0
        for idx,word in enumerate(sent):
            if len(re.sub('\W','',word,flags=re.U))==0:
                if idx >Left:
                    seqs.append(list(''.join(sent[Left:idx])))
                    wlenss.append([len(word) for word in sent[Left:idx]])
                Left = idx+1
        if Left!=len(sent):
            seqs.append(list(''.join(sent[Left:])))
            wlenss.append([ len(word) for word in sent[Left:]])
    seqs = [[ character_idx_map[character] if character in character_idx_map else 0 for character in seq] for seq in seqs]
    f.close()
    if test:
        return seqs
    for wlens in wlenss:
        idxss.append(SMEB(wlens))
    return seqs,wlenss,idxss