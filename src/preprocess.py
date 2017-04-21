# -*- coding: UTF-8 -*-
import re
import sys

Maximum_Word_Length = 4

def OT(str):
    print str.encode('utf8')

def strQ2B(ustring):
    """全角转半角"""
    rstring = ""
    for uchar in ustring:
        inside_code=ord(uchar)
        if inside_code == 12288:                              #全角空格直接转换
            inside_code = 32
        elif (inside_code >= 65281 and inside_code <= 65374): #全角字符（除空格）根据关系转化
            inside_code -= 65248
        
        rstring += unichr(inside_code)
    return rstring

def preprocess(path, longws= set()):
    rNUM = u'(-|\+)?\d+((\.|·)\d+)?%?'
    rENG = u'[A-Za-z_.]+'
    word_count, char_count, sent_count = 0, 0, 0
    count_longws = 0
    with open(path,'r') as f:
        sents = []
        for line in f.readlines():
            sent = strQ2B(unicode(line.decode('utf8')).strip()).split()
            new_sent = []
            for word in sent:
                word = re.sub(u'\s+','',word,flags =re.U)
                word = re.sub(rNUM,u'0',word,flags= re.U)
                word = re.sub(rENG,u'X',word)
                if word in longws:
                    count_longws+=1
                    word = u'L'
                new_sent.append(word)
                char_count+=len(word)
                word_count+=1
            sents.append(new_sent)
            sent_count+=1
    print  path
    print 'long words count', count_longws
    print  'sents %d, words %d chars %d' %(sent_count, word_count, char_count)
    return sents

def write(filename, sents):
    f= open(filename,'w')
    for sent in sents:
        f.write('  '.join(sent).encode('utf8')+'\r\n')
    f.close() 

def check(sents): # get those words longer than our maximum word length setting
    count = 0
    all_count = 0
    longwords = []
    for sent in sents:
        for word in sent:
            all_count+=1
            if len(word)>Maximum_Word_Length:
                count+=1
                longwords.append(word)
    for word in set(longwords):
        OT(word)
    print 'len>%d words count'%Maximum_Word_Length,count,100.0*count/all_count,'%'
    return set(longwords)

if __name__ == "__main__":
    dataset = sys.argv[1]
    sents = preprocess('../original/%s_training.utf8'%dataset)
    longwords = check(sents)
    write('../data/%s_train_all'%dataset, preprocess('../original/%s_training.utf8'%dataset, longwords))
    write('../data/%s_test'%dataset, preprocess('../original/%s_test_gold.utf8'%dataset, longwords))
