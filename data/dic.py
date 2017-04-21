# -*- coding: UTF-8 -*-
words = set()
with open('pku_train_all') as f:
	for line in f.readlines():
		sent =  unicode(line.decode('utf8')).split()
		for word in sent:
			words.add(word)
for word in words:
	print word.encode('utf8')

