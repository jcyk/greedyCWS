
# -*- coding: UTF-8 -*-

dataset = 'pku' 
#dataset =  'as'
with open(dataset+'_train_all') as f:
	lines = f.readlines()
	idx = int(len(lines)*0.9)
	with open(dataset+'_trainx','wb') as fo:
		for line in lines[:idx]:
			fo.write(line.strip()+'\r\n')
	with open(dataset+'_devx','wb') as fo:
		for line in lines[idx:]:
			fo.write(line.strip()+'\r\n')
