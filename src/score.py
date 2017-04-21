import os
import sys

golden_file = sys.argv[1]
s = int(sys.argv[2])
e = int(sys.argv[3])

for i in xrange(s,e+1):
    cmd = './score ../data/dic %s ../result/dev_result%s > tmp'%(golden_file,i)
    os.system(cmd)
    cmd = 'grep \'F MEASURE\' tmp '
    os.system(cmd)
    cmd = 'rm tmp'
    os.system(cmd)

