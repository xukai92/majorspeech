# -*- coding: utf-8 -*-


import numpy as np
import sys
import os
# import glob
import gzip
import operator


def print_help():
    print '-' * 50
    print 'Usage:'
    print '  python cnc2mlf.py path_to_cnc_folder output_mlf'
    print ''
    print 'Example:'
    print '  python cnc2mlf.py /home/msa53/MLSALT11/grph-plp-bg/dev03_DEV001-20010117-XX2000/decode_cn/lattices/ cnc.mlf'
    print '-' * 50


if len(sys.argv) < 3:
    print_help()
    exit(1)

# arg1 = "/home/msa53/MLSALT11/grph-plp-bg/dev03_DEV001-20010117-XX2000/decode_cn/lattices/"
arg1, arg2 = sys.argv[1], sys.argv[2]

# Get all file names
print arg1+"*.gz"
gz_files = [f for f in os.listdir(arg1) if f.endswith('.gz')]
for _ in gz_files:
    print _

lattices=[]
for gz in gz_files:
    lattices.append(["F="+gz])
    f = gzip.open(arg1+gz, 'r')
    for iline in f:
        line = iline.split()
        lattices.append(line)

########################################
print "############### Printing Off Output ####################"

THS = 0.1 # purning threshold

# cnc to mlf
lattices_dict = {}
k = -1
first = False
for line in lattices:
    if line[0][0] == 'F':
        fileName = line[0].strip("F=")[:-6] + 'rec'
        lattices_dict[fileName] = {}
    if line[0][0] == 'k':
        k = int(line[0].split('=')[1])
        node_set = []
        first = True
    if k > 0 and line[0][0] == 'W':
        entries = []
        for entry in line:
            entries.append(entry.split('=')[1])
        node_set.append(entries)
        k -= 1
    if k == 0:
        logProbs = [float(e[3]) for e in node_set]
        max_index, _ = max(enumerate(logProbs), key=operator.itemgetter(1))
        startTime = float(node_set[max_index][1])
        endTime = float(node_set[max_index][2])
        lattices_dict[fileName][startTime] = {'endTime': endTime, 'entries': []}
        for entries in node_set:
            token = entries[0]
            logProb = float(entries[3])
            # if np.exp(logProb) > THS:
            lattices_dict[fileName][startTime]['entries'].append({'token': token,
                                                                  'logProb': logProb})


out = '#!MLF!#\n'
for fileName in lattices_dict.keys():
    out += '"*/{fileName}"\n'.format(fileName=fileName)
    for startTime in sorted(lattices_dict[fileName].keys()):
        entries = lattices_dict[fileName][startTime]['entries']
        entries = sorted(entries, key=lambda x: x['token'], reverse=True)
        if entries[0]['token'] != '<s>' and entries[0]['token'] != '</s>':
            startTimeString = '{0:.7f}'.format(startTime).replace('.', '')
            if startTimeString[0] == '0':
                startTimeString = startTimeString[1:]
            endTimeString = '{0:.7f}'.format(lattices_dict[fileName][startTime]['endTime']).replace('.', '')
            if endTimeString[0] == '0':
                endTimeString = endTimeString[1:]
            out += '{startTime} {endTime} '.format(startTime=startTimeString,
                                                   endTime=endTimeString)
            l = len(entries)
            if entries[0]['token'] == '!NULL':
                entries[0]['token'] = '<DEL>'
            if l == 1:
                out += '{token} '.format(token=entries[0]['token'])
                out += '{0:.6f}'.format(np.exp(entries[0]['logProb']))
            else:
                out += '{token1}_<ALTSTART>_'.format(token1=entries[0]['token'])
                for i in range(1, l - 1):
                    if entries[i]['token'] == '!NULL':
                        out += '<DEL>_'
                    else:
                        out += entries[i]['token'] + '_<ALT>_'
                if entries[l-1]['token'] != '!NULL':
                    out += entries[l-1]['token']
                else:
                    out += '<DEL>_'
                out += '<ALTEND> '
                for i in range(l):
                    out += '{0:.6f}'.format(np.exp(entries[i]['logProb']))
                    if i != l - 1:
                        out += '_'
            out += '\n'
    out += '.\n'

# print out
fout = open(arg2, 'w')
fout.write(out)
fout.close()
