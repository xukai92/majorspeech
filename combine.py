import os, sys, time, gzip, operator
import helper as h
import numpy as np

# command constants
CNRESCORE = './scripts/cnrescore.sh {show} {model} decode {model}'
CNRESCORE2 = './scripts/cnrescore.sh {show} {model} decode-hybrid {model}' # decode/decode-hybrid

CNC2MLF = 'python /home/kx216/MLSALT11/iSpeech/final/cnc2mlf.py /home/kx216/MLSALT11/iSpeech/{model}/{show}/decode_cn/lattices/ /home/kx216/MLSALT11/iSpeech/{model}/{show}/decode_cn/lattices/cnc.mlf'

CNC2MLF2 = 'python /home/kx216/MLSALT11/iSpeech/final/cnc2mlf.py /home/kx216/MLSALT11/iSpeech/{model}/{show}/decode-hybrid_cn/lattices/ /home/kx216/MLSALT11/iSpeech/{model}/{show}/decode-hybrid_cn/lattices/cnc.mlf'

MAPTREE = 'base/conftools/smoothtree-mlf.pl {tree} {mlf_in} > {mlf_out}'

COMBINE1BEST = 'python /home/kx216/MLSALT11/iSpeech/final/combine_core.py /home/kx216/MLSALT11/iSpeech/combined/{show}/decode/rescore.mlf /home/kx216/MLSALT11/iSpeech/hybrid-int/{show}/decode_cn/rescore.mlf /home/kx216/MLSALT11/iSpeech/plp-adapt-int/{show}/decode-hybrid_cn/rescore.mlf /home/kx216/MLSALT11/iSpeech/tandem-adapt-int/{show}/decode-hybrid_cn/rescore.mlf /home/kx216/MLSALT11/iSpeech/grph-tandem-adapt-int/{show}/decode-hybrid_cn/rescore.mlf'

COMBINE1BEST = 'python /home/kx216/MLSALT11/iSpeech/final/combine_core.py /home/kx216/MLSALT11/iSpeech/combined/{show}/decode/rescore.mlf /home/kx216/MLSALT11/iSpeech/hybrid-int/{show}/decode_cn/rescore.mlf /home/kx216/MLSALT11/iSpeech/plp-adapt-int/{show}/decode-hybrid_cn/rescore.mlf /home/kx216/MLSALT11/iSpeech/tandem-adapt-int/{show}/decode-hybrid_cn/rescore.mlf /home/kx216/MLSALT11/iSpeech/grph-tandem-adapt-int/{show}/decode-hybrid_cn/rescore.mlf'

def print_table(table):
    for row in table:
        print row

class Entry(object):
    '''
    class to store a line of entries
    '''
    def __init__(self, line):
        attr = line.split()
        self.start = int(attr[0]) / 10000000.0
        self.end = int(attr[1]) / 10000000.0
        self.tokens = attr[2].replace('_<ALTSTART>', '').replace('_<ALT>', '').replace('_<ALTEND>', '').split('_')
        self.scores = [float(score) for score in attr[3].split('_')]

    def sort(self):
        pairs = zip(self.tokens, self.scores)
        pairs = sorted(pairs, key=operator.itemgetter(1), reverse=True)
        self.tokens = zip(*pairs)[0]
        self.scores = zip(*pairs)[1]

    def determinise(self):
        self.tokens = [self.tokens[0]]
        self.scores = [self.scores[0]]

    def add(self, another_entry):
        self.tokens += another_entry.tokens
        self.scores += another_entry.scores

    def __str__(self):
        start = str(int(self.start * 10000000))
        end = str(int(self.end * 10000000))
        scores = '_'.join('{0:.6f}'.format(score) for score in self.scores)
        if len(self.tokens) == 1:
            tokens = self.tokens[0]
        else:
            tokens = self.tokens[0] + '_<ALTSTART>_' + '_<ALT>_'.join(self.tokens[1:]) + '_<ALTEND>'
        return ' '.join([start, end, tokens, scores])

NULL = Entry('0 0 !NULL 0.2')

def read_mlf(mlf_file):
    '''
    read .mlf file
    '''
    mlf_list = {}
    f = open(mlf_file, 'r')
    for line in f:
        if line[0] == '"':
            stream = line.split('/')[-1][:-2]
            mlf_list[stream] = []
        elif line[0] != '#' and line[0] != '.':
            mlf_list[stream].append(Entry(line))   # [start, end, token, score]
    f.close()
    return mlf_list

def cost(s_entry, t_entry):
    '''
    cost function between entries
    entry attrs: start, end, token, score
    '''
    a, b, c, d = s_entry.start, s_entry.end, t_entry.start, t_entry.end
    if s_entry.tokens[0] == '!NULL' and t_entry.tokens[0] == '!NULL':   # start
        return 1000

    if s_entry.tokens[0] == t_entry.tokens[0]:
        return 0

    if s_entry.tokens[0] == '!NULL':     # ins
        return len(t_entry.tokens[0]) + 20
    elif t_entry.tokens[0] == '!NULL':   # del
        return len(s_entry.tokens[0]) + 20

    # TODO: include
    if a <= c and b <= c:
        gap = c - b
    elif a >= d and b >= d:
        gap = a - d
    else:
        gap = 0

    # gap = abs((b - a) / 2 - (d - c) / 2)

    if gap > 0.2:   # unmatched
        return 1000
    else:

        edit_distance, _, _ = compute_edit_distance([''] + list(s_entry.tokens[0]), [''] + list(t_entry.tokens[0]))
        return edit_distance

def compute_distance(s_list, t_list):
    '''
    compute distance between two mlf lists
    '''
    s_len, t_len = len(s_list), len(t_list)   # calculate length


    # initialization table E
    table_E = [[None for dummy_j in range(t_len)] for dummy_i in range(s_len)]
    for i in range(s_len):
        if i == 0:
            table_E[i][0] = cost(s_list[i], NULL)
        else:
            table_E[i][0] = table_E[i - 1][0] + cost(s_list[i], NULL)
    for j in range(t_len):
        if j == 0:
            table_E[0][j] = cost(NULL, t_list[j])
        else:
            table_E[0][j] = table_E[0][j - 1] + cost(NULL, t_list[j])

    # initialization table T
    # 0 for del, 1 for ins, 2 for sub
    table_T = [[None for dummy_j in range(t_len)] for dummy_i in range(s_len)]
    for i in range(s_len):
        table_T[i][0] = 0
    for j in range(t_len):
        table_T[0][j] = 1

    # complete table E
    for i in range(1, s_len):
        for j in range(1, t_len):
            # index, value = min(enumerate(), key=operator.itemgetter(1))
            table_T[i][j], table_E[i][j] = min(enumerate([
                table_E[i - 1][j] + cost(s_list[i], NULL),
                table_E[i][j - 1] + cost(NULL, t_list[j]),
                table_E[i - 1][j - 1] + cost(s_list[i], t_list[j])
            ]), key=operator.itemgetter(1))

    return table_E[i][j], table_E, table_T

def trace_back(table_E, table_T):
    '''
    trace back the editances table to find the alignment
    '''
    i, j = len(table_E) - 1, len(table_E[0]) - 1
    alignment = []
    while True:
        edit_type = table_T[i][j]
        if edit_type == 0:      # del
            alignment.append((i, 0))
            i -= 1
        elif edit_type == 1:    # ins
            alignment.append((0, j))
            j -= 1
        else:                   # sub
            alignment.append((i, j))
            i -= 1
            j -= 1
        if i < 0 or j < 0:
            break
    alignment.reverse()
    return alignment

def align_mlf(source_mlf, target_mlf):
    '''
    align mlf by defined cost function
    '''
    stream_list = source_mlf.keys()
    aligned_mlfs = {}
    for stream in stream_list:
        source_seq = [NULL] + source_mlf[stream]
        target_seq = [NULL] + target_mlf[stream]
        # if stream == 'DEV001-20010117-XX2000-en_MFWXXXX_0002506_0003705.rec':
        #     source_seq = source_seq[:10]
        #     target_seq = target_seq[:10]
        edit_distance, table_E, table_T = compute_distance(source_seq, target_seq)

        aligned_stream = []
        alignment = trace_back(table_E, table_T)
        for index in alignment:    # get rid of the empty string in the beginning
            source_idx, target_idx = index
            if len(source_seq) == 0:
                entry = Entry(str(target_seq[0]))
            elif len(target_seq) == 0:
                entry = Entry(str(source_seq[0]))
            else:
                if source_idx == 0:     # ins
                    entry = Entry(str(target_seq[target_idx]))
                    entry.tokens[0] = '!NULL'
                elif target_idx:
                    entry = Entry(str(source_seq[source_idx]))
                entry.add(target_seq[target_idx])

            # if stream == 'DEV001-20010117-XX2000-en_MFWXXXX_0002506_0003705.rec':
            #     print str(source_seq[source_idx]), str(target_seq[target_idx])
            #     print print_table(table_E)
            #     exit(1)
            # if source_idx < 0 and target_idx >= 0:      # ins
            #     entry = Entry(str(target_mlf[stream][target_idx]))
            #     entry.tokens = ['!NULL'] + entry.tokens
            #     entry.scores = [0.2] + entry.scores
            # elif source_idx >= 0 and target_idx < 0:    # del
            #     entry = Entry(str(source_mlf[stream][source_idx]))
            #     entry.add(NULL)
            # elif source_idx == target_idx:              # hit
            #     entry = Entry(str(target_mlf[stream][target_idx]))
            # else:                                       # sub
            #     entry = Entry(str(source_mlf[stream][source_idx]))
            #     entry.add(target_mlf[stream][target_idx])

            aligned_stream.append(entry)

        aligned_mlfs[stream] = aligned_stream
    return aligned_mlfs

def write_mlf(aligned_mlfs, output_path):
    '''
    write aligned mlfs to file
    '''
    out = '#!MLF!#\n'
    for stream in aligned_mlfs.keys():
        out += '"*/' + stream + '"\n'
        for entry in aligned_mlfs[stream]:
            entry = Entry(str(entry))
            if entry.tokens[0] != '!NULL':
                if entry.tokens[0] == 'OFFENDER' and stream == 'DEV001-20010117-XX2000-en_FFWXXXX_0113251_0115240.rec':
                    print entry
                    # exit(1)
                entry.sort()
                entry.determinise()
                if stream == 'DEV001-20010117-XX2000-en_FFWXXXX_0113251_0115240.rec':
                    print entry

                if entry.tokens[0] != '!NULL':
                    out += str(entry)
                    out += '\n'
        out += '.\n'

    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    f = open('{output_path}rescore.mlf'.format(output_path=output_path), "w")
    f.write(out)
    f.close()

def compute_edit_distance(s_list, t_list, cost=None):
    '''
    compute edit distance between two lists
    '''
    # default cost dictionary
    if not cost:
        cost = {
            'ins'   :   1,
            'del'   :   1,
            'sub'   :   1
        }
    s_len, t_len = len(s_list), len(t_list)   # calculate length

    # initialization table E
    table_E = [[None for dummy_j in range(t_len)] for dummy_i in range(s_len)]
    for i in range(s_len):
        table_E[i][0] = i * cost['del']
    for j in range(t_len):
        table_E[0][j] = j * cost['ins']

    # initialization table T
    # 0 for del, 1 for ins, 2 for sub
    table_T = [[None for dummy_j in range(t_len)] for dummy_i in range(s_len)]
    for i in range(s_len):
        table_T[i][0] = 0
    for j in range(t_len):
        table_T[0][j] = 1

    # complete table E
    for i in range(1, s_len):
        for j in range(1, t_len):
            # index, value = min(enumerate(), key=operator.itemgetter(1))
            table_T[i][j], table_E[i][j] = min(enumerate([
                table_E[i - 1][j] + cost['del'],
                table_E[i][j - 1] + cost['ins'],
                table_E[i - 1][j - 1] + (s_list[i] != t_list[j]) * cost['sub']
            ]), key=operator.itemgetter(1))

    return table_E[i][j], table_E, table_T

def align(source_mlf, target_mlf, cost=None):
    '''
    align two mlf by edit distance
    '''
    if not cost:
        cost = {
            'ins'   :   1,
            'del'   :   1,
            'sub'   :   1
        }

    stream_list = source_mlf.keys()
    aligned_mlfs = {}
    for stream in stream_list:
        source_seq = [''] + [entry[2] for entry in source_mlf[stream]]
        target_seq = [''] + [entry[2] for entry in target_mlf[stream]]
        edit_distance, table_E, table_T = compute_edit_distance(source_seq, target_seq, cost)

        aligned_stream = []
        alignment = trace_back(table_E, table_T)
        for index in alignment[1:]:    # get rid of the empty string in the beginning
            source_idx = index[0] - 1
            target_idx = index[1] - 1

            entry = {}
            if source_idx < 0 and target_idx >= 0:   # ins
                entry['start'] = target_mlf[stream][target_idx][0]
                entry['end'] = target_mlf[stream][target_idx][1]
                entry['tokens'] = ['!NULL', target_mlf[stream][target_idx][2]]
                entry['scores'] = ['0.2', target_mlf[stream][target_idx][3]]
            elif source_idx >= 0 and target_idx < 0: # del
                entry['start'] = source_mlf[stream][source_idx][0]
                entry['end'] = source_mlf[stream][source_idx][1]
                entry['tokens'] = [source_mlf[stream][source_idx][2], '!NULL']
                entry['scores'] = [source_mlf[stream][source_idx][3], '0.2']
            else:                                   # hit or sub
                entry['start'] = source_mlf[stream][source_idx][0]
                entry['end'] = source_mlf[stream][source_idx][1]
                entry['tokens'] = [source_mlf[stream][source_idx][2], target_mlf[stream][target_idx][2]]
                entry['scores'] = [source_mlf[stream][source_idx][3], target_mlf[stream][target_idx][3]]

            aligned_stream.append(entry)

        aligned_mlfs[stream] = aligned_stream
    return aligned_mlfs

def main():
    source_mlf_file = 'plp-bg/dev03_DEV001-20010117-XX2000/decode_cn/rescore.mlf'
    target_mlf_file = 'grph-plp-bg/dev03_DEV001-20010117-XX2000/decode_cn/rescore.mlf'
    target_mlf, source_mlf  = read_mlf(source_mlf_file), read_mlf(target_mlf_file)
    # compute_WER(source_mlf, target_mlf)
    #
    aligned_mlfs = align_mlf(source_mlf, target_mlf)
    # aa = aligned_mlfs['DEV001-20010117-XX2000-en_MFWXXXX_0170731_0172865.rec']
    # for e in aa:
    #     e.sort()
    #     e.determinise()
    #     print e.tokens[0] == '!NULL'
    #     print str(e)
    # exit(1)
    write_mlf(aligned_mlfs, 'plp-bg/dev03_DEV001-20010117-XX2000/combined/')

def test():
    e1 = Entry('25800000 32000000 INCLUDING 0.999956')
    e2 = Entry('32000000 32900000 INCLUDING 0.999996')


    print cost(e1, e2)
    print cost(e1, NULL)

    e = Entry('13100000 19100000 MILLY_<ALTSTART>_MOLLY_<ALT>_MILITARY_<ALTEND> 0.97_0.99_1.00')
    print(e)
    e.sort()
    print(e)
    e.determinise()
    print(e)
    exit(1)

def experiment():
    print 'System Combination'

    # model constants
    MODELS = ['plp-adapt-int', 'tandem-adapt-int', 'grph-tandem-adapt-int', 'hybrid-int']

    # combination
    showset = 'YTBEeval'
    print 'working on show set: {showset}'.format(showset=showset)

    for model in MODELS:
        for show in h.SHOWLIST[showset]:
            if model == 'hybrid-int':
                cmd = CNRESCORE.format(show=show,
                                       model=model)
            else:
                cmd = CNRESCORE2.format(show=show,
                                        model=model)
            print 'Running command:\n  {cmd}'.format(cmd=cmd)
            os.system(cmd)

    h.wait_qsub()

    cnc = False

    if cnc:

        for model in MODELS:
            for show in h.SHOWLIST[showset]:
                if model == 'hybrid-int':
                    cmd = CNC2MLF.format(show=show,
                                         model=model)
                else:
                    cmd = CNC2MLF2.format(show=show,
                                          model=model)
                print 'Running command:\n  {cmd}'.format(cmd=cmd)
                os.system(cmd)

    else:

        for model in MODELS:
            for show in h.SHOWLIST[showset]:
                if model == 'hybrid-int':
                    mlf_in = '/home/kx216/MLSALT11/iSpeech/{model}/{show}/decode_cn/rescore.mlf'.format(model=model,
                                                                                                        show=show)
                    mlf_out = '/home/kx216/MLSALT11/iSpeech/{model}/{show}/decode_cn/rescore_mapped.mlf'.format(model=model,
                                                                                                                show=show)
                else:
                    mlf_in = '/home/kx216/MLSALT11/iSpeech/{model}/{show}/decode-hybrid_cn/rescore.mlf'.format(model=model,
                                                                                                               show=show)
                    mlf_out = '/home/kx216/MLSALT11/iSpeech/{model}/{show}/decode-hybrid_cn/rescore_mapped.mlf'.format(model=model,
                                                                                                                       show=show)
                if model == 'hybrid-int':
                    modeltype = 'hybrid'
                else:
                    modeltype = str(model[:-10])
                tree = 'lib/trees/{modeltype}-bg_decode_cn.tree'.format(modeltype=modeltype)
                cmd = MAPTREE.format(tree=tree,
                                     mlf_in=mlf_in,
                                     mlf_out=mlf_out)
                print 'Running command:\n  {cmd}'.format(cmd=cmd)
                os.system(cmd)

    h.wait_qsub()

    if cnc:
        for show in h.SHOWLIST[showset]:
            cmd = COMBINECN.format(show=show)
            print 'Running command:\n  {cmd}'.format(cmd=cmd)
            os.system(cmd)
    else:

        for show in h.SHOWLIST[showset]:
            cmd = COMBINE1BEST.format(show=show)
            print 'Running command:\n  {cmd}'.format(cmd=cmd)
            os.system(cmd)

if __name__ == '__main__':
    main()
    # test()

def cn2mlf(cn_folder, mlf_file):
    '''
    convert confusion network to .mlf file
    '''
    # Get all cn file names
    print cn_folder + "*.gz"
    gz_files = [f for f in os.listdir(cn_folder) if f.endswith('.gz')]
    for _ in gz_files:
        print _

    lattices=[]
    for gz in gz_files:
        lattices.append(["F="+gz])
        f = gzip.open(cn_folder+gz, 'r')
        for iline in f:
            line = iline.split()
            lattices.append(line)

    # purning threshold
    THS = 0.1

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

    # format output
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

    # output to .mlf file
    fout = open(mlf_file, 'w')
    fout.write(out)
    fout.close()
