import subprocess
import os, sys, time
import helper as h

# command constants
LMRESCORE = './scripts/lmrescore.sh {show} lattices decode {lm} {sys} FALSE'
SCORE = './scripts/score.sh {sys} {show_set} rescore'
LPLEX = 'base/bin/LPlex -C lib/cfgs/hlm.cfg -s {stream} -u -t {lm} {dat_file}'  # LPlex with stream
LPLEX2 = 'base/bin/LPlex -C lib/cfgs/hlm.cfg -u -t {lm} {dat_file}'             # LPlex without stream
LMERGE = 'base/bin/LMerge -C lib/cfgs/hlm.cfg -i {weights[1]} lms/lm2 -i {weights[2]} lms/lm3 -i {weights[3]} lms/lm4 -i {weights[4]} lms/lm5 lib/wlists/train.lst lms/lm1 {lm}'

def lmrescore(show, lm, sys):
    '''
    wrapper for LMRESCORE command using in this task
    '''
    cmd = LMRESCORE.format(
        show=show,
        lm=lm,
        sys=sys
    )

    # print log
    log_txt = 'Running command:\n  {cmd}'.format(cmd=cmd)
    h.print_log(log_txt)

    os.system(cmd)

def lmrescore_batch(show_set, lm, sys):
    '''
    running LMRESCORE for all shows in a show set given a LM
    '''
    for show in h.SHOWLIST[show_set]:
        lmrescore(show, lm, sys)

    # print log
    log_txt = 'LMRESCORE: show_set = {show_set}, lm = {lm}, sys = {sys}'.format(
        show_set=show_set,
        lm=lm,
        sys=sys
    )
    h.print_log(log_txt)

def lmrescore_batch_show_specific(show_set, sys):
    '''
    running LMRESCORE for all shows in a show set given a LM
    '''
    for show in h.SHOWLIST[show_set]:
        lm = 'my_lms/lm_int_{show}'.format(show=show)
        lmrescore(show, lm, sys)

    # print log
    log_txt = 'LMRESCORE: show_set = {show_set}, lm = {lm}, sys = {sys}'.format(
        show_set=show_set,
        lm=lm,
        sys=sys
    )
    h.print_log(log_txt)

def score(sys, show_set):
    '''
    wrapper for SCORE command using in this task
    '''
    cmd = SCORE.format(
        sys=sys,
        show_set=show_set
    )

    # print log
    log_txt = 'Running command:\n  {cmd}'.format(cmd=cmd)
    h.print_log(log_txt)

    f = os.popen(cmd)
    return f.read()

def mlf2dat(mlf_file, dat_file):
    '''
    convert .mlf file to .dat file
    '''
    # input and convert
    out = ''
    f = open(mlf_file, 'r')
    for line in f:
        if line[0] == '"':              # if show name
            w_list = []                 # create a new sentence
            w_list.append('<s>')        # sentence start symbol
        elif line[0] == '.':            # if ending
            w_list.append("</s>")       # sentence end symbol
            sentence = ' '.join(w_list) # convert world list to sentence string
            out += sentence             # append sentence to output string
            out += '\n'                 # line break
        elif line[0] != '#':            # if not '#!MLF!#'
            word = line.split(' ')[2]   # get the word
            w_list.append(word)
    f.close()

    # output
    f = open(dat_file, "w")
    f.write(out)
    f.close()

def batch_mlf2dat(sys, show_set):
    '''
    convert all .mlf files in a show set to .dat for a given system
    '''
    # print log
    log_txt = 'mlf2dat: sys = {sys}, show_set = {show_set}'.format(
        sys=sys,
        show_set=show_set
    )
    h.print_log(log_txt)

    for show in h.SHOWLIST[show_set]:
        sentences = []
        mlf_file = "{sys}/{show}/rescore/rescore.mlf".format(
            sys=sys,
            show=show
        )
        dat_file = "{sys}/{show}/rescore/rescore.dat".format(
            sys=sys,
            show=show
        )
        mlf2dat(mlf_file, dat_file)

def lplex(stream, lm, dat_file):
    '''
    wrapper for LPLEX command using in this task
    '''
    cmd = LPLEX.format(
        stream=stream,
        lm=lm,
        dat_file=dat_file
    )

    # print log
    log_txt = 'Running command:\n  {cmd}'.format(cmd=cmd)
    h.print_log(log_txt)

    os.system(cmd)

def lplex2(lm, dat_file):
    '''
    wrapper for LPLEX2 command using in this task
    '''
    cmd = LPLEX2.format(
        lm=lm,
        dat_file=dat_file
    )

    # print log
    log_txt = 'Running command:\n  {cmd}'.format(cmd=cmd)
    h.print_log(log_txt)

    f = os.popen(cmd)
    return f.read()

def read_stream(stream):
    '''
    read the probability stream into a list
    '''
    f = open(stream)
    p_list = []
    for line in f:
        p_list.append(float(line.replace("\n", "")))
    f.close()

    return p_list

def weighted_p(weights, p_lists):
    '''
    compute the weighted probabilities and store them in a list
    '''
    weighted_p_lists = []
    lm_num = len(p_lists)
    word_num = len(p_lists[0])
    for lm_id in range(lm_num):
        p = []
        for i in range(word_num):
            p.append(weights[lm_id] * p_lists[lm_id][i] /
                     (weights[0] * p_lists[0][i] +
                      weights[1] * p_lists[1][i] +
                      weights[2] * p_lists[2][i] +
                      weights[3] * p_lists[3][i] +
                      weights[4] * p_lists[4][i]
                     ))
        weighted_p_lists.append(p)
    return weighted_p_lists

def estimate_weights(p_lists, ths=0.000001, max_iter=500):
    '''
    use EM algorithm to estimate the the optimal interpolated weights
    '''
    # print log
    log_txt = 'Estimate weights ...'
    h.print_log(log_txt)

    lm_num = len(p_lists)
    weights = [1.0 / lm_num for dummy_i in range(lm_num)]
    w_1_pre = weights[0]
    for iter_num in range(max_iter):
        weighted_p_lists = weighted_p(weights, p_lists)
        for lm_id in range(lm_num):
            word_num = len(weighted_p_lists[lm_id])
            weights[lm_id] = 1.0 / (word_num + 1) * sum(weighted_p_lists[lm_id])
        if abs(w_1_pre - weights[0]) < ths:
            break
        else:
            w_1_pre = weights[0]

    # print log
    log_txt = '... {iter_num} iteration used'.format(iter_num=iter_num)
    h.print_log(log_txt)

    return weights

def lmerge(weights, lm):
    '''
    merge language models given weights
    '''
    cmd = LMERGE.format(
        weights=weights,
        lm=lm
    )

    # print log
    log_txt = 'Running command:\n  {cmd}'.format(cmd=cmd)
    h.print_log(log_txt)

    os.system(cmd)

def lm_inter():
    # task setting
    dev_set = 'dev03'
    eval_set = 'eval03'
    lm_list = ['lms/lm1', 'lms/lm2', 'lms/lm3', 'lms/lm4', 'lms/lm5']
    sys_list = ['plp-tglm1', 'plp-tglm2', 'plp-tglm3', 'plp-tglm4', 'plp-tglm5']
    out = ''

    # generate mlf using the five given LMs
    for lm, sys in zip(lm_list, sys_list):
        lmrescore_batch(dev_set, lm, sys)
        lmrescore_batch(eval_set, lm, sys)

    h.wait_qsub()

    # score systems
    for sys in sys_list:
        out += '{sys} {show_set} score result\n'.format(
            sys=sys,
            show_set=dev_set
        )
        out += score(sys, dev_set) + '\n'

    for sys in sys_list:
        out += '{sys} {show_set} score result\n'.format(
            sys=sys,
            show_set=eval_set
        )
        out += score(sys, eval_set) + '\n'

    # find perplexity of the five LMs
    for sys, lm in zip(sys_list, lm_list):
        out += '{sys} {show_set} perplexity\n'.format(
            sys=sys,
            show_set=dev_set
        )
        out += lplex2(lm, 'lib/texts/dev03.dat') + '\n'

    for sys, lm in zip(sys_list, lm_list):
        out += '{sys} {show_set} perplexity\n'.format(
            sys=sys,
            show_set=eval_set
        )
        out += lplex2(lm, 'lib/texts/eval03.dat') + '\n'

    # generate streams
    for sys, lm in zip(sys_list, lm_list):
        stream = '{sys}/stream'.format(sys=sys)
        lplex(stream, lm, 'lib/texts/dev03.dat')

    # read stream
    p_lists = []
    for sys in sys_list:
        stream = '{sys}/stream'.format(sys=sys)
        p_lists.append(read_stream(stream))

    # estimate weight
    weights = estimate_weights(p_lists)

    # merge LMs
    lm = 'my_lms/lm_int_dev03'.format(show_set=dev_set)
    lmerge(weights, lm)

    # generate mlf using interpolated LM
    sys = 'plp-tglm_int_dev03'
    lmrescore_batch(dev_set, lm, sys)
    lmrescore_batch(eval_set, lm, sys)

    h.wait_qsub()

    # score the interpolated LM
    out += '{sys} {show_set} score result\n'.format(
        sys=sys,
        show_set=dev_set
    )
    out += score(sys, dev_set) + '\n'

    out += '{sys} {show_set} score result\n'.format(
        sys=sys,
        show_set=eval_set
    )
    out += score(sys, eval_set) + '\n'

    # fin the perplexity of the interpolated LM
    out += '{sys} {show_set} perplexity\n'.format(
        sys=sys,
        show_set=dev_set
    )
    out += lplex2(lm, 'lib/texts/dev03.dat') + '\n'

    out += '{sys} {show_set} perplexity\n'.format(
        sys=sys,
        show_set=eval_set
    )
    out += lplex2(lm, 'lib/texts/eval03.dat') + '\n'

    # write result to file
    f = open('report/inter-03.txt', 'w')
    f.write(out)
    f.close()

def lm_inter_show_specific():
    # task setting
    dev_set = 'dev03'
    eval_set = 'eval03'
    lm_int = 'my_lms/lm_int_dev03'
    sys_int = 'plp-tglm_int_dev03'
    lm_list = ['lms/lm1', 'lms/lm2', 'lms/lm3', 'lms/lm4', 'lms/lm5']
    sys_list = ['plp-tglm1', 'plp-tglm2', 'plp-tglm3', 'plp-tglm4', 'plp-tglm5']
    out = ''

    # convert .mlf to .dat for stream generation
    batch_mlf2dat(sys_int, eval_set)

    for show in h.SHOWLIST[eval_set]:
        # generate streams
        for sys, lm in zip(sys_list, lm_list):
            stream = '{sys}/{show}/stream'.format(
                sys=sys,
                show=show
            )
            dat_file = "{sys_int}/{show}/rescore/rescore.dat".format(
                sys_int=sys_int,
                show=show
            )
            lplex(stream, lm, dat_file)

        # read stream
        p_lists = []
        for sys in sys_list:
            stream = '{sys}/{show}/stream'.format(
                sys=sys,
                show=show
            )
            p_lists.append(read_stream(stream))

        # estimate weight
        weights = estimate_weights(p_lists)

        # merge LMs
        lm = 'my_lms/lm_int_{show}'.format(show=show)
        lmerge(weights, lm)

    # generate mlf using interpolated LM
    sys = 'plp-tglm_int-show-specific'
    lmrescore_batch_show_specific(eval_set, sys)

    h.wait_qsub()

    # score the interpolated LM
    out += '{sys} {show_set} score result\n'.format(
        sys=sys,
        show_set=eval_set
    )
    out += score(sys, eval_set) + '\n'

    # find the perplexity of the interpolated LM
    for show in h.SHOWLIST[eval_set]:
        lm = 'my_lms/lm_int_{show}'.format(show=show)
        dat_file = "{sys_int}/{show}/rescore/rescore.dat".format(
            sys_int=sys_int,
            show=show
        )
        out += '{sys} {show_set} perplexity\n'.format(
            sys=sys,
            show_set=eval_set
        )
        out += lplex2(lm, dat_file) + '\n'

    # write result to file
    f = open('report/inter-03_show_specific.txt', 'w')
    f.write(out)
    f.close()

def main():
    lm_inter()
    lm_inter_show_specific()

if __name__ == '__main__':
    main()


# 5.4.6
# i) using the global, dev03 tuned, interpolation weights (from (3)) generate the 1-best hypotheses;
# this step is done in the 5.4.5
# , and the results are stored in the corresponding dev folders in plp-tglm_int

# ii) map the 1-best hypotheses to an appropriate text data format to train the interpolation weights;
# this is done by converting the mlf files to .dat files

# iii) compute interpolation weights using the data using the mapped hypotheses in ii), and merge the LMs;

# run the following command to generate streams for each show and each language model
# base/bin/LPlex -C lib/cfgs/hlm.cfg -s stream -u -t lms/lm1 lib/texts/dev03.dat

# iv) apply the new show-specifc LM to the lattices.
# base/bin/LMerge -C lib/cfgs/hlm.cfg -i 0.6 lms/lm2 -i 0.1 lms/lm3 lib/wlists/train.lst lms/lm1 lm_int

# w_dict = {'eval03_DEV015-20010225-XX0900':
#           [0.24472594018952235, 0.13317904702881725, 0.1440927688623227, 0.06849029257497409, 0.40929241237619723],
#           'eval03_DEV014-20010221-XX1830':
#           [0.4212825979101972, 0.04930039652691214, 0.23251847416036353, 0.09916000519946955, 0.19752235672618715],
#           'eval03_DEV016-20010228-XX2100':
#           [0.4070774768379272, 0.2150202620204508, 0.07676296462686798, 0.20759426665429326, 0.0933256835125806],
#           'eval03_DEV012-20010217-XX1000':
#           [0.3483917738542753, 0.032022213347708724, 0.15801135796982085, 0.057185195399993144, 0.404184876776809],
#           'eval03_DEV013-20010220-XX2000':
#           [0.39524107026204536, 0.06711321944125913, 0.11549352446050926, 0.1466183354071699, 0.27534492165516194],
#           'eval03_DEV011-20010206-XX1830':
#           [0.448326864380132, 0.07452711788392193, 0.2288566427016197, 0.06881494618318533, 0.17924533377668556]}
