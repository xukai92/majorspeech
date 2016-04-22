'''
helper functions and global constants for the practical
'''

import os, sys, time


# read file list
SETLIST = ['dev03sub', 'dev03', 'eval03', 'YTBEdevsub', 'YTBEdev', 'YTBEdev-v2', 'YTBEeval']
SHOWLIST = {}
for show_set in SETLIST:
    show_list = []
    f_in = open('/home/kx216/MLSALT11/iSpeech/lib/testlists/{show_set}.lst'.format(show_set=show_set))
    for line in f_in:
        show_list.append(line[:-1])
    f_in.close()
    SHOWLIST[show_set] = show_list

def print_log(log_txt):
    print '[{time}] {log_txt}'.format(
        time=time.ctime(int(time.time())),  # format current time
        log_txt=log_txt
    )

def wait_qsub():
    '''
    wait for all submissions in qstat to be finished
    '''
    while True:
        f = os.popen('qstat | wc -l')
        taks_num = int(f.read()) - 2
        if taks_num <= 0:
            break
        time.sleep(10)
        log_txt = '{taks_num} tasks running in qusb ...'.format(
            time=time.ctime(int(time.time())),  # format current time
            taks_num=taks_num
        )
        print_log(log_txt)
