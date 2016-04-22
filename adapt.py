import os, sys, time


args = sys.argv[1:]
os.system('echo Acoustic Model Adaptation')

# read file list
SETLIST = ['dev03sub', 'dev03', 'YTBEdevsub', 'YTBEdev', 'YTBEeval']
FILELIST = {}
for show_set in SETLIST:
    show_list = []
    f_in = open('/home/kx216/MLSALT11/iSpeech/lib/testlists/{show_set}.lst'.format(show_set=show_set))
    for line in f_in:
        show_list.append(line[:-1])
    f_in.close()
    FILELIST[show_set] = show_list

# command constants
LMRESCORE = './scripts/lmrescore.sh -INSPEN -4.0 -LMSCALE 12.0 {show} lattices decode {lm} plp-int TRUE'
# LMRESCORE = './scripts/lmrescore.sh -INSPEN -10.0 -LMSCALE 12.0 {show} lattices decode {lm} plp-test4 FALSE' # testing
MERGELATS = './scripts/mergelats.sh {show} plp-int rescore plp-int'
HMMRESCORE = './scripts/hmmrescore.sh {show} plp-int merge {smodel} {smodeltype}'
HMMADAPT = './scripts/hmmadapt.sh -OUTPASS {adapt} {show} {smodel} decode {amodel} {modeltype}'
ADAPTRESCORE = './scripts/hmmrescore.sh -ADAPT {amodel} {adapt} -OUTPASS {decode} {show} plp-int merge {amodel} {modeltype}'

# model constants
MODELS = ['plp-int', 'grph-plp-int', 'tandem-int', 'grph-tandem-int', 'hybrid-int']
MODELTYPES = ['plp', 'grph-plp', 'tandem', 'grph-tandem', 'hybrid']
SUBMODELS = ['plp-int', 'tandem-int', 'grph-tandem-int']
SUBMODELTYPES = ['plp', 'tandem', 'grph-tandem']


def qstatwait():
    '''
    wait for all submissions in qstat to be finished
    '''
    finished = False
    while not finished:
        time.sleep(10)
        f = os.popen('qstat | wc -l')
        n = int(f.read())
        print '[tick] n = {n}'.format(n=n)
        if n == 0:
            finished = True


# adaptation
# lm = 'lm_int_YTB'
lm = 'lm_int'

# test
showset = 'YTBEeval'
print 'working on show set: {showset}'.format(showset=showset)

for show in FILELIST[showset]:
    cmd = LMRESCORE.format(show=show,
                           lm=lm)
    print 'Running command:\n  {cmd}'.format(cmd=cmd)
    os.system(cmd)

qstatwait()

for show in FILELIST[showset]:
    cmd = MERGELATS.format(show=show)
    print 'Running command:\n  {cmd}'.format(cmd=cmd)
    os.system(cmd)

qstatwait()

for smodel, smodeltype in zip(MODELS, MODELTYPES):
    for show in FILELIST[showset]:
        cmd = HMMRESCORE.format(show=show,
                                smodel=smodel,
                                smodeltype=smodeltype)
        print 'Running command:\n  {cmd}'.format(cmd=cmd)
        os.system(cmd)

qstatwait()

for model, modeltype in zip(SUBMODELS, SUBMODELTYPES):
    # for smodel, smodeltype in zip(MODELS, MODELTYPES):
    for smodel, smodeltype in zip(['hybrid-int'], ['hybrid']):
        adapt = 'adapt-{smodeltype}'.format(smodeltype=smodeltype)
        amodel = '{modeltype}-adapt-int'.format(modeltype=modeltype)
        for show in FILELIST[showset]:
            cmd = HMMADAPT.format(adapt=adapt,
                                  show=show,
                                  smodel=smodel,
                                  amodel=amodel,
                                  modeltype=modeltype)
            print 'Running command:\n  {cmd}'.format(cmd=cmd)
            os.system(cmd)

qstatwait()

for model, modeltype in zip(SUBMODELS, SUBMODELTYPES):
    # for smodel, smodeltype in zip(MODELS, MODELTYPES):
    for smodel, smodeltype in zip(['hybrid-int'], ['hybrid']):
        amodel = '{modeltype}-adapt-int'.format(modeltype=modeltype)
        adapt = 'adapt-{smodeltype}'.format(smodeltype=smodeltype)
        decode = 'decode-{smodeltype}'.format(smodeltype=smodeltype)
        for show in FILELIST[showset]:
            cmd = ADAPTRESCORE.format(amodel=amodel,
                                      show=show,
                                      model=model,
                                      modeltype=modeltype,
                                      adapt=adapt,
                                      decode=decode)
            print 'Running command:\n  {cmd}'.format(cmd=cmd)
            os.system(cmd)
