import os, sys, time
import helper as h


# command constants
LMRESCORE = './scripts/lmrescore.sh {show} lattices decode {lm} plp-int TRUE'
MERGELATS = './scripts/mergelats.sh {show} plp-int rescore plp-int'
HMMRESCORE = './scripts/hmmrescore.sh {show} plp-int merge {smodel} {smodeltype}'
HMMADAPT = './scripts/hmmadapt.sh -OUTPASS {adapt} {show} {smodel} decode {amodel} {modeltype}'
ADAPTRESCORE = './scripts/hmmrescore.sh -ADAPT {amodel} {adapt} -OUTPASS {decode} {show} plp-int merge {amodel} {modeltype}'

# model constants
MODELS = ['plp-int', 'grph-plp-int', 'tandem-int', 'grph-tandem-int', 'hybrid-int']
MODELTYPES = ['plp', 'grph-plp', 'tandem', 'grph-tandem', 'hybrid']
SUBMODELS = ['plp-int', 'tandem-int', 'grph-tandem-int']    # subset of models for cross-adaptation
SUBMODELTYPES = ['plp', 'tandem', 'grph-tandem']            # subset of model types for cross-adaptation

def main():
    print 'Acoustic Model Adaptation'
    lm = 'my_lms/lm_int_dev03'          # LM to be used
    showset = 'dev03'
    print 'working on show set: {showset}'.format(showset=showset)

    # rescore using the interpolated language model
    for show in h.SHOWLIST[showset]:
        cmd = LMRESCORE.format(
            show=show,
            lm=lm
        )

        # print log
        log_txt = 'Running command:\n  {cmd}'.format(cmd=cmd)
        h.print_log(log_txt)

        os.system(cmd)

    h.wait_qsub()

    # merge lattices
    for show in h.SHOWLIST[showset]:
        cmd = MERGELATS.format(show=show)

        # print log
        log_txt = 'Running command:\n  {cmd}'.format(cmd=cmd)
        h.print_log(log_txt)

        os.system(cmd)

    h.wait_qsub()

    # rescore using acoustic models
    for smodel, smodeltype in zip(MODELS, MODELTYPES):
        for show in h.SHOWLIST[showset]:
            cmd = HMMRESCORE.format(
                show=show,
                smodel=smodel,
                smodeltype=smodeltype
            )

            # print log
            log_txt = 'Running command:\n  {cmd}'.format(cmd=cmd)
            h.print_log(log_txt)

            os.system(cmd)

    h.wait_qsub()

    # generate transformations
    for model, modeltype in zip(SUBMODELS, SUBMODELTYPES):
        for smodel, smodeltype in zip(MODELS, MODELTYPES):
        # for smodel, smodeltype in zip(['hybrid-int'], ['hybrid']):
            adapt = 'adapt-{smodeltype}'.format(smodeltype=smodeltype)
            amodel = '{modeltype}-adapt-int'.format(modeltype=modeltype)
            for show in h.SHOWLIST[showset]:
                cmd = HMMADAPT.format(
                    adapt=adapt,
                    show=show,
                    model=smodel,
                    amodel=amodel,
                    modeltype=modeltype
                )

                # print log
                log_txt = 'Running command:\n  {cmd}'.format(cmd=cmd)
                h.print_log(log_txt)

                os.system(cmd)

    h.wait_qsub()

    # apply transformations
    for model, modeltype in zip(SUBMODELS, SUBMODELTYPES):
        for smodel, smodeltype in zip(MODELS, MODELTYPES):
        # for smodel, smodeltype in zip(['hybrid-int'], ['hybrid']):
            amodel = '{modeltype}-adapt-int'.format(modeltype=modeltype)
            adapt = 'adapt-{smodeltype}'.format(smodeltype=smodeltype)
            decode = 'decode-{smodeltype}'.format(smodeltype=smodeltype)
            for show in h.SHOWLIST[showset]:
                cmd = ADAPTRESCORE.format(
                    amodel=amodel,
                    show=show,
                    model=model,
                    modeltype=modeltype,
                    adapt=adapt,
                    decode=decode
                )

                # print log
                log_txt = 'Running command:\n  {cmd}'.format(cmd=cmd)
                h.print_log(log_txt)

                os.system(cmd)

if __name__ == '__main__':
    main()
