import numpy as np
import re

def sub_parameters(model, param_dict, log=True, KDs=True, generic_kf=1.5e4, verbose=False):
    generic_kf_log = np.log10(generic_kf)
    if KDs == True:
        for param, value in param_dict.items():
            if 'KD' in param:
                x = re.compile('(KD)(\S*)', re.IGNORECASE)
                y = x.search(param)
                param_name = y.groups()[1]
                kf_param_name = 'kf' + param_name
                kr_param_name = 'kr' + param_name
                if log == True:
                    model.parameters[kf_param_name].value = 10 ** generic_kf_log
                    model.parameters[kr_param_name].value = 10 ** (value + generic_kf_log)
                    if verbose:
                        print 'Changed parameter ' + str(kf_param_name) + ' to ' + str(10 ** generic_kf_log)
                        print 'Changed parameter ' + str(kr_param_name) + ' to ' + str(10 ** (value + generic_kf_log))

                else:
                    model.parameters[kf_param_name].value = generic_kf
                    model.parameters[kr_param_name].value = value * generic_kf

                    if verbose:
                        print 'Changed parameter ' + str(kf_param_name) + 'to ' + str(generic_kf)
                        print 'Changed parameter ' + str(kr_param_name) + ' to ' + str(value * generic_kf)
            else:
                if log == True:
                    model.parameters[param].value = 10 ** value
                    if verbose:
                        print 'Changed parameter ' + str(param) + ' to ' + str(10 ** value)
                else:
                    model.parameters[param].value = value
                    if verbose:
                        print 'Changed parameter ' + str(param) + ' to ' + str(value)
    else:
        for param, value in param_dict.items():
            if log == True:
                model.parameters[param].value = 10 ** value
                if verbose:
                    print 'Changed parameter ' + str(param) + ' to ' + str(10 ** value)
            else:
                model.parameters[param].value = value
                if verbose:
                    print 'Changed parameter ' + str(param) + ' to ' + str(value)
