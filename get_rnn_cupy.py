#!/usr/bin/env python

import numpy as np
import curbd_cupy as curbd
import pylab
import scipy.io
import time
import math
import neurolib as utils
import os

M1_data = scipy.io.loadmat('../../Data/EN10_d1_all/EN10_d1_imec0_M1_rates.mat')['rates']
"""CPm_data = scipy.io.loadmat('../../Data/EN10_d1_all/EN10_d1_imec0_CPm_rates.mat')['rates']
DN_data = scipy.io.loadmat('../../Data/EN10_d1_all/EN10_d1_imec1_DN_rates.mat')['rates']
CPl_data = scipy.io.loadmat('../../Data/EN10_d1_all/EN10_d1_imec2_CPl_rates.mat')['rates']"""
M2_data = scipy.io.loadmat('../../Data/EN10_d1_all/EN10_d1_imec2_M2_rates.mat')['rates']
"""CPp_data = scipy.io.loadmat('../../Data/EN10_d1_all/EN10_d1_imec3_CPp_rates.mat')['rates']
GPe_data = scipy.io.loadmat('../../Data/EN10_d1_all/EN10_d1_imec3_GPe_rates.mat')['rates']
S1_data = scipy.io.loadmat('../../Data/EN10_d1_all/EN10_d1_imec3_S1_rates.mat')['rates']
GPi_data = scipy.io.loadmat('../../Data/EN10_d1_all/EN10_d1_imec4_GPi_rates.mat')['rates']"""
VAL_data = scipy.io.loadmat('../../Data/EN10_d1_all/EN10_d1_imec5_VAL_rates.mat')['rates']

sign = utils.loadmat('/mnt/nfs/Joseph/Data/EN10/EN10.mat')['data']['sign']
early_loop = utils.loadmat('/mnt/nfs/Joseph/Data/EN10/EN10.mat')['data']['earlyloop']

M1_data = utils.sign_early_loop2(M1_data, sign, early_loop)
"""CPm_data = utils.sign_early_loop2(CPm_data, sign, early_loop)
DN_data = utils.sign_early_loop2(DN_data, sign, early_loop)
CPl_data = utils.sign_early_loop2(CPl_data, sign, early_loop)"""
M2_data = utils.sign_early_loop2(M2_data, sign, early_loop)
"""CPp_data = utils.sign_early_loop2(CPp_data, sign, early_loop)
GPe_data = utils.sign_early_loop2(GPe_data, sign, early_loop)
S1_data = utils.sign_early_loop2(S1_data, sign, early_loop)
GPi_data = utils.sign_early_loop2(GPi_data, sign, early_loop)"""
VAL_data = utils.sign_early_loop2(VAL_data, sign, early_loop)
total_reaches = M1_data.shape[2]
batch_size = 10


for i in range(math.ceil(total_reaches/batch_size)):
    reach_start = i*batch_size

    if reach_start+batch_size < total_reaches:
        reach_end = reach_start+batch_size
        num_trials = batch_size
    else:
        reach_end = total_reaches
        num_trials = total_reaches-reach_start

    dirname = '../Data/three_region/reach_'+str(reach_start)+'-'+str(reach_end-1)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    dirname = '../Data/three_region/reach_'+str(reach_start)+'-'+str(reach_end-1)

    M1_tensor = np.array(M1_data)[:,:,reach_start:reach_end]
    """CPm_tensor = np.array(CPm_data)[:,:,reach_start:reach_end]
    DN_tensor = np.array(DN_data)[:,:,reach_start:reach_end]
    CPl_tensor = np.array(CPl_data)[:,:,reach_start:reach_end]"""
    M2_tensor = np.array(M2_data)[:,:,reach_start:reach_end]
    """CPp_tensor = np.array(CPp_data)[:,:,reach_start:reach_end]
    GPe_tensor = np.array(GPe_data)[:,:,reach_start:reach_end]
    S1_tensor = np.array(S1_data)[:,:,reach_start:reach_end]
    GPi_tensor = np.array(GPi_data)[:,:,reach_start:reach_end]"""
    VAL_tensor = np.array(VAL_data)[:,:,reach_start:reach_end]

    nM1 = len(M1_tensor)
    """nCPm = len(CPm_tensor)
    nDN = len(DN_tensor)
    nCPl = len(CPl_tensor)"""
    nM2 = len(M2_tensor)
    """nCPp = len(CPp_tensor)
    nGPe = len(GPe_tensor)
    nS1 = len(S1_tensor)
    nGPi = len(GPi_tensor)"""
    nVAL = len(VAL_tensor)

    regions = []
    regions.append(['Region M1', np.arange(0, nM1)])
    regions.append(['Region M2', np.arange(nM1, nM1 + nM2)])
    regions.append(['Region VAL', np.arange(nM1 + nM2, nM1 + nM2 + nVAL)])
    """regions.append(['Region CPm', np.arange(nM1, nM1 + nCPm)])
    regions.append(['Region Dn', np.arange(nM1 + nCPm, nM1 + nCPm + nDN)])
    regions.append(['Region CPl', np.arange(nM1 + nCPm + nDN, nM1 + nCPm + nDN + nCPl)])
    regions.append(['Region M2', np.arange(nM1 + nCPm + nDN + nCPl, nM1 + nCPm + nDN + nCPl + nM2)])
    regions.append(['Region CPp', np.arange(nM1 + nCPm + nDN + nCPl + nM2, nM1 + nCPm + nDN + nCPl + nM2 + nCPp)])
    regions.append(['Region GPe', np.arange(nM1 + nCPm + nDN + nCPl + nM2 + nCPp, nM1 + nCPm + nDN + nCPl + nM2 + nCPp + nGPe)])
    regions.append(['Region S1', np.arange(nM1 + nCPm + nDN + nCPl + nM2 + nCPp + nGPe, nM1 + nCPm + nDN + nCPl + nM2 + nCPp + nGPe + nS1)])
    regions.append(['Region GPi', np.arange(nM1 + nCPm + nDN + nCPl + nM2 + nCPp + nGPe + nS1, nM1 + nCPm + nDN + nCPl + nM2 + nCPp + nGPe + nS1 + nGPi)])
    regions.append(['Region VAL', np.arange(nM1 + nCPm + nDN + nCPl + nM2 + nCPp + nGPe + nS1 + nGPi, nM1 + nCPm + nDN + nCPl + nM2 + nCPp + nGPe + nS1 + nGPi + nVAL)])"""

    regions = np.array(regions, dtype=object)
    models = []

    start = time.time()
    for i in range(num_trials):
        cropped1 = M1_tensor[:, :, i]
        """cropped2 = CPm_tensor[:, :, i]
        cropped3 = DN_tensor[:, :, i]
        cropped4 = CPl_tensor[:, :, i]"""
        cropped5 = M2_tensor[:, :, i]
        """cropped6 = CPp_tensor[:, :, i]
        cropped7 = GPe_tensor[:, :, i]
        cropped8 = S1_tensor[:, :, i]
        cropped9 = GPi_tensor[:, :, i]"""
        cropped10 = VAL_tensor[:, :, i]

        #activity = np.concatenate((cropped1, cropped2, cropped3, cropped4, cropped5, cropped6, cropped7, cropped8, cropped9, cropped10), 0)
        activity = np.concatenate((cropped1, cropped5, cropped10), 0)
        model = curbd.trainMultiRegionRNN(activity,
                                            dtData=0.002,
                                            dtFactor=1,
                                            nRunTrain=1000,
                                            nRunFree=10,
                                            tauWN=0.1,
                                            ampInWN=0.005,
                                            regions=regions,
                                            verbose=True,
                                            plotStatus=False)

        pVars = [model['pVars'][i].get() for i in range(len(model['pVars']))]
        chi2s = [model['chi2s'][i].get() for i in range(len(model['chi2s']))]
        newModel = { 'pVars': pVars, 'chi2s': chi2s, 'J': model['J'].get() }
        models.append(newModel)

        if i == 0:
            rnn_data = model['RNN'].get()
            CURBD, CURBD_labels = curbd.computeCURBD(model)
        else:
            rnn_data = np.dstack((rnn_data, model['RNN'].get()))
            tmpCURBD, tmpCURBD_labels = curbd.computeCURBD(model)
            CURBD = np.dstack((CURBD, tmpCURBD))
            CURBD_labels = np.dstack((CURBD_labels, tmpCURBD_labels))

    end = time.time()
    print(end-start)

    """rnn_data = np.split(rnn_data, [nM1,
                                    nM1+nCPm, 
                                    nM1+nCPm+nDN, 
                                    nM1+nCPm+nDN+nCPl, 
                                    nM1+nCPm+nDN+nCPl+nM2, 
                                    nM1+nCPm+nDN+nCPl+nM2+nCPp, 
                                    nM1+nCPm+nDN+nCPl+nM2+nCPp+nGPe, 
                                    nM1+nCPm+nDN+nCPl+nM2+nCPp+nGPe+nS1, 
                                    nM1+nCPm+nDN+nCPl+nM2+nCPp+nGPe+nS1+nGPi])"""
    rnn_data = np.split(rnn_data, [nM1,
                                    nM1+nM2, 
                                    nM1+nM2+nVAL])

    # save simulated RNN data
    scipy.io.savemat(f'{dirname}/M1_rnn_data.mat', mdict={ 'RNN_data': rnn_data[0]})
    scipy.io.savemat(f'{dirname}/M2_rnn_data.mat', mdict={ 'RNN_data': rnn_data[1]})
    scipy.io.savemat(f'{dirname}/VAL_rnn_data.mat', mdict={ 'RNN_data': rnn_data[2]})
    """scipy.io.savemat(f'{dirname}/M1_rnn_data.mat', mdict={ 'RNN_data': rnn_data[0]})
    scipy.io.savemat(f'{dirname}/CPm_rnn_data.mat', mdict={ 'RNN_data': rnn_data[1]})
    scipy.io.savemat(f'{dirname}/DN_rnn_data.mat', mdict={ 'RNN_data': rnn_data[2]})
    scipy.io.savemat(f'{dirname}/CPl_rnn_data.mat', mdict={ 'RNN_data': rnn_data[3]})
    scipy.io.savemat(f'{dirname}/M2_rnn_data.mat', mdict={ 'RNN_data': rnn_data[4]})
    scipy.io.savemat(f'{dirname}/CPp_rnn_data.mat', mdict={ 'RNN_data': rnn_data[5]})
    scipy.io.savemat(f'{dirname}/GPe_rnn_data.mat', mdict={ 'RNN_data': rnn_data[6]})
    scipy.io.savemat(f'{dirname}/S1_rnn_data.mat', mdict={ 'RNN_data': rnn_data[7]})
    scipy.io.savemat(f'{dirname}/GPi_rnn_data.mat', mdict={ 'RNN_data': rnn_data[8]})
    scipy.io.savemat(f'{dirname}/VAL_rnn_data.mat', mdict={ 'RNN_data': rnn_data[9]})"""

    # model includes J matrix, pvars, chi2
    scipy.io.savemat(f'{dirname}/models.mat', mdict={'models': models})

    scipy.io.savemat(f'{dirname}/CURBD.mat', mdict={ 'CURBD': CURBD })

    scipy.io.savemat(f'{dirname}/CURBD_labels.mat', mdict={ 'CURBD_labels': CURBD_labels })
