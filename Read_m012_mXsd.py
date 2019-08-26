'''
This code calculates the number of traces outisde of the envelope, which is between
median+m*SD and median-m*SD. To do this, first we need to get m that corresponds to the threshold we define (such as FPR = 1%)
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import csv
import statistics
import pickle

plt.close('all')
out_list = []

plt.rcParams['font.family'] ='sans-serif'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
'''
out_list.append("C:\\Users\\Kwon\\Documents\\Panc_OCM\\Subject_01_20180928\\run1.npy")  #Before water
out_list.append("C:\\Users\\Kwon\\Documents\\Panc_OCM\\Subject_01_20180928\\run2.npy")  #After water
out_list.append("C:\\Users\\Kwon\\Documents\\Panc_OCM\\Subject_01_20181102\\run1.npy")
out_list.append("C:\\Users\\Kwon\\Documents\\Panc_OCM\\Subject_01_20181102\\run2.npy")
out_list.append("C:\\Users\\Kwon\\Documents\\Panc_OCM\\Subject_02_20181102\\run1.npy")
out_list.append("C:\\Users\\Kwon\\Documents\\Panc_OCM\\Subject_02_20181102\\run2.npy")
out_list.append("C:\\Users\\Kwon\\Documents\\Panc_OCM\\Subject_02_20181220\\run1.npy")
out_list.append("C:\\Users\\Kwon\\Documents\\Panc_OCM\\Subject_02_20181220\\run2.npy")
out_list.append("C:\\Users\\Kwon\\Documents\\Panc_OCM\\Subject_03_20190228\\run1.npy")
out_list.append("C:\\Users\\Kwon\\Documents\\Panc_OCM\\Subject_03_20190228\\run2.npy")
out_list.append("C:\\Users\\Kwon\\Documents\\Panc_OCM\\Subject_03_20190320\\run1.npy")
out_list.append("C:\\Users\\Kwon\\Documents\\Panc_OCM\\Subject_03_20190320\\run2.npy")
sr_list = ['s1r1', 's1r2', 's2r1', 's2r2', 's3r1', 's3r2']
rep_list = [8196, 8196, 8192, 8192, 6932, 6932, 3690, 3690, 3401, 3401, 3690, 3690]

'''
out_list.append("C:\\Users\\Kwon\\Documents\\Panc_OCM\\Subject_01_20180928\\run1.npy")  #Before water
out_list.append("C:\\Users\\Kwon\\Documents\\Panc_OCM\\Subject_01_20180928\\run2.npy")  #After water
sr_list = ['s1r1', 's1r1']
rep_list = [8192, 8192]


tole = 0.01  # What FPR you want to get for test set.

num_train = 2
num_test = 10 - num_train
num_ocm = 3
num_bh = 5 # number of bh in each state
str_norm = '_norm'
#str_name = '_unnorm'

for fidx in range(0,np.size(rep_list)):
    Sub_run = sr_list[fidx]
    plt.rcParams["font.size"] = 11
    in_filename = out_list[fidx]
    ocm = np.load(in_filename)

    #crop data
    ocm = ocm[300:650, :]
    s, t = np.shape(ocm)

    # variables initialization
    median0 = np.zeros([s, num_bh])  # median
    median1 = np.zeros([s, num_bh])
    median2 = np.zeros([s, num_bh])
    if fidx % 2 == 0:
        thr0 = np.zeros([s])  # threshold
        thr1 = np.zeros([s])
        thr2 = np.zeros([s])
    out0_test = np.zeros([num_bh])  # output test result
    out1_test = np.zeros([num_bh])
    out2_test = np.zeros([num_bh])

    # divide the data into each OCM and store absolute value
    b = np.linspace(0, t-1, t)
    b0 = np.mod(b,4) == 0
    ocm0 = ocm[:, b0]
    b1 = np.mod(b,4) == 1
    ocm1 = ocm[:, b1]
    b2 = np.mod(b,4) == 2
    ocm2 = ocm[:, b2]
    s, c0 = np.shape(ocm0)
    print('ocm0:', ocm0.shape)

    # first few traces are strange. Remove them
    ocm0 = ocm0[:, c0 - num_bh*rep_list[fidx]:]
    ocm1 = ocm1[:, c0 - num_bh*rep_list[fidx]:]
    ocm2 = ocm2[:, c0 - num_bh*rep_list[fidx]:]

    print('ocm0 new:', ocm0.shape)
    s, c0_new = np.shape(ocm0)
    t_sub = int(c0_new / num_bh)

    ocm0_filt = np.zeros([s, c0_new])  # filtered signal (median based filtering)
    ocm1_filt = np.zeros([s, c0_new])
    ocm2_filt = np.zeros([s, c0_new])
    ocm0_norm = np.zeros([s, c0_new])  # Absolute and Normalized
    ocm1_norm = np.zeros([s, c0_new])
    ocm2_norm = np.zeros([s, c0_new])

    #### Depending on norm or unnorm, change path
    if str_norm is '_norm':
        # Normalization
        for p in range(0, c0_new):
            # Get absolute value and normalize
            ocm0_norm[:, p] = np.divide(np.abs(ocm0[:, p]), np.max(np.abs(ocm0[:, p])))
            ocm1_norm[:, p] = np.divide(np.abs(ocm1[:, p]), np.max(np.abs(ocm1[:, p])))
            ocm2_norm[:, p] = np.divide(np.abs(ocm2[:, p]), np.max(np.abs(ocm2[:, p])))
        for bh in range(0, num_bh):
            for depth in range(0, s):
                # When want to use normalized
                median0[depth, bh] = statistics.median(ocm0_norm[depth, bh * t_sub:(bh + 1) * t_sub])
                median1[depth, bh] = statistics.median(ocm1_norm[depth, bh * t_sub:(bh + 1) * t_sub])
                median2[depth, bh] = statistics.median(ocm2_norm[depth, bh * t_sub:(bh + 1) * t_sub])
        # Median-based filtering
        bh = -1
        for p in range(0, c0_new):
            if p % rep_list[fidx] == 0:
                bh = bh + 1
            # filtering all traces with median trace
            for depth in range(0, s):
                # filter the signal (subtract median from each trace of corresponding bh)
                ocm0_filt[depth, p] = np.abs(ocm0_norm[depth, p] - median0[depth, bh])
                ocm1_filt[depth, p] = np.abs(ocm1_norm[depth, p] - median1[depth, bh])
                ocm2_filt[depth, p] = np.abs(ocm2_norm[depth, p] - median2[depth, bh])

    elif str_norm is '_unnorm':
        # Normalization
        # Median-based filtering
        for bh in range(0, num_bh):
            for depth in range(0, s):
                # get median for each bh
                median0[depth, bh] = statistics.median(ocm0[depth, bh * t_sub:(bh + 1) * t_sub])
                median1[depth, bh] = statistics.median(ocm1[depth, bh * t_sub:(bh + 1) * t_sub])
                median2[depth, bh] = statistics.median(ocm2[depth, bh * t_sub:(bh + 1) * t_sub])
        # filtering all traces with median trace
        bh = -1
        for p in range(0, c0_new):
            if p % rep_list[fidx] == 0:
                bh = bh + 1
            for depth in range(0, s):
                # filter the signal (subtract median from each trace of corresponding bh)
                ocm0_filt[depth, p] = np.abs(ocm0[depth, p] - median0[depth, bh])
                ocm1_filt[depth, p] = np.abs(ocm1[depth, p] - median1[depth, bh])
                ocm2_filt[depth, p] = np.abs(ocm2[depth, p] - median2[depth, bh])


    #### Threshold generation ####
    # if state 1
    print('start searching for m')
    if fidx % 2 == 0:
        #### Get parameter m ####
        # m and OoE (out of envelop) distribution
        m_max = 100
        scale = 10  # number divides m
        fname = 'm012_' + str(Sub_run) + '_max' + str(m_max) + '_scale' + str(scale) + '_train' + str(num_train) + '_tole'+ str(tole) + str_norm + '.pkl'
        print('Reading file: ', fname)
        #fname = 'm012_s1r2_max100_scale10_train1_tole0.01_test.pkl'
        with open(fname, 'rb') as f:
            count0, count1, count2, median0_base, median1_base, median2_base, sd0, sd1, sd2 = pickle.load(f)

        flag0_m = 0
        flag1_m = 0
        flag2_m = 0
        m0=0
        m1=0
        m2=0

        for m in range(0, m_max):
            # Get m if EoF is below predefined FPR
            if (count0[m] < rep_list[fidx] * tole) and (flag0_m == 0):
                m0 = m / scale
                flag0_m = 1
            if (count1[m] < rep_list[fidx] * tole) and (flag1_m == 0):
                m1 = m / scale
                flag1_m = 1
            if (count2[m] < rep_list[fidx] * tole) and (flag2_m == 0):
                m2 = m / scale
                flag2_m = 1

        print('m0:', m0, 'm1:', m1, 'm2:', m2)
        print('count0:', count0[int(m0*10)], 'count1:', count1[int(m1*10)], 'count2:', count2[int(m2*10)])

    #### Performance Evaluation with remaining data####
    # set threshold
    thr0[:] = np.abs(median0_base[:]) + m0 * sd0[:]
    thr1[:] = np.abs(median1_base[:]) + m1 * sd1[:]
    thr2[:] = np.abs(median2_base[:]) + m2 * sd2[:]
    print('thr0:', thr0[0:20])
    print('sd0:', sd0[0:20])
    print('m0:', m0)
    print('median0_base:', median0_base[0:20])

    # ========================Visualize==============================================
    d = np.linspace(2.3, 4.9, s)  # My Depth
    fig = plt.figure(figsize=(7, 4))
    # OCM1
    ax1 = fig.add_subplot(111)
    a0 = ax1.plot(d, ocm0_filt[:,0], linewidth=1, label="filt")
    a0 = ax1.plot(d, ocm0_filt[:,500], linewidth=1, label="filt")
    a0 = ax1.plot(d, ocm0_filt[:,1000], linewidth=1, label="filt")
    a0 = ax1.plot(d, ocm0_filt[:,1500], linewidth=1, label="filt")
    a0 = ax1.plot(d, ocm0_filt[:,2000], linewidth=1, label="filt")
    a0 = ax1.plot(d, ocm0_filt[:,2500], linewidth=1, label="filt")
    a0 = ax1.plot(d, ocm0_filt[:,3000], linewidth=1, label="filt")
    a0 = ax1.plot(d, ocm0_filt[:,3500], linewidth=1, label="filt")
    a0 = ax1.plot(d, ocm0_filt[:,4000], linewidth=1, label="filt")
    a0 = ax1.plot(d, ocm0_filt[:,4500], linewidth=1, label="filt")
    a0 = ax1.plot(d, median0_base[:], color='black', linewidth=2, label="0")
    a0 = ax1.plot(d, median0_base[:] + sd0[:], color='black', linewidth=1, label="0")
    ax1.set_title('Distribution of parameter m')
    ax1.set_xlabel('Depth')
    ax1.set_ylabel('Intensity')
    plt.legend(loc='upper right')
    fig.show()
    f_name = 'Signal_check_' + Sub_run + '.png'
    plt.savefig(f_name)
    # ========================Visualize==============================================


    # Check test data
    for bh in range(0, num_bh):
        print('bh', bh, ' starting')
        for p in range(0, t_sub):
            flag0 = 0
            flag1 = 0
            flag2 = 0
            for depth in range(0, s):
                # if not detected yet
                if flag0 < 1:  # OCM0
                    # check every depth and count if it's larger than the threshold
                    if ocm0_filt[depth, bh*t_sub+p] > thr0[depth]:
                        out0_test[bh] = out0_test[bh] + 1
                        flag0 = 1
                if flag1 < 1:  # OCM1
                    if ocm1_filt[depth, bh*t_sub+p] > thr1[depth]:
                        out1_test[bh] = out1_test[bh] + 1
                        flag1 = 1
                if flag2 < 1:  # OCM2
                    if ocm2_filt[depth, bh*t_sub+p] > thr2[depth]:
                        out2_test[bh] = out2_test[bh] + 1
                        flag2 = 1

    # if state 1, bh=1 (baseline) has to be subtracted
    total = 100 / rep_list[fidx]
    if fidx % 2 == 0:
        print('### ', Sub_run, ' Results ###')
        print('OCM0 number:', 'bh1:', out0_test[0], 'bh2:', out0_test[1], 'bh3:', out0_test[2], 'bh4:', out0_test[3], 'bh5:', out0_test[4])
        print('OCM1 number:', 'bh1:', out1_test[0], 'bh2:', out1_test[1], 'bh3:', out1_test[2], 'bh4:', out1_test[3], 'bh5:', out1_test[4])
        print('OCM2 number:', 'bh1:', out2_test[0], 'bh2:', out2_test[1], 'bh3:', out2_test[2], 'bh4:', out2_test[3], 'bh5:', out2_test[4])
        print('OCM0 rate:', 'bh1:', '{:.3f}'.format(out0_test[0] * total), 'bh2:', '{:.3f}'.format(out0_test[1] * total),
              'bh3:', '{:.3f}'.format(out0_test[2] * total), 'bh4:', '{:.3f}'.format(out0_test[3] * total), 'bh5:', '{:.3f}'.format(out0_test[4] * total))
        print('OCM1 rate:', 'bh1:', '{:.3f}'.format(out1_test[0] * total), 'bh2:', '{:.3f}'.format(out1_test[1] * total),
              'bh3:', '{:.3f}'.format(out1_test[2] * total), 'bh4:', '{:.3f}'.format(out1_test[3] * total), 'bh5:', '{:.3f}'.format(out1_test[4] * total))
        print('OCM2 rate:', 'bh1:', '{:.3f}'.format(out2_test[0] * total), 'bh2:', '{:.3f}'.format(out2_test[1] * total),
              'bh3:', '{:.3f}'.format(out2_test[2] * total), 'bh4:', '{:.3f}'.format(out2_test[3] * total), 'bh5:', '{:.3f}'.format(out2_test[4] * total))
    else:
        print('OCM0 number:', 'bh6:', out0_test[0], 'bh7:', out0_test[1], 'bh8:', out0_test[2], 'bh9:', out0_test[3], 'bh10:', out0_test[4])
        print('OCM1 number:', 'bh6:', out1_test[0], 'bh7:', out1_test[1], 'bh8:', out1_test[2], 'bh9:', out1_test[3], 'bh10:', out1_test[4])
        print('OCM2 number:', 'bh6:', out2_test[0], 'bh7:', out2_test[1], 'bh8:', out2_test[2], 'bh9:', out2_test[3], 'bh10:', out2_test[4])
        print('OCM0 rate:', 'bh6:', '{:.3f}'.format(out0_test[0] * total), 'bh7:', '{:.3f}'.format(out0_test[1] * total), 'bh8:', '{:.3f}'.format(out0_test[2] * total), 'bh9:', '{:.3f}'.format(out0_test[3] * total), 'bh10:', '{:.3f}'.format(out0_test[4] * total))
        print('OCM1 rate:', 'bh6:', '{:.3f}'.format(out1_test[0] * total), 'bh7:', '{:.3f}'.format(out1_test[1] * total), 'bh8:', '{:.3f}'.format(out1_test[2] * total), 'bh9:', '{:.3f}'.format(out1_test[3] * total), 'bh10:', '{:.3f}'.format(out1_test[4] * total))
        print('OCM2 rate:', 'bh6:', '{:.3f}'.format(out2_test[0] * total), 'bh7:', '{:.3f}'.format(out2_test[1] * total), 'bh8:', '{:.3f}'.format(out2_test[2] * total), 'bh9:', '{:.3f}'.format(out2_test[3] * total), 'bh10:', '{:.3f}'.format(out2_test[4] * total))

