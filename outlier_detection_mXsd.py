'''
This code calculates the number of traces outisde of the envelope, which is between
median+m*SD and median-m*SD. To do this, first we need to get m that corresponds to the threshold we define (such as FPR = 1%)
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import csv
import statistics

plt.close('all')
out_list = []

plt.rcParams['font.family'] ='sans-serif'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

'''
#Jihun Local
out_list.append("C:\\Users\\Kwon\\Documents\\Panc_OCM\\Subject_01_20180928\\run1.npy") #Before water
out_list.append("C:\\Users\\Kwon\\Documents\\Panc_OCM\\Subject_01_20180928\\run2.npy") #After water
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
out_list.append("C:\\Users\\Kwon\\Documents\\Panc_OCM\\Subject_01_20181102\\run1.npy")
out_list.append("C:\\Users\\Kwon\\Documents\\Panc_OCM\\Subject_01_20181102\\run2.npy")
sr_list = ['s1r2']
rep_list = [4000]


tole = 0.01  # What FPR you want to get for test set.

num_train = 1
num_test = 10 - num_train
num_ocm = 3
num_bh = 5 # number of bh in each state
str_norm = '_norm'


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
        median0_base = np.zeros([s])  # median of filtered signal
        median1_base = np.zeros([s])
        median2_base = np.zeros([s])
        sd0 = np.zeros([s])  # sd of (median - train)
        sd1 = np.zeros([s])
        sd2 = np.zeros([s])
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
    offset0 = np.zeros([s])
    offset1 = np.zeros([s])
    offset2 = np.zeros([s])
    ocm0_off = np.zeros([s, c0_new])  # offset corrected OCM
    ocm1_off = np.zeros([s, c0_new])
    ocm2_off = np.zeros([s, c0_new])
    ocm0_norm_off = np.zeros([s, c0_new])  # offset corrected normalized OCM
    ocm1_norm_off = np.zeros([s, c0_new])
    ocm2_norm_off = np.zeros([s, c0_new])

    #### Normalization ####
    if str_norm is '_norm':
        for p in range(0, c0_new):
            # Get absolute value and normalize
            ocm0_norm[:, p] = np.divide(np.abs(ocm0[:, p]), np.max(np.abs(ocm0[:, p])))
            ocm1_norm[:, p] = np.divide(np.abs(ocm1[:, p]), np.max(np.abs(ocm1[:, p])))
            ocm2_norm[:, p] = np.divide(np.abs(ocm2[:, p]), np.max(np.abs(ocm2[:, p])))
            tr0 = ocm0_norm[:, p]
            tr1 = ocm1_norm[:, p]
            tr2 = ocm2_norm[:, p]
            offset0 = signal.detrend(tr0)
            offset1 = signal.detrend(tr1)
            offset2 = signal.detrend(tr2)
            ocm0_off[:, p] = offset0
            ocm1_off[:, p] = offset1
            ocm2_off[:, p] = offset2
            ocm0_norm_off[:, p] = np.divide(np.abs(ocm0_off[:, p]), np.max(np.abs(ocm0_off[:, p])))
            ocm1_norm_off[:, p] = np.divide(np.abs(ocm1_off[:, p]), np.max(np.abs(ocm1_off[:, p])))
            ocm2_norm_off[:, p] = np.divide(np.abs(ocm2_off[:, p]), np.max(np.abs(ocm2_off[:, p])))
        for bh in range(0, num_bh):
            for depth in range(0, s):
                # When want to use normalized
                median0[depth, bh] = statistics.median(ocm0_norm[depth, bh*t_sub:(bh+1)*t_sub])
                median1[depth, bh] = statistics.median(ocm1_norm[depth, bh*t_sub:(bh+1)*t_sub])
                median2[depth, bh] = statistics.median(ocm2_norm[depth, bh*t_sub:(bh+1)*t_sub])
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
        #### Normalization ####
        # Median-based filtering
        for bh in range(0, num_bh):
            for depth in range(0, s):
                # get median for each bh
                median0[depth, bh] = statistics.median(ocm0[depth, bh*t_sub:(bh+1)*t_sub])
                median1[depth, bh] = statistics.median(ocm1[depth, bh*t_sub:(bh+1)*t_sub])
                median2[depth, bh] = statistics.median(ocm2[depth, bh*t_sub:(bh+1)*t_sub])
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
    if fidx % 2 == 0:
        # Median-based filtering
        for depth in range(0, s):
            # Calculate median of baseline signal
            median0_base[depth] = statistics.median(ocm0_filt[depth, 0:t_sub*num_train])
            median1_base[depth] = statistics.median(ocm1_filt[depth, 0:t_sub*num_train])
            median2_base[depth] = statistics.median(ocm2_filt[depth, 0:t_sub*num_train])
            # Get SD of (Median - train)
            sd0[depth] = np.std(median0_base[depth] - ocm0_filt[depth, 0:t_sub*num_train])
            sd1[depth] = np.std(median1_base[depth] - ocm1_filt[depth, 0:t_sub*num_train])
            sd2[depth] = np.std(median2_base[depth] - ocm2_filt[depth, 0:t_sub*num_train])

        #### Get parameter m ####
        # m and OoE (out of envelop) distribution
        m_max = 100
        scale = 10  # number divides m
        count0 = np.zeros([m_max])
        count1 = np.zeros([m_max])
        count2 = np.zeros([m_max])
        flag0_m = 0
        flag1_m = 0
        flag2_m = 0
        m0=0
        m1=0
        m2=0

        for m in range(0, m_max):
            if m % 10 == 0:
                print('m:', m)
            thr0[:] = np.abs(median0_base[:]) + m / scale * sd0[:]
            thr1[:] = np.abs(median1_base[:]) + m / scale * sd1[:]
            thr2[:] = np.abs(median2_base[:]) + m / scale * sd2[:]
            # loop inside the training set
            for p in range(0, t_sub*num_train):
                flag0 = 0
                flag1 = 0
                flag2 = 0
                for depth in range(0, s):
                    # if not detected yet
                    if flag0 < 1:  # OCM0
                        # check every depth and count if it's larger than the threshold
                        if ocm0_filt[depth, p] > thr0[depth]:
                            count0[m] = count0[m] + 1
                            flag0 = 1
                    if flag1 < 1:  # OCM1
                        if ocm1_filt[depth, p] > thr1[depth]:
                            count1[m] = count1[m] + 1
                            flag1 = 1
                    if flag2 < 1:  # OCM2
                        if ocm2_filt[depth, p] > thr2[depth]:
                            count2[m] = count2[m] + 1
                            flag2 = 1

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


    '''
        # ========================Visualize==============================================
        d = np.linspace(2.3, 4.9, s)  # My Depth
        fig = plt.figure(figsize=(7, 4))
        # OCM1
        ax1 = fig.add_subplot(111)
        a0 = ax1.plot(d, ocm1_filt[:, 0], linewidth=1, label="0")
        a1 = ax1.plot(d, ocm1_filt[:, 10], linewidth=1, label="10")
        a2 = ax1.plot(d, ocm1_filt[:, 20], linewidth=1, label="20")
        a3 = ax1.plot(d, ocm1_filt[:, 30], linewidth=1, label="30")
        a4 = ax1.plot(d, ocm1_filt[:, 40], linewidth=1, label="40")
        a5 = ax1.plot(d, median1_base[:], 'g', linewidth=1, label="median")
        a6 = ax1.plot(d, median1_base[:]+sd1[:], 'b', linewidth=1, label="med+sd")
        ax1.set_title('Distribution of parameter m')
        ax1.set_xlabel('Depth')
        ax1.set_ylabel('Intensity')
        plt.legend(loc='upper right')
        fig.show()
        f_name = 'Signal_check_' + Sub_run + '.png'
        plt.savefig(f_name)
        # ========================Visualize==============================================
        m_ = np.linspace(0, m_max/scale, m_max)  # My m
        fig = plt.figure(figsize=(7, 4))
        # OCM1
        ax1 = fig.add_subplot(111)
        a0 = ax1.plot(m_, count0[:], 'r', linewidth=1, label="count0")
        a1 = ax1.plot(m_, count1[:], 'g', linewidth=1, label="count1")
        a2 = ax1.plot(m_, count2[:], 'b', linewidth=1, label="count2")
        ax1.set_title('Distribution of parameter m')
        ax1.set_xlabel('m')
        ax1.set_ylabel('Number of traces above threshold')
        plt.legend(loc='upper right')
        fig.show()
        f_name = 'm_dist_' + Sub_run + '.png'
        plt.savefig(f_name)
    '''