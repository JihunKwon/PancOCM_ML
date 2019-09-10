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
from trace_outlier_check import outlier_remove

plt.close('all')
out_list = []

plt.rcParams['font.family'] ='sans-serif'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

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

#sr_list = ['s1r1', 's1r1', 's1r2', 's1r2']
#rep_list = [8196, 8196, 8192, 8192]
sr_list = ['s1r1', 's1r1', 's1r2', 's1r2', 's2r1', 's2r1', 's2r2', 's2r2', 's3r1', 's3r1', 's3r2', 's3r2']
rep_list = [8196, 8196, 8192, 8192, 6932, 6932, 3690, 3690, 3401, 3401, 3690, 3690]
#rep_list = [200, 200, 200, 100, 100, 100, 100, 3690, 3401, 3401, 3690, 3690]

'''
out_list.append("C:\\Users\\Kwon\\Documents\\Panc_OCM\\Subject_01_20180928\\run1.npy") #Before water
out_list.append("C:\\Users\\Kwon\\Documents\\Panc_OCM\\Subject_01_20180928\\run2.npy") #After water
sr_list = ['s1r1']
rep_list = [100]
'''

num_train = 3
num_test = 10 - num_train
num_ocm = 3
num_bh = 5 # number of bh in each state
bin = 100
scale = 10  # number divides m

for fidx in range(0, np.size(rep_list)):
    if fidx%2 == 0:
        Sub_run_name = sr_list[fidx]
        print('Status: train'+str(num_train)+'_'+Sub_run_name)
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

        # Manually remove outlier. (OCM0 contains about 0.5% of outlier)
        ocm0_new = outlier_remove(Sub_run_name, c0_new, ocm0)
        s, c0_new_removed = np.shape(ocm0_new)
        t_sub_removed = int(c0_new_removed / num_bh)
        dif = ocm0.shape[1]-ocm0_new.shape[1]

        print('ocm0 new_removed:', ocm0_new.shape)
        print(str(dif), 'was removed. It is', '{:.2f}'.format(dif*100/ocm0.shape[1]), '%')

        ocm0_filt = np.zeros([s, c0_new_removed])  # filtered signal (median based filtering)
        ocm1_filt = np.zeros([s, c0_new])
        ocm2_filt = np.zeros([s, c0_new])
        ocm0_low = np.zeros([s, c0_new_removed])  # low pass
        ocm1_low = np.zeros([s, c0_new])
        ocm2_low = np.zeros([s, c0_new])
        f1 = np.ones([10])  # low pass kernel

        # Median-based filtering
        for bh in range(0, num_bh):
            for depth in range(0, s):
                # get median for each bh
                median0[depth, bh] = statistics.median(ocm0_new[depth, bh * t_sub_removed:(bh + 1) * t_sub_removed])
                median1[depth, bh] = statistics.median(ocm1[depth, bh * t_sub:(bh + 1) * t_sub])
                median2[depth, bh] = statistics.median(ocm2[depth, bh * t_sub:(bh + 1) * t_sub])
        # filtering all traces with median trace
        # The size of ocm0 is different with ocm1 and ocm2. Has to be filtered separately.
        ## OCM0
        bh = -1
        for p in range(0, c0_new_removed):
            # have to consider the number of traces removed
            if p % rep_list[fidx] - dif//num_bh == 0:
                bh = bh + 1
            for depth in range(0, s):
                # filter the signal (subtract median from each trace of corresponding bh)
                ocm0_filt[depth, p] = ocm0_new[depth, p] - median0[depth, bh]
            tr0 = ocm0_filt[:, p]
            ocm0_low[:, p] = np.convolve(np.sqrt(np.square(tr0)), f1, 'same')

        ## OCM1 and 2
        bh = -1
        for p in range(0, c0_new):
            if p % rep_list[fidx] == 0:
                bh = bh + 1
            for depth in range(0, s):
                # filter the signal (subtract median from each trace of corresponding bh)
                ocm1_filt[depth, p] = ocm1[depth, p] - median1[depth, bh]
                ocm2_filt[depth, p] = ocm2[depth, p] - median2[depth, bh]
            tr1 = ocm1_filt[:, p]
            tr2 = ocm2_filt[:, p]
            ocm1_low[:, p] = np.convolve(np.sqrt(np.square(tr1)), f1, 'same')
            ocm2_low[:, p] = np.convolve(np.sqrt(np.square(tr2)), f1, 'same')

        #### Threshold generation ####
        # if state 1
        if fidx % 2 == 0:
            # Median-based filtering
            for depth in range(0, s):
                # Calculate median of baseline signal
                median0_base[depth] = statistics.median(ocm0_low[depth, 0:t_sub_removed*num_train])
                median1_base[depth] = statistics.median(ocm1_low[depth, 0:t_sub_removed*num_train])
                median2_base[depth] = statistics.median(ocm2_low[depth, 0:t_sub_removed*num_train])
                # Get SD of (Median - train)
                sd0[depth] = np.abs(np.std(median0_base[depth] - ocm0_low[depth, 0:t_sub_removed*num_train]))
                sd1[depth] = np.abs(np.std(median1_base[depth] - ocm1_low[depth, 0:t_sub*num_train]))
                sd2[depth] = np.abs(np.std(median2_base[depth] - ocm2_low[depth, 0:t_sub*num_train]))

            '''
            ### visualization
            d = np.linspace(2.3, 4.9, s)
            fig = plt.figure(figsize=(16,8))
            ax1 = fig.add_subplot(111)
            a1 = ax1.plot(d, 2*sd1[:], linewidth=1, label='2sd')
            a1 = ax1.plot(d, 4*sd1[:], linewidth=1, label='4sd')
            a1 = ax1.plot(d, 6*sd1[:], linewidth=1, label='6sd')
            a1 = ax1.plot(d, np.abs(ocm1_filt[:, 0] - median1_base[:]), linewidth=1, label='0')
            a1 = ax1.plot(d, np.abs(ocm1_filt[:, 10] - median1_base[:]), linewidth=1, label='10')
            a1 = ax1.plot(d, np.abs(ocm1_filt[:, 20] - median1_base[:]), linewidth=1, label='20')
            plt.legend(loc='upper right')
            fig.show()
            f_name = 'test.png'
            plt.savefig(f_name)
            '''

            #### Get parameter m ####
            # m and OoE (out of envelop) distribution
            count0 = np.zeros([bin])
            count1 = np.zeros([bin])
            count2 = np.zeros([bin])
            flag0_m = 0
            flag1_m = 0
            flag2_m = 0
            m0=0
            m1=0
            m2=0

            for m in range(0, bin):
                if m % 10 == 0:
                    print('m:', m)
                thr0[:] = m / scale * sd0[:]
                thr1[:] = m / scale * sd1[:]
                thr2[:] = m / scale * sd2[:]

                # loop inside the training set
                # OCM0
                for p in range(0, t_sub_removed*num_train):
                    flag0 = 0
                    for depth in range(0, s):
                        # if not detected yet
                        if flag0 < 1:  # OCM0
                            # check every depth and count if it's larger than the threshold
                            if np.abs(ocm0_low[depth, p] - median0_base[depth]) > thr0[depth]:
                                count0[m] = count0[m] + 1
                                flag0 = 1

                # OCM1 and OCM2
                for p in range(0, t_sub*num_train):
                    flag1 = 0
                    flag2 = 0
                    for depth in range(0, s):
                        # if not detected yet
                        if flag1 < 1:  # OCM1
                            if np.abs(ocm1_low[depth, p] - median1_base[depth]) > thr1[depth]:
                                count1[m] = count1[m] + 1
                                flag1 = 1
                        if flag2 < 1:  # OCM2
                            if np.abs(ocm2_low[depth, p] - median2_base[depth]) > thr2[depth]:
                                count2[m] = count2[m] + 1
                                flag2 = 1

        fname = 'm012_' + str(Sub_run_name) + '_bin' + str(bin) + '_scale' + str(scale) + '_train' + str(
            num_train) + '.pkl'
        with open(fname, 'wb') as f:
            pickle.dump([count0, count1, count2, median0_base, median1_base, median2_base, sd0, sd1, sd2], f)


    # ========================Visualize==============================================
    m_ = np.linspace(0, bin/scale, bin)  # My m
    fig = plt.figure(figsize=(7, 4))
    # OCM1
    ax1 = fig.add_subplot(111)
    a0 = ax1.plot(m_, count0[:], 'ro-', linewidth=1, label="ocm0")
    a1 = ax1.plot(m_, count1[:], 'go-', linewidth=1, label="ocm1")
    a2 = ax1.plot(m_, count2[:], 'bo-', linewidth=1, label="ocm2")
    ax1.set_title('Distribution of parameter m')
    ax1.set_xlabel('m')
    ax1.set_ylabel('Number of traces above threshold')
    plt.legend(loc='upper right')
    fig.show()
    f_name = 'm_dist_train' + str(num_train) + '_' + Sub_run_name + '.png'
    plt.savefig(f_name)
