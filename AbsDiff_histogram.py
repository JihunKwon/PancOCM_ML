# This code draws histogram of absolute difference and compare between bh=4 (state 1) and 6 (state 2)

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import csv
import statistics
import pickle
from trace_outlier_check import outlier_remove

plt.close('all')

out_list = []

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams["font.size"] = 13
plt.rcParams["legend.fontsize"] = 11
plt.rcParams["legend.edgecolor"] = 'k'
plt.rcParams["legend.labelspacing"] = 0.5

# Jihun Local
out_list.append("C:\\Users\\Kwon\\Documents\\Panc_OCM\\Subject_01_20180928\\run1.npy")  # Before water
out_list.append("C:\\Users\\Kwon\\Documents\\Panc_OCM\\Subject_01_20180928\\run2.npy")  # After water
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

#rep_list = [8196, 8196, 8192, 8192, 6932, 6932, 3690, 3690, 3401, 3401, 3690, 3690]
rep_list = [8196, 8196, 8192, 8192, 6932, 6932, 3690, 3690, 3401, 3401, 3690, 3690]
sr_list = ['s1r1', 's1r1', 's1r2', 's1r2', 's2r1', 's2r1', 's2r2', 's2r2', 's3r1', 's3r1', 's3r2', 's3r2']


# these are where the runs end in each OCM file
num_train = 3
num_test = 10 - num_train
num_ocm = 3
num_bh = 5  # number of bh in each state
bin = 1000
scale = 50  # number divides m

s_new = 296  # the depth your interest


#for fidx in range(0, np.size(rep_list)):
for fidx in range(4, 6):
    Sub_run_name = sr_list[fidx]
    print('Status: train' + str(num_train) + '_' + Sub_run_name)

    in_filename = out_list[fidx]
    ocm = np.load(in_filename)

    # crop data
    ocm = ocm[300:650, :]
    s, t = np.shape(ocm)

    # variables initialization
    median1 = np.zeros([s_new, num_bh])
    if fidx % 2 == 0:
        median1_base = np.zeros([s_new])  # median of filtered signal
        sd1 = np.zeros([s_new])  # sd of (median - train)
        thr1 = np.zeros([s_new])  # threshold

    # divide the data into each OCM and store absolute value
    b = np.linspace(0, t - 1, t)
    b0 = np.mod(b, 4) == 0
    ocm0 = ocm[:, b0]
    b1 = np.mod(b, 4) == 1
    ocm1 = ocm[:, b1]
    s, c0 = np.shape(ocm0)
    print('ocm0:', ocm0.shape)

    # first few traces are strange. Remove them
    ocm0 = ocm0[:, c0 - num_bh * rep_list[fidx]:]
    ocm1 = ocm1[:, c0 - num_bh * rep_list[fidx]:]

    print('ocm0 new:', ocm0.shape)
    s, c0_new = np.shape(ocm0)
    t_sub = int(c0_new / num_bh)

    # Manually remove outlier. (OCM0 contains about 0.5% of outlier)
    ocm0_new = outlier_remove(Sub_run_name, c0_new, ocm0)
    ocm1_filt = np.zeros([s_new, c0_new])  # filtered signal (median based filtering)
    ocm1_low = np.zeros([s_new, c0_new])  # low pass
    median1_low = np.zeros([s_new, num_bh])
    f1 = np.ones([10])  # low pass kernel

    # Median-based filtering
    for bh in range(0, num_bh):
        for depth in range(0, s_new):
            # get median for each bh
            median1[depth, bh] = statistics.median(ocm1[depth, bh * t_sub:(bh + 1) * t_sub])

    # Filter the median
    length = 50  # size of low pass filter
    f1_medi = np.ones([length])
    for bh in range(0, num_bh):
        tr1_medi = median1[:, bh]
        median1_low[:, bh] = np.convolve(tr1_medi, f1_medi, 'same') / length

    # filtering all traces with median trace
    # The size of ocm0 is different with ocm1 and ocm2. Has to be filtered separately.
    ## OCM1
    bh = -1
    for p in range(0, c0_new):
        if p % rep_list[fidx] == 0:
            bh = bh + 1
        for depth in range(0, s_new):
            # filter the signal (subtract median from each trace of corresponding bh)
            ocm1_filt[depth, p] = ocm1[depth, p] - median1_low[depth, bh]
        tr1 = ocm1_filt[:, p]
        ocm1_low[:, p] = np.convolve(np.sqrt(np.square(tr1)), f1, 'same')

    #### Threshold generation ####
    # if state 1
    if fidx % 2 == 0:
        # Median-based filtering
        for depth in range(0, s_new):
            # Calculate median of baseline signal
            median1_base[depth] = statistics.median(ocm1_low[depth, 0:t_sub * num_train])
            sd1[depth] = np.abs(np.std(median1_base[depth] - ocm1_low[depth, 0:t_sub*num_train]))

    # store data
    if fidx % 2 == 0:
        ocm1_low_state1 = np.zeros([s_new, c0_new])  # store low pass of state 1 for later visualization
        abs_diff_st1 = np.zeros([s_new, c0_new])
        ocm1_low_state1 = ocm1_low  # if state 1, store ocm1_low to use later for visualization.

        for p in range(0, c0_new):
            abs_diff_st1[:, p] = np.abs(ocm1_low_state1[:, p] - median1_base[:])

    else:
        ocm1_low_state2 = np.zeros([s_new, c0_new])  # store low pass of state 2
        abs_diff_st2 = np.zeros([s_new, c0_new])
        ocm1_low_state2 = ocm1_low

        for p in range(0, c0_new):
            abs_diff_st2[:, p] = np.abs(ocm1_low_state2[:, p] - median1_base[:])



    #### Visualzation ####
    if fidx % 2 == 1:
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(1,1,1)

        target_depth = 94  # the depth you want to get a histogram
        ax.hist(abs_diff_st1[target_depth, :], bins=200, color='red', alpha=0.7, label='State 1 (n=6932)')
        ax.hist(abs_diff_st2[target_depth, :], bins=200, color='blue', alpha=0.4, label='State 2 (n=6932)')
        plt.axvline(10 * sd1[target_depth], color='k', linewidth=1.3, label='Threshold (m*SD, m=10)')
        ax.set_xlabel("Absolute difference (a.u.)")
        ax.set_ylabel("Count")
        ax.legend(loc='upper right')
        #plt.show()

        f_name = 'hist_AbsDiff_s2r1.png'
        f_name_eps = 'hist_AbsDiff_s2r1.eps'
        plt.savefig(f_name)
        plt.savefig(f_name_eps)
