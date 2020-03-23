'''
In Hotelling's law, abnormality a is given by
    a(x') = (x' - mean)^2 / SD^2
Following codes calculate abnormality a(x') and compare it with Chi^2-distribution.
'''

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import csv
import statistics
import pickle
import colorsys
from trace_outlier_check import outlier_remove
from scipy.stats import chi2
from Count_Outlier import count_outlier

plt.close('all')
out_list = []

plt.rcParams['font.family'] ='sans-serif'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

# dropbox
dropbox_path = "C:\\Users\\Kwon\\Dropbox\\result_figures"

# plot style list
color_list = ['b', 'b', 'b', 'b', 'r', 'r', 'r', 'r', 'g', 'g', 'g', 'g']
linestyle_list = ['-', '-', '--', '--', '-', '-', '--', '--', '-', '-', '--', '--', ]
shape_list = ['o', 'o', 'o', 'o', 's', 's', 's', 's', '^', '^', '^', '^']
fill_list = ['full', 'full', 'none', 'none', 'full', 'full', 'none', 'none', 'full', 'full', 'none', 'none', ]

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

sr_list = ['s1r1', 's1r1', 's1r2', 's1r2', 's2r1', 's2r1', 's2r2', 's2r2', 's3r1', 's3r1', 's3r2', 's3r2']
rep_list = [8196, 8196, 8192, 8192, 6932, 6932, 3690, 3690, 3401, 3401, 3690, 3690]
#rep_list = [500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500]

'''
out_list.append("C:\\Users\\Kwon\\Documents\\Panc_OCM\\Subject_01_20180928\\run1.npy") #Before water
out_list.append("C:\\Users\\Kwon\\Documents\\Panc_OCM\\Subject_01_20180928\\run2.npy") #After water
out_list.append("C:\\Users\\Kwon\\Documents\\Panc_OCM\\Subject_01_20181102\\run1.npy")
out_list.append("C:\\Users\\Kwon\\Documents\\Panc_OCM\\Subject_01_20181102\\run2.npy")
sr_list = ['s1r1', 's1r1', 's1r2', 's1r2']
rep_list = [200, 200, 200, 200]
'''

num_train = 3
num_test = 10 - num_train
num_ocm = 3
num_bh = 5 # number of bh in each state
#s_new = 350  # Original depth. 2.3cm to 4.9cm
#s_new = 297  # 2.3cm to 4.5cm
#s_new = 231  # 2.3cm to 4.0cm
s_new = 269  # 2.2cm to 4.2cm
threshold = 0.01  # Threshold for Chi^2
interval = 2  # Interval for averaging
plot_interval = 1000
interval_list = [5]

#### Set the threshold based on Chi^2 ####
Chi_list = [0.00000001, 0.000000001, 0.0000000001, 0.00000000001, 0.000000000001]
_, chi2_interval_max_0000001 = chi2.interval(alpha=1 - 0.000001, df=1)
_, chi2_interval_max_00000001 = chi2.interval(alpha=1 - 0.0000001, df=1)
_, chi2_interval_max_000000001 = chi2.interval(alpha=1 - 0.00000001, df=1)
_, chi2_interval_max_0000000001 = chi2.interval(alpha=1 - 0.000000001, df=1)
print('chi2_interval_max_00000001: ' + str(chi2_interval_max_00000001))
print('chi2_interval_max_000000001: ' + str(chi2_interval_max_000000001))
print('chi2_interval_max_0000000001: ' + str(chi2_interval_max_0000000001))

out0_test = np.zeros([len(Chi_list), int(len(rep_list) / 2), 10])  # output test result
out1_test = np.zeros([len(Chi_list), int(len(rep_list) / 2), 10])
out2_test = np.zeros([len(Chi_list), int(len(rep_list) / 2), 10])
out_mean = np.zeros([len(Chi_list), int(len(rep_list) / 2), 10])  # Mean of all OCM
out_mean01 = np.zeros([len(Chi_list), int(len(rep_list) / 2), 10])  # Mean of OCM0 and OCM1

for interval_idx in range(0, np.size(rep_list)):
    interval = interval_list[interval_idx]
    for fidx in range(0, np.size(rep_list)):
        # for fidx in range(0, 2):
        # fidx = 0
        Sub_run_name = sr_list[fidx]
        print('Status: train' + str(num_train) + '_' + Sub_run_name)
        plt.rcParams["font.size"] = 11
        in_filename = out_list[fidx]
        ocm = np.load(in_filename)

        # crop data
        if s_new is 269:
            ocm = ocm[285:285+s_new, :]
        else:
            ocm = ocm[300:650, :]
        s, t = np.shape(ocm)

        # variables initialization
        median0 = np.zeros([s_new, num_bh])  # median
        median1 = np.zeros([s_new, num_bh])
        median2 = np.zeros([s_new, num_bh])
        if fidx % 2 == 0:
            mean0_base = np.zeros([s_new])  # mean of filtered signal
            mean1_base = np.zeros([s_new])
            mean2_base = np.zeros([s_new])
            sd0 = np.zeros([s_new])  # sd of (median - train)
            sd1 = np.zeros([s_new])
            sd2 = np.zeros([s_new])

        # divide the data into each OCM and store absolute value
        b = np.linspace(0, t - 1, t)
        b0 = np.mod(b, 4) == 0
        ocm0 = ocm[:, b0]
        b1 = np.mod(b, 4) == 1
        ocm1 = ocm[:, b1]
        b2 = np.mod(b, 4) == 2
        ocm2 = ocm[:, b2]
        s, c0 = np.shape(ocm0)
        print('ocm0:', ocm0.shape)

        # first few traces are strange. Remove them
        ocm0 = ocm0[:, c0 - num_bh * rep_list[fidx]:]
        ocm1 = ocm1[:, c0 - num_bh * rep_list[fidx]:]
        ocm2 = ocm2[:, c0 - num_bh * rep_list[fidx]:]

        print('ocm0 new:', ocm0.shape)
        s, c0_new = np.shape(ocm0)
        t_sub = int(c0_new / num_bh)

        # Manually remove outlier. (OCM0 contains about 0.5% of outlier)
        ocm0_new = outlier_remove(Sub_run_name, c0_new, ocm0)
        s, c0_new_removed = np.shape(ocm0_new)
        t_sub_removed = int(c0_new_removed / num_bh)
        dif = ocm0.shape[1] - ocm0_new.shape[1]

        print('ocm0 new_removed:', ocm0_new.shape)
        print(str(dif), 'was removed. It is', '{:.2f}'.format(dif * 100 / ocm0.shape[1]), '%')

        ocm0_filt = np.zeros([s_new, c0_new_removed])  # filtered signal (median based filtering)
        ocm1_filt = np.zeros([s_new, c0_new])
        ocm2_filt = np.zeros([s_new, c0_new])
        ocm0_low = np.zeros([s_new, c0_new_removed])  # low pass
        ocm1_low = np.zeros([s_new, c0_new])
        ocm2_low = np.zeros([s_new, c0_new])
        # Abnormality
        A0 = np.zeros([s_new, c0_new_removed])
        A1 = np.zeros([s_new, c0_new])
        A2 = np.zeros([s_new, c0_new])
        A0_pre_medi = np.zeros([s_new])
        A1_pre_medi = np.zeros([s_new])
        A2_pre_medi = np.zeros([s_new])
        A0_post_medi = np.zeros([s_new])
        A1_post_medi = np.zeros([s_new])
        A2_post_medi = np.zeros([s_new])
        median0_low = np.zeros([s_new, num_bh])
        median1_low = np.zeros([s_new, num_bh])
        median2_low = np.zeros([s_new, num_bh])
        f1 = np.ones([10])  # low pass kernel

        # Median-based filtering
        for bh in range(0, num_bh):
            for depth in range(0, s_new):
                # get median for each bh
                median0[depth, bh] = statistics.median(ocm0_new[depth, bh * t_sub_removed:(bh + 1) * t_sub_removed])
                median1[depth, bh] = statistics.median(ocm1[depth, bh * t_sub:(bh + 1) * t_sub])
                median2[depth, bh] = statistics.median(ocm2[depth, bh * t_sub:(bh + 1) * t_sub])

        # Filter the median
        length = 50  # size of low pass filter
        f1_medi = np.ones([length])
        for bh in range(0, num_bh):
            tr0_medi = median0[:, bh]
            tr1_medi = median1[:, bh]
            tr2_medi = median2[:, bh]
            median0_low[:, bh] = np.convolve(tr0_medi, f1_medi, 'same') / length
            median1_low[:, bh] = np.convolve(tr1_medi, f1_medi, 'same') / length
            median2_low[:, bh] = np.convolve(tr2_medi, f1_medi, 'same') / length

        # filtering all traces with median trace
        # The size of ocm0 is different with ocm1 and ocm2. Has to be filtered separately.
        ## OCM0
        bh = -1
        for p in range(0, c0_new_removed):
            # have to consider the number of traces removed
            if p % rep_list[fidx] - dif // num_bh == 0:
                bh = bh + 1
            for depth in range(0, s_new):
                # filter the signal (subtract median from each trace of corresponding bh)
                ocm0_filt[depth, p] = ocm0_new[depth, p] - median0_low[depth, bh]
            tr0 = ocm0_filt[:, p]
            ocm0_low[:, p] = np.convolve(np.sqrt(np.square(tr0)), f1, 'same')

        ## OCM1 and 2
        bh = -1
        for p in range(0, c0_new):
            if p % rep_list[fidx] == 0:
                bh = bh + 1
            for depth in range(0, s_new):
                # filter the signal (subtract median from each trace of corresponding bh)
                ocm1_filt[depth, p] = ocm1[depth, p] - median1_low[depth, bh]
                ocm2_filt[depth, p] = ocm2[depth, p] - median2_low[depth, bh]
            tr1 = ocm1_filt[:, p]
            tr2 = ocm2_filt[:, p]
            ocm1_low[:, p] = np.convolve(np.sqrt(np.square(tr1)), f1, 'same')
            ocm2_low[:, p] = np.convolve(np.sqrt(np.square(tr2)), f1, 'same')



        ## Averaging the traces into the time dimension using the "interval" parameter ""
        ocm0_ave = np.zeros([s_new, c0_new_removed//interval+1])
        ocm1_ave = np.zeros([s_new, c0_new//interval+1])
        ocm2_ave = np.zeros([s_new, c0_new//interval+1])

        count0 = 0
        count12 = 0
        for p in range(0, c0_new_removed):
            if p % interval is 0:
                ocm0_ave[:, count0] = np.mean(ocm0_low[:, count0*interval:(count0+1)*interval], axis=1)
                count0+=1
        for p in range(0, c0_new):
            if p % interval is 0:
                ocm1_ave[:, count12] = np.mean(ocm1_low[:, count12*interval:(count12+1)*interval], axis=1)
                ocm2_ave[:, count12] = np.mean(ocm2_low[:, count12*interval:(count12+1)*interval], axis=1)
                count12+=1
        # check zero array
        if np.sum(ocm0_ave[:, -1]) is 0.0:
            ocm0_ave = ocm0_ave[:, -1]  # remove last element
            print('Last Element Removed!, ocm0')
        if np.sum(ocm1_ave[:, -1]) is 0.0:
            ocm1_ave = ocm1_ave[:, -1]
            print('Last Element Removed!, ocm1')
        if np.sum(ocm1_ave[:, -1]) is 0.0:
            ocm2_ave = ocm2_ave[:, -1]
            print('Last Element Removed!, ocm2')
        # update parameters
        c0_new_removed = ocm0_ave.shape[1]
        c0_new = ocm1_ave.shape[1]
        t_sub = int(c0_new / num_bh)
        t_sub_removed = int(c0_new_removed / num_bh)

        # Abnormality
        A0 = np.zeros([s_new, c0_new_removed])
        A1 = np.zeros([s_new, c0_new])
        A2 = np.zeros([s_new, c0_new])
        A0_pre_medi = np.zeros([s_new])
        A1_pre_medi = np.zeros([s_new])
        A2_pre_medi = np.zeros([s_new])
        A0_post_medi = np.zeros([s_new])
        A1_post_medi = np.zeros([s_new])
        A2_post_medi = np.zeros([s_new])

        # if state 1, get mean and sd. Later use it to calculate abnormality.
        if fidx % 2 == 0:
            # Median-based filtering
            for depth in range(0, s_new):
                # Instead of median, use mean for anomality calculation
                mean0_base[depth] = statistics.mean(ocm0_ave[depth, 0:t_sub_removed * num_train])
                mean1_base[depth] = statistics.mean(ocm1_ave[depth, 0:t_sub * num_train])
                mean2_base[depth] = statistics.mean(ocm2_ave[depth, 0:t_sub * num_train])
                # Get SD of (Median - train)
                sd0[depth] = np.abs(np.std(mean0_base[depth] - ocm0_ave[depth, 0:t_sub_removed * num_train]))
                sd1[depth] = np.abs(np.std(mean1_base[depth] - ocm1_ave[depth, 0:t_sub * num_train]))
                sd2[depth] = np.abs(np.std(mean2_base[depth] - ocm2_ave[depth, 0:t_sub * num_train]))

        #### Abnormality calculation ####
        ## OCM0
        for p in range(0, c0_new_removed):
            for depth in range(0, s_new):
                A0[depth, p] = np.square((mean0_base[depth] - ocm0_ave[depth, p]) / sd0[depth])

        ## OCM1 and 2
        for p in range(0, c0_new):
            for depth in range(0, s_new):
                A1[depth, p] = np.square((mean1_base[depth] - ocm1_ave[depth, p]) / sd1[depth])
                A2[depth, p] = np.square((mean2_base[depth] - ocm2_ave[depth, p]) / sd2[depth])

        if fidx % 2 == 0:
            A0_pre = A0
            A1_pre = A1
            A2_pre = A2
        else:
            A0_post = A0
            A1_post = A1
            A2_post = A2
            # Get median
            for depth in range(0, s_new):
                A0_pre_medi[depth] = statistics.median(A0_pre[depth, :])
                A1_pre_medi[depth] = statistics.median(A1_pre[depth, :])
                A2_pre_medi[depth] = statistics.median(A2_pre[depth, :])
                A0_post_medi[depth] = statistics.median(A0_post[depth, :])
                A1_post_medi[depth] = statistics.median(A1_post[depth, :])
                A2_post_medi[depth] = statistics.median(A2_post[depth, :])

        # ========================Visualize==============================================

        if fidx % 2 == 1:
            if s_new is 231:
                d = np.linspace(2.3, 4.0, s_new)
            elif s_new is 297:
                d = np.linspace(2.3, 4.5, s_new)
            elif s_new is 350:
                d = np.linspace(2.3, 4.9, s_new)
            elif s_new is 269:
                d = np.linspace(2.2, 4.2, s_new)

            ## Plot Log Scale ##
            fig = plt.figure(figsize=(20, 8))

            # State 1
            ax1 = fig.add_subplot(231)
            a1 = ax1.plot(d, A0_pre_medi[:], linewidth=2.0, color='k', label='median')
            plt.axhline(y=chi2_interval_max_0000000001, color='k', linestyle='--', linewidth=2, label='0.000001% threshold')
            plt.title('OCM0, State 1')
            plt.legend(loc='upper right')
            plt.ylabel('Abnormality')
            ax1.set_yscale('log')
            ax1.set_ylim(0.1, 10000)
            if s_new is 269:
                plt.xticks(np.arange(2.2, 4.2+1, 0.5))
                plt.xlim(2.2, 4.2)

            nValues = range(0, int(A0_pre.shape[1]))
            # Change line colors continuously. Red: Bh=1, Blue: Bh=5
            colors = {n: colorsys.hsv_to_rgb(hue, 0.9, 0.7) for n, hue in zip(nValues, np.linspace(0, 0.7, (len(nValues))))}
            for i in range(0, A0_pre.shape[1], plot_interval):
                a1 = ax1.plot(d, A0_pre[:, i], linewidth=1, color=colors[i])
            a1 = ax1.plot(d, A0_pre_medi[:], linewidth=2.0, color='k', label='median')
            plt.axhline(y=chi2_interval_max_0000000001, color='k', linestyle='--', linewidth=2, label='0.000001% threshold')

            ax2 = fig.add_subplot(232)
            a2 = ax2.plot(d, A1_pre_medi[:], linewidth=2.0, color='k', label='median')
            plt.axhline(y=chi2_interval_max_0000000001, color='k', linestyle='--', linewidth=2, label='0.000001% threshold')
            plt.title('OCM1, State 1')
            plt.legend(loc='upper right')
            plt.ylabel('Abnormality')
            ax2.set_yscale('log')
            ax2.set_ylim(0.1, 10000)
            if s_new is 269:
                plt.xticks(np.arange(2.2, 4.2+1, 0.5))
                plt.xlim(2.2, 4.2)

            nValues = range(0, int(A1_pre.shape[1]))
            colors = {n: colorsys.hsv_to_rgb(hue, 0.9, 0.7) for n, hue in zip(nValues, np.linspace(0, 0.7, (len(nValues))))}
            for i in range(0, A1_pre.shape[1], plot_interval):
                a2 = ax2.plot(d, A1_pre[:, i], linewidth=1, color=colors[i])
            a2 = ax2.plot(d, A1_pre_medi[:], linewidth=2.0, color='k', label='median')
            plt.axhline(y=chi2_interval_max_0000000001, color='k', linestyle='--', linewidth=2, label='0.000001% threshold')

            ax3 = fig.add_subplot(233)
            a3 = ax3.plot(d, A2_pre_medi[:], linewidth=2.0, color='k', label='median')
            plt.axhline(y=chi2_interval_max_0000000001, color='k', linestyle='--', linewidth=2, label='0.000001% threshold')
            plt.title('OCM2, State 1')
            plt.legend(loc='upper right')
            plt.ylabel('Abnormality')
            ax3.set_yscale('log')
            ax3.set_ylim(0.1, 10000)
            if s_new is 269:
                plt.xticks(np.arange(2.2, 4.2+1, 0.5))
                plt.xlim(2.2, 4.2)

            nValues = range(0, int(A2_pre.shape[1]))
            colors = {n: colorsys.hsv_to_rgb(hue, 0.9, 0.7) for n, hue in zip(nValues, np.linspace(0, 0.7, (len(nValues))))}
            for i in range(0, A2_pre.shape[1], plot_interval):
                a3 = ax3.plot(d, A2_pre[:, i], linewidth=1, color=colors[i])
            a3 = ax3.plot(d, A2_pre_medi[:], linewidth=2.0, color='k', label='median')
            plt.axhline(y=chi2_interval_max_0000000001, color='k', linestyle='--', linewidth=2, label='0.000001% threshold')

            # State 2
            if fidx is not 11:  # The data in OCM0 for s3r2 state 2 is strange. We don't use this.
                ax4 = fig.add_subplot(234)
                a4 = ax4.plot(d, A0_post_medi[:], linewidth=2.0, color='k', label='median')
                plt.axhline(y=chi2_interval_max_0000000001, color='k', linestyle='--', linewidth=2, label='0.000001% threshold')
                plt.title('OCM0, State 2')
                plt.legend(loc='upper right')
                plt.xlabel('Depth (cm)')
                plt.ylabel('Abnormality')
                ax4.set_yscale('log')
                ax4.set_ylim(0.1, 10000)
                if s_new is 269:
                    plt.xticks(np.arange(2.2, 4.2+1, 0.5))
                    plt.xlim(2.2, 4.2)

                nValues = range(0, int(A0_post.shape[1]))
                colors = {n: colorsys.hsv_to_rgb(hue, 0.9, 0.7) for n, hue in
                          zip(nValues, np.linspace(0, 0.7, (len(nValues))))}
                for i in range(0, A0_post.shape[1], plot_interval):
                    a4 = ax4.plot(d, A0_post[:, i], linewidth=1, color=colors[i])
                a4 = ax4.plot(d, A0_post_medi[:], linewidth=2.0, color='k', label='median')
                plt.axhline(y=chi2_interval_max_0000000001, color='k', linestyle='--', linewidth=2, label='0.000001% threshold')

            ax5 = fig.add_subplot(235)
            a5 = ax5.plot(d, A1_post_medi[:], linewidth=2.0, color='k', label='median')
            plt.axhline(y=chi2_interval_max_0000000001, color='k', linestyle='--', linewidth=2, label='0.000001% threshold')
            plt.title('OCM1, State 2')
            plt.legend(loc='upper right')
            plt.xlabel('Depth (cm)')
            plt.ylabel('Abnormality')
            ax5.set_yscale('log')
            ax5.set_ylim(0.1, 10000)
            if s_new is 269:
                plt.xticks(np.arange(2.2, 4.2+1, 0.5))
                plt.xlim(2.2, 4.2)

            nValues = range(0, int(A1_post.shape[1]))
            colors = {n: colorsys.hsv_to_rgb(hue, 0.9, 0.7) for n, hue in zip(nValues, np.linspace(0, 0.7, (len(nValues))))}
            for i in range(0, A1_post.shape[1], plot_interval):
                a5 = ax5.plot(d, A1_post[:, i], linewidth=1, color=colors[i])
            a5 = ax5.plot(d, A1_post_medi[:], linewidth=2.0, color='k', label='median')
            plt.axhline(y=chi2_interval_max_0000000001, color='k', linestyle='--', linewidth=2, label='0.000001% threshold')

            ax6 = fig.add_subplot(236)
            a6 = ax6.plot(d, A2_post_medi[:], linewidth=2.0, color='k', label='median')
            plt.axhline(y=chi2_interval_max_0000000001, color='k', linestyle='--', linewidth=2, label='0.000001% threshold')
            plt.title('OCM2, State 2')
            plt.legend(loc='upper right')
            plt.xlabel('Depth (cm)')
            plt.ylabel('Abnormality')
            ax6.set_yscale('log')
            ax6.set_ylim(0.1, 10000)
            if s_new is 269:
                plt.xticks(np.arange(2.2, 4.2+1, 0.5))
                plt.xlim(2.2, 4.2)

            nValues = range(0, int(A2_post.shape[1]))
            colors = {n: colorsys.hsv_to_rgb(hue, 0.9, 0.7) for n, hue in zip(nValues, np.linspace(0, 0.7, (len(nValues))))}
            for i in range(0, A2_post.shape[1], plot_interval):
                a6 = ax6.plot(d, A2_post[:, i], linewidth=1, color=colors[i])
            a6 = ax6.plot(d, A2_post_medi[:], linewidth=2.0, color='k', label='median')
            plt.axhline(y=chi2_interval_max_0000000001, color='k', linestyle='--', linewidth=2, label='0.000001% threshold')

            fig.show()
            plt.tight_layout()
            f_name = 'Abnomality_' + Sub_run_name +'_train'+str(num_train) + '_log'+'_s'+str(s_new)+'_interval'+str(interval)+'.png'
            plt.savefig(f_name)
            # plt.savefig(os.path.join(dropbox_path, f_name))  # Save on dropbox directory
            plt.close()


        for chi_idx in range(0, len(Chi_list)):
            Chi_area = Chi_list[chi_idx]
            if fidx % 2 is 0:  # State 1
                out0_test, out1_test, out2_test = count_outlier(chi_idx, Chi_area, fidx, t_sub_removed, t_sub, s_new, out0_test, out1_test, out2_test, A0_pre, A1_pre, A2_pre)
            else:  # State 2
                out0_test, out1_test, out2_test = count_outlier(chi_idx, Chi_area, fidx, t_sub_removed, t_sub, s_new, out0_test, out1_test, out2_test, A0_post, A1_post, A2_post)

                # Normalize and show in percentage
                out0_test[chi_idx, int(fidx / 2), :] = out0_test[chi_idx, int(fidx / 2), :] / t_sub_removed * 100  # Show in percentage
                out1_test[chi_idx, int(fidx / 2), :] = out1_test[chi_idx, int(fidx / 2), :] / t_sub * 100  # Show in percentage
                out2_test[chi_idx, int(fidx / 2), :] = out2_test[chi_idx, int(fidx / 2), :] / t_sub * 100  # Show in percentage
                # Get average for later visualization
                if fidx is not 11:
                    out_mean[chi_idx, int(fidx / 2), :] = (out0_test[chi_idx, int(fidx / 2), :] + out1_test[chi_idx, int(fidx / 2), :] + out2_test[chi_idx, int(fidx / 2), :]) / 3
                    out_mean01[chi_idx, int(fidx / 2), :] = (out0_test[chi_idx, int(fidx / 2), :] + out1_test[chi_idx, int(fidx / 2), :]) / 2
                else:  # The data in OCM0 for s3r2 state 2 is strange. We don't use this.
                    out_mean[chi_idx, int(fidx / 2), :] = (out1_test[chi_idx, int(fidx / 2), :] + out2_test[chi_idx, int(fidx / 2), :]) / 2
                    out_mean01[chi_idx, int(fidx / 2), :] = out1_test[chi_idx, int(fidx / 2), :]

    print('out0_test fidx_'+str(fidx)+'_chi_'+ str(chi_idx), out0_test)
    print('out1_test fidx_'+str(fidx)+'_chi_'+ str(chi_idx), out1_test)
    print('out2_test fidx_'+str(fidx)+'_chi_'+ str(chi_idx), out2_test)

    for chi_idx in range(0, len(Chi_list)):
        # Visualize outlier number
        bh_idx = np.linspace(1, 10, 10)

        fig = plt.figure(figsize=(10, 9))
        ax = fig.add_subplot(221)
        plt.title('Outlier detection rate, OCM0')
        plt.xlabel('Breath-holds')
        plt.ylabel('Outlier detection rate (%)')
        plt.xticks(np.arange(1, 10+1, 1))
        plt.yticks(np.arange(0, 100+1, 10))
        plt.xlim(1, 10)
        plt.ylim(0, 100)
        plt.grid(True)
        ax.axvspan(1, 5, color='tab:blue', alpha=0.15)
        ax.axvspan(6, 10, color='tab:orange', alpha=0.15)
        for fidx in range(0, len(rep_list), 2):
            print('fidx:', fidx)
            if fidx is not 10:  # The data in OCM0 for s3r2 state 2 is strange. We don't use this.
                a0 = ax.plot(bh_idx, out0_test[chi_idx, int(fidx / 2), :], linewidth=1.5, marker=shape_list[fidx], \
                             fillstyle=fill_list[fidx], color=color_list[fidx], linestyle=linestyle_list[fidx] ,label=sr_list[fidx])

        plt.legend(loc='upper left')
        ax = fig.add_subplot(222)
        plt.title('Outlier detection rate, OCM1')
        plt.xlabel('Breath-holds')
        plt.ylabel('Outlier detection rate (%)')
        plt.xticks(np.arange(1, 10+1, 1))
        plt.yticks(np.arange(0, 100+1, 10))
        plt.xlim(1, 10)
        plt.ylim(0, 100)
        plt.grid(True)
        ax.axvspan(1, 5, color='tab:blue', alpha=0.15)
        ax.axvspan(6, 10, color='tab:orange', alpha=0.15)
        for fidx in range(0, len(rep_list), 2):
            a1 = ax.plot(bh_idx, out1_test[chi_idx, int(fidx / 2), :], linewidth=1.5, marker=shape_list[fidx], \
                         fillstyle=fill_list[fidx], color=color_list[fidx], linestyle=linestyle_list[fidx], label=sr_list[fidx])

        plt.legend(loc='upper left')
        ax = fig.add_subplot(223)
        plt.title('Outlier detection rate, OCM2')
        plt.xlabel('Breath-holds')
        plt.ylabel('Outlier detection rate (%)')
        plt.xticks(np.arange(1, 10+1, 1))
        plt.yticks(np.arange(0, 100+1, 10))
        plt.xlim(1, 10)
        plt.ylim(0, 100)
        plt.grid(True)
        ax.axvspan(1, 5, color='tab:blue', alpha=0.15)
        ax.axvspan(6, 10, color='tab:orange', alpha=0.15)
        for fidx in range(0, len(rep_list), 2):
            a2 = ax.plot(bh_idx, out2_test[chi_idx, int(fidx / 2), :], linewidth=1.5, marker=shape_list[fidx], \
                         fillstyle=fill_list[fidx], color=color_list[fidx], linestyle=linestyle_list[fidx], label=sr_list[fidx])

        plt.legend(loc='upper left')
        ax = fig.add_subplot(224)
        plt.title('Outlier detection rate, average')
        plt.xlabel('Breath-holds')
        plt.ylabel('Outlier detection rate (%)')
        plt.xticks(np.arange(1, 10+1, 1))
        plt.yticks(np.arange(0, 100+1, 10))
        plt.xlim(1, 10)
        plt.ylim(0, 100)
        plt.grid(True)
        ax.axvspan(1, 5, color='tab:blue', alpha=0.15)
        ax.axvspan(6, 10, color='tab:orange', alpha=0.15)
        for fidx in range(0, len(rep_list), 2):
            a3 = ax.plot(bh_idx, out_mean[chi_idx, int(fidx / 2), :], linewidth=1.5, marker=shape_list[fidx], \
                         fillstyle=fill_list[fidx], color=color_list[fidx], linestyle=linestyle_list[fidx], label=sr_list[fidx])

        plt.legend(loc='upper left')
        plt.tight_layout()
        f_name = 'Outlier_rate_'+str(Chi_list[chi_idx])+'_train'+str(num_train)+'_s'+str(s_new)+'_interval'+str(interval)+'.png'
        plt.savefig(f_name)
        #plt.savefig(os.path.join(dropbox_path, f_name))  # Save on dropbox directory