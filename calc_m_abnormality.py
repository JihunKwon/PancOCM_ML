'''
In Hotelling's law, abnormality a is given by
    a(x') = (x' - mean)^2 / SD^2
Following codes calculate abnormality a(x') and compare it with Chi^2-distribution.
'''


import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import csv
import statistics
import pickle
from trace_outlier_check import outlier_remove
from scipy.stats import chi2

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

sr_list = ['s1r1', 's1r1', 's1r2', 's1r2', 's2r1', 's2r1', 's2r2', 's2r2', 's3r1', 's3r1', 's3r2', 's3r2']
#rep_list = [8196, 8196, 8192, 8192, 6932, 6932, 3690, 3690, 3401, 3401, 3690, 3690]
rep_list = [500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500]


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
#bin = 1000
#scale = 50  # number divides m
bin = 2000
scale = 20  # number divides m
s_new = 296  # the depth your interest
threshold = 0.01  # Threshold for Chi^2

#### Set the threshold based on Chi^2 ####
_, chi2_interval_max_0001 = chi2.interval(alpha=1 - 0.001, df=1)
_, chi2_interval_max_0005 = chi2.interval(alpha=1 - 0.005, df=1)
_, chi2_interval_max_001 = chi2.interval(alpha=1 - 0.01, df=1)
_, chi2_interval_max_005 = chi2.interval(alpha=1 - 0.05, df=1)
print('chi2_interval_max_0001: ' + str(chi2_interval_max_0001))
print('chi2_interval_max_0005: ' + str(chi2_interval_max_0005))
print('chi2_interval_max_001: ' + str(chi2_interval_max_001))
print('chi2_interval_max_005: ' + str(chi2_interval_max_005))

for fidx in range(0, np.size(rep_list)):
    # for fidx in range(0, 2):
    # fidx = 0
    Sub_run_name = sr_list[fidx]
    print('Status: train' + str(num_train) + '_' + Sub_run_name)
    plt.rcParams["font.size"] = 11
    in_filename = out_list[fidx]
    ocm = np.load(in_filename)

    # crop data
    ocm = ocm[300:650, :]
    s, t = np.shape(ocm)

    # variables initialization
    median0 = np.zeros([s_new, num_bh])  # median
    median1 = np.zeros([s_new, num_bh])
    median2 = np.zeros([s_new, num_bh])
    out0_test = np.zeros([len(rep_list), num_bh])  # output test result
    out1_test = np.zeros([len(rep_list), num_bh])
    out2_test = np.zeros([len(rep_list), num_bh])
    if fidx % 2 == 0:
        # median0_base = np.zeros([s_new]) # median of filtered signal
        # median1_base = np.zeros([s_new])
        # median2_base = np.zeros([s_new])
        mean0_base = np.zeros([s_new])  # mean of filtered signal
        mean1_base = np.zeros([s_new])
        mean2_base = np.zeros([s_new])
        sd0 = np.zeros([s_new])  # sd of (median - train)
        sd1 = np.zeros([s_new])
        sd2 = np.zeros([s_new])
        A0 = np.zeros([s_new])  # Abnormality. a(x') = (x' - mean)^2 / SD^2
        A1 = np.zeros([s_new])
        A2 = np.zeros([s_new])

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

    # if state 1, get mean and sd. Later use it to calculate abnormality.
    if fidx % 2 == 0:
        # Median-based filtering
        for depth in range(0, s_new):
            # Calculate median of baseline signal
            '''
            median0_base[depth] = statistics.median(ocm0_low[depth, 0:t_sub_removed*num_train])
            median1_base[depth] = statistics.median(ocm1_low[depth, 0:t_sub*num_train])
            median2_base[depth] = statistics.median(ocm2_low[depth, 0:t_sub*num_train])
            '''
            # Instead of median, use mean for anomality calculation
            mean0_base[depth] = statistics.mean(ocm0_low[depth, 0:t_sub_removed * num_train])
            mean1_base[depth] = statistics.mean(ocm1_low[depth, 0:t_sub * num_train])
            mean2_base[depth] = statistics.mean(ocm2_low[depth, 0:t_sub * num_train])
            # Get SD of (Median - train)
            sd0[depth] = np.abs(np.std(mean0_base[depth] - ocm0_low[depth, 0:t_sub_removed * num_train]))
            sd1[depth] = np.abs(np.std(mean1_base[depth] - ocm1_low[depth, 0:t_sub * num_train]))
            sd2[depth] = np.abs(np.std(mean2_base[depth] - ocm2_low[depth, 0:t_sub * num_train]))

    #### Abnormality calculation ####
    ## OCM0
    for p in range(0, c0_new_removed):
        for depth in range(0, s_new):
            A0[depth, p] = np.square((mean0_base[depth] - ocm0_low[depth, p]) / sd0[depth])

    ## OCM1 and 2
    for p in range(0, c0_new):
        for depth in range(0, s_new):
            A1[depth, p] = np.square((mean1_base[depth] - ocm1_low[depth, p]) / sd1[depth])
            A2[depth, p] = np.square((mean2_base[depth] - ocm2_low[depth, p]) / sd2[depth])

    if fidx % 2 == 0:
        A0_pre = A0
        A1_pre = A1
        A2_pre = A2
    else:
        A0_post = A0
        A1_post = A1
        A2_post = A2
    '''
    # ========================Visualize==============================================
    if fidx % 2 == 1:
        d = np.linspace(2.3, 4.5, s_new)
        fig = plt.figure(figsize=(26, 8))
       
        # State 1
        ax1 = fig.add_subplot(231)
        plt.axhline(y=chi2_interval_max_0001, color='k', linestyle=':', label='0.1% threshold')
        plt.axhline(y=chi2_interval_max_001, color='k', linestyle='--', label='1% threshold')
        plt.title('OCM0, State 1')
        plt.legend(loc='upper right')
        for i in range(0, 100):
            a1 = ax1.plot(d, A0_pre[:, i], linewidth=1)
       
        ax2 = fig.add_subplot(232)
        plt.axhline(y=chi2_interval_max_0001, color='k', linestyle=':', label='0.1% threshold')
        plt.axhline(y=chi2_interval_max_001, color='k', linestyle='--', label='1% threshold')
        plt.title('OCM1, State 1')
        plt.legend(loc='upper right')
        for i in range(0, 100):
            a2 = ax2.plot(d, A1_pre[:, i], linewidth=1)
       
        ax3 = fig.add_subplot(233)
        plt.axhline(y=chi2_interval_max_0001, color='k', linestyle=':', label='0.1% threshold')
        plt.axhline(y=chi2_interval_max_001, color='k', linestyle='--', label='1% threshold')
        plt.title('OCM2, State 1')
        plt.legend(loc='upper right')
        for i in range(0, 100):
            a3 = ax3.plot(d, A2_pre[:, i], linewidth=1)
       
        # State 2
        ax4 = fig.add_subplot(234)
        plt.axhline(y=chi2_interval_max_0001, color='k', linestyle=':', label='0.1% threshold')
        plt.axhline(y=chi2_interval_max_001, color='k', linestyle='--', label='1% threshold')
        plt.title('OCM0, State 2')
        plt.legend(loc='upper right')
        for i in range(0, 100):
            a4 = ax4.plot(d, A0_post[:, i], linewidth=1)
       
        ax5 = fig.add_subplot(235)
        plt.axhline(y=chi2_interval_max_0001, color='k', linestyle=':', label='0.1% threshold')
        plt.axhline(y=chi2_interval_max_001, color='k', linestyle='--', label='1% threshold')
        plt.title('OCM1, State 2')
        plt.legend(loc='upper right')
        for i in range(0, 100):
            a5 = ax5.plot(d, A1_post[:, i], linewidth=1)
       
        ax6 = fig.add_subplot(236)
        plt.axhline(y=chi2_interval_max_0001, color='k', linestyle=':', label='0.1% threshold')
        plt.axhline(y=chi2_interval_max_001, color='k', linestyle='--', label='1% threshold')
        plt.title('OCM2, State 2')
        plt.legend(loc='upper right')
        for i in range(0, 100):
            a6 = ax6.plot(d, A2_post[:, i], linewidth=1)
       
        fig.show()
        f_name = 'Abnomality_'+Sub_run_name+'.png'
        plt.savefig(f_name)
    '''

    ### Count the number of traces above threshold ###
    # OCM0
    for bh in range(0, num_bh):
        for p in range(0, t_sub_removed):
            flag0 = 0
            for depth in range(0, s_new):
                # if not detected yet
                if flag0 < 1:  # OCM0
                    # check every depth and count if it's larger than the threshold
                    if A0[depth, bh * t_sub_removed + p] > chi2_interval_max_0001:
                        out0_test[fidx, bh] = out0_test[fidx, bh] + 1
                        flag0 = 1
        out0_test[fidx, bh] = out0_test[fidx, bh] / t_sub_removed  # Show in percentage

    ##OCM1 and OCM2
    for bh in range(0, num_bh):
        for p in range(0, t_sub):
            flag1 = 0
            flag2 = 0
            for depth in range(0, s_new):
                # if not detected yet
                if flag1 < 1:  # OCM1
                    if A1[depth, bh * t_sub + p] > chi2_interval_max_0001:
                        out1_test[fidx, bh] = out1_test[fidx, bh] + 1
                        flag0 = 1
                if flag2 < 1:  # OCM2
                    if A2[depth, bh * t_sub + p] > chi2_interval_max_0001:
                        out2_test[fidx, bh] = out2_test[fidx, bh] + 1
                        flag2 = 1

        out1_test[fidx, bh] = out1_test[fidx, bh] / t_sub  # Show in percentage
        out2_test[fidx, bh] = out2_test[fidx, bh] / t_sub  # Show in percentage