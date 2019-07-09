'''
This code detects outlier and calculate TPR, based on "multiplying and difference"-based method
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import pickle
import seaborn as sns
import csv
from sklearn import preprocessing

plt.rcParams['font.family'] ='sans-serif'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

sr_list = ['s1r1', 's1r2', 's2r1', 's2r2', 's3r1', 's3r2']
#sr_list = ['s2r2']
tole = 10  # tolerance level

bh_train = 2
bh_test = 10 - bh_train
batch = 1000  # subset of OCM traces used to get local average
area_level = 0.5  # What FPR you want to get for first bh of test set.


for fidx in range(0, np.size(sr_list)):
    Sub_run = sr_list[fidx]
    sr_name = 'Raw_det_ocm012_' + Sub_run + '.pkl'  # raw
    #sr_name = 'ocm012_' + Sub_run + '.pkl'  # filtered

    with open(sr_name, 'rb') as f:
        ocm0_all, ocm1_all, ocm2_all = pickle.load(f)

    print(ocm0_all.shape)
    d = np.linspace(0, ocm0_all.shape[0] - 1, ocm0_all.shape[0])
    depth = ocm0_all.shape[0]
    num_max = ocm0_all.shape[1]  # Total num of traces
    bh = int(num_max // 5)

    # Remove "10min after" component
    ocm0_ba = np.zeros([depth, 2 * ocm0_all.shape[1]])  # 2nd dim is "before" and "after"
    ocm1_ba = np.zeros([depth, 2 * ocm0_all.shape[1]])
    ocm2_ba = np.zeros([depth, 2 * ocm0_all.shape[1]])

    ocm0_ba[:, 0:num_max] = ocm0_all[:, :, 0]  # add "before"
    ocm1_ba[:, 0:num_max] = ocm1_all[:, :, 0]
    ocm2_ba[:, 0:num_max] = ocm2_all[:, :, 0]
    ocm0_ba[:, num_max:2 * num_max] = ocm0_all[:, :, 1]  # add "after"
    ocm1_ba[:, num_max:2 * num_max] = ocm1_all[:, :, 1]
    ocm2_ba[:, num_max:2 * num_max] = ocm2_all[:, :, 1]

    # Add to one variable
    ocm_ba = np.zeros([depth, 2 * num_max, 3])
    ocm_ba[:, :, 0] = ocm0_ba[:, :]
    ocm_ba[:, :, 1] = ocm1_ba[:, :]
    ocm_ba[:, :, 2] = ocm2_ba[:, :]

    # Multiply to remove +- component. We only need "scalar" here.
    ocm_ba_multi = np.zeros(ocm_ba.shape)
    for ocm in range(0,3):
        for num in range(0, num_max):
            for d in range(0, depth):
                ocm_ba_multi[d, num, ocm] = ocm_ba[d, num, ocm] * ocm_ba[d, num, ocm]

    # Split data to "train" and "test". Here, only bh=1 is "train".
    ocm_train = np.zeros([depth, bh * bh_train, 3])
    ocm_test = np.zeros([depth, bh * bh_test, 3])

    ocm_train = ocm_ba_multi[:, 0:bh * bh_train, :]
    ocm_test = ocm_ba_multi[:, bh * bh_train:bh * 10, :]
    print('ocm_train', ocm_train.shape)
    print('ocm_test', ocm_test.shape)

    # Transpose
    ocm_train = np.einsum('abc->bac', ocm_train)
    ocm_test = np.einsum('abc->bac', ocm_test)
    print('ocm_train', ocm_train.shape)
    print('ocm_test', ocm_test.shape)

    # Initialize mean and SD
    ocm_train_m = np.zeros([depth, 3])
    ocm_train_sd = np.zeros([depth, 3])
    ocm_train_mini_m = np.zeros([round(bh_train*bh/batch), depth, 3])
    ocm_train_diff_mini = np.zeros([round(bh_train*bh/batch), depth, 3])
    train_diff_m = np.zeros([depth, 3])
    train_diff_sd = np.zeros([depth, 3])
    ocm_train_diff = np.zeros([bh_train*bh, depth, 3])
    ocm_test_diff = np.zeros([bh_test*bh, depth, 3])
    print('ocm_train_mini_m', ocm_train_mini_m.shape)

    # get mean
    for ocm in range(0, 3):
        # Calculate overall mean of "train"
        ocm_train_m[:, ocm] = np.mean(ocm_train[:, :, ocm], axis=0)
        ocm_train_sd[:, ocm] = np.std(ocm_train[:, :, ocm], axis=0)
        # Calculate mean every batch
        for mini in range(0, round(bh_train*bh/batch)):
            ocm_train_mini_m[mini, :, ocm] = np.mean(ocm_train[mini*batch:(mini+1)*batch, :, ocm], axis=0)

    # Calculate subtraction (mean_all - mean_batch)
    for mini in range(0, round(bh_train*bh/batch)):
        for d in range(0, depth):
            ocm_train_diff_mini[mini, d, :] = ocm_train_m[d, :] - ocm_train_mini_m[mini, d, :]

    # Get mean and SD of difference
    for ocm in range(0, 3):
        train_diff_m[:, ocm] = np.mean(ocm_train_diff_mini[:, :, ocm], axis=0)
        train_diff_sd[:, ocm] = np.std(ocm_train_diff_mini[:, :, ocm], axis=0)

    # [Train] Calculate subtraction (mean_all - mean_each)
    for n in range(0, bh_train*bh):
        for d in range(0, depth):
            ocm_train_diff[n, d, :] = ocm_train_m[d, :] - ocm_train[n, d, :]
    # [Test] Calculate subtraction (mean_all - mean_each)
    for n in range(0, bh_test*bh):
        for d in range(0, depth):
            ocm_test_diff[n, d, :] = ocm_train_m[d, :] - ocm_test[n, d, :]


    ## Check performance of "train" set
    outside_train = [0, 0, 0]
    outside_train_area = np.zeros([bh * bh_train, 3])
    diff = 0
    for num in range(0, bh * bh_train):  # each traces
        # If any of the depth is out of the envelope, flag will be 1.
        flag = [0, 0, 0]
        # Detect out of envelope
        for d in range(0, depth):
            for ocm in range(0, 3):
                mean = train_diff_m[d, ocm]
                sd = train_diff_sd[d, ocm]
                # if out of envelope
                if ((ocm_train_diff[num, d, ocm] < (mean - 3 * sd)) or ((mean + 3 * sd) < ocm_train_diff[num, d, ocm])):
                    diff = ((mean - 3 * sd) - ocm_train_diff[num, d, ocm])
                    if diff < 0:  # if upper part is out of envelope
                        diff = (ocm_train_diff[num, d, ocm] - (mean + 3 * sd))
                    outside_train_area[num, ocm] = outside_train_area[num, ocm] + diff


    print('ocm_train_diff', ocm_train_diff.shape)
    TP = np.zeros([10, 3])  # [bh_test, ocm]
    FN = np.zeros([10, 3])
    TN = np.zeros([10, 3])
    FP = np.zeros([10, 3])
    outside_test = np.zeros([bh_test, 3])
    outside_test_area = np.zeros([bh, 3])
    train_area_bh = np.zeros([10, bh, 3])
    test_area_bh = np.zeros([10, bh * bh_test, 3])
    area_sort = [0, 0, 0]
    area_thr = [0, 0, 0]
    target_idx = [0, 0, 0]
    area_total = [0, 0, 0]
    flag = [0, 0, 0]
    area_cumulat = np.zeros([bh, 3])
    area_sort = np.zeros(outside_test_area.shape)

    print('###### Get Area Threshold ######')
    # Using first bh of test set, set a threshold which defines outlier
    bh_cnt = 0
    for num in range(0, bh):
        # If any of the depth is out of the envelope, flag will be 1.
        flag = [0, 0, 0]
        # Detect out of envelope
        for d in range(0, depth):
            for ocm in range(0, 3):
                mean = train_diff_m[d, ocm]
                sd = train_diff_sd[d, ocm]
                # if out of envelope
                if ((ocm_test_diff[num, d, ocm] < (mean - 3 * sd)) or ((mean + 3 * sd) < ocm_test_diff[num, d, ocm])):
                    diff = ((mean - 3 * sd) - ocm_test_diff[num, d, ocm])  # if lower part is out of envelope
                    if diff < 0:  # if upper part is out of envelope
                        diff = (ocm_test_diff[num, d, ocm] - (mean + 3 * sd))
                    outside_test_area[num, ocm] = outside_test_area[num, ocm] + diff

    print('outside_test_area', outside_test_area.shape)
    # start searching threshold here
    flag = [0, 0, 0]
    for ocm in range(0, 3):
        area_sort[:, ocm] = sorted(outside_test_area[:, ocm])  # sort
        area_total[ocm] = sum(outside_test_area[:, ocm])
        print('area_total', ocm, ':', area_total[0])

        for a in range(0, outside_test_area.shape[0]):
            # pile up area and get when it becomes equal to threshold
            area_cumulat[a, ocm] = area_cumulat[a-1, ocm] + area_sort[a, ocm]
            if flag[ocm] < 1:
                if (area_cumulat[a, ocm] > area_total[ocm] * area_level):
                    target_idx[ocm] = a  # This is the "index" of array where the cumulative area meets the area_level
                    flag[ocm] = flag[ocm] + 1

        area_thr[ocm] = area_sort[target_idx[ocm], ocm]
        print('target_idx', ocm, ':', target_idx[ocm])
        print('area_thr', ocm, ': ', area_thr[ocm])


    print('###### Test Start ######')
    outside_test_area_fin = np.zeros([bh * bh_test, 3])
    # Calculate mean of "test" (each bh separately)
    for bh_cnt in range(0, bh_test):
        ## Check performance of "test" set
        for num in range(0, bh):
            # If any of the depth is out of the envelope, flag will be 1.
            flag = [0, 0, 0]
            # Detect out of envelope
            for d in range(0, depth):
                for ocm in range(0, 3):
                    if flag[ocm] < 1:
                        mean = train_diff_m[d, ocm]
                        sd = train_diff_sd[d, ocm]
                        # if out of envelope
                        if ((ocm_test_diff[num, d, ocm] < (mean - 3 * sd)) or ((mean + 3 * sd) < ocm_test_diff[num, d, ocm])):
                            diff = ((mean - 3 * sd) - ocm_test_diff[num, d, ocm])
                            if diff < 0:  # if upper part is out of envelope
                                diff = (ocm_test_diff[num, d, ocm] - (mean + 3 * sd))
                            outside_test_area_fin[num, ocm] = outside_test_area_fin[num, ocm] + diff
                            # if area outside of envelope for this trace became larger than the predefined area_thr, consider it as an outlier
                            if outside_test_area_fin[num, ocm] >= area_thr[ocm]:
                                outside_test[bh_cnt, ocm] = outside_test[bh_cnt, ocm] + 1
                                flag[ocm] = flag[ocm] + 1
    ## Gather each bh data
        if 0 <= bh_cnt < (bh_test-5):  # if "before water"
            for ocm in range(0, 3):
                TN[bh_cnt, ocm] = bh - outside_test[bh_cnt, ocm]  # Before, change not detected
                FP[bh_cnt, ocm] = outside_test[bh_cnt, ocm]  # Before, change detected
                test_area_bh[bh_cnt, :, ocm] = outside_test_area_fin[:, ocm]
        elif (bh_test-5) <= bh_cnt:  # if "after water"
            for ocm in range(0, 3):
                TP[bh_cnt, ocm] = outside_test[bh_cnt, ocm]  # After, change detected
                FN[bh_cnt, ocm] = bh - outside_test[bh_cnt, ocm]  # After, change not detected
                test_area_bh[bh_cnt, :, ocm] = outside_test_area_fin[:, ocm]

    print('#### Test Ends ####')
    print('test_area_bh:', test_area_bh.shape)

    '''
    ## Visualize count and area of outlier
    d = np.linspace(0, ocm0_all.shape[0] - 1, ocm0_all.shape[0])
    binwidth = 0.03
    fig2 = plt.figure(figsize=(12, 6))
    ax0 = fig2.add_subplot(234)
    ocm = 0
    ax0.hist(test_area_bh[3, :, ocm], bins=np.arange(min(test_area_bh[3, :, ocm]), max(test_area_bh[3, :, ocm]) + binwidth, binwidth), label='aft1', alpha=0.5)
    ax0.hist(test_area_bh[4, :, ocm], bins=np.arange(min(test_area_bh[4, :, ocm]), max(test_area_bh[4, :, ocm]) + binwidth, binwidth), label='aft2', alpha=0.5)
    ax0.hist(test_area_bh[5, :, ocm], bins=np.arange(min(test_area_bh[5, :, ocm]), max(test_area_bh[5, :, ocm]) + binwidth, binwidth), label='aft3', alpha=0.5)
    ax0.hist(test_area_bh[6, :, ocm], bins=np.arange(min(test_area_bh[6, :, ocm]), max(test_area_bh[6, :, ocm]) + binwidth, binwidth), label='aft4', alpha=0.5)
    ax0.hist(test_area_bh[7, :, ocm], bins=np.arange(min(test_area_bh[7, :, ocm]), max(test_area_bh[7, :, ocm]) + binwidth, binwidth), label='aft5', alpha=0.5)
    ax0.set_title('Area, OCM0, After')
    ax0.set_xlabel('Area out of envelop')
    ax0.set_ylabel('Frequency')
    #ax0.set_yscale('log')
    Xmin_0, Xmax_0 = ax0.get_xlim()
    Ymin_0, Ymax_0 = ax0.get_ylim()
    ax0.set_xlim(0, Xmax_0)
    ax0.set_ylim(0, 50)
    plt.legend(loc='best')
    print('1')

    ax0 = fig2.add_subplot(235)
    ocm = 1
    ax0.hist(test_area_bh[3, :, ocm], bins=np.arange(min(test_area_bh[3, :, ocm]), max(test_area_bh[3, :, ocm]) + binwidth, binwidth), label='aft1', alpha=0.5)
    ax0.hist(test_area_bh[4, :, ocm], bins=np.arange(min(test_area_bh[4, :, ocm]), max(test_area_bh[4, :, ocm]) + binwidth, binwidth), label='aft2', alpha=0.5)
    ax0.hist(test_area_bh[5, :, ocm], bins=np.arange(min(test_area_bh[5, :, ocm]), max(test_area_bh[5, :, ocm]) + binwidth, binwidth), label='aft3', alpha=0.5)
    ax0.hist(test_area_bh[6, :, ocm], bins=np.arange(min(test_area_bh[6, :, ocm]), max(test_area_bh[6, :, ocm]) + binwidth, binwidth), label='aft4', alpha=0.5)
    ax0.hist(test_area_bh[7, :, ocm], bins=np.arange(min(test_area_bh[7, :, ocm]), max(test_area_bh[7, :, ocm]) + binwidth, binwidth), label='aft5', alpha=0.5)
    ax0.set_title('Area, OCM1, After')
    ax0.set_xlabel('Area out of envelop')
    ax0.set_ylabel('Frequency')
    #ax0.set_yscale('log')
    Xmin_1, Xmax_1 = ax0.get_xlim()
    Ymin_1, Ymax_1 = ax0.get_ylim()
    ax0.set_xlim(0, Xmax_1)
    ax0.set_ylim(0, 50)
    plt.legend(loc='best')
    print('2')

    ax0 = fig2.add_subplot(236)
    ocm = 2
    ax0.hist(test_area_bh[3, :, ocm], bins=np.arange(min(test_area_bh[3, :, ocm]), max(test_area_bh[3, :, ocm]) + binwidth, binwidth), label='aft1', alpha=0.5)
    ax0.hist(test_area_bh[4, :, ocm], bins=np.arange(min(test_area_bh[4, :, ocm]), max(test_area_bh[4, :, ocm]) + binwidth, binwidth), label='aft2', alpha=0.5)
    ax0.hist(test_area_bh[5, :, ocm], bins=np.arange(min(test_area_bh[5, :, ocm]), max(test_area_bh[5, :, ocm]) + binwidth, binwidth), label='aft3', alpha=0.5)
    ax0.hist(test_area_bh[6, :, ocm], bins=np.arange(min(test_area_bh[6, :, ocm]), max(test_area_bh[6, :, ocm]) + binwidth, binwidth), label='aft4', alpha=0.5)
    ax0.hist(test_area_bh[7, :, ocm], bins=np.arange(min(test_area_bh[7, :, ocm]), max(test_area_bh[7, :, ocm]) + binwidth, binwidth), label='aft5', alpha=0.5)
    ax0.set_title('Area, OCM2, After')
    ax0.set_xlabel('Area out of envelop')
    ax0.set_ylabel('Frequency')
    #ax0.set_yscale('log')
    Xmin_2, Xmax_2 = ax0.get_xlim()
    Ymin_2, Ymax_2 = ax0.get_ylim()
    ax0.set_xlim(0, Xmax_2)
    ax0.set_ylim(0, 50)
    plt.legend(loc='best')
    print('3')

    ax0 = fig2.add_subplot(231)
    ocm = 0
    ax0.hist(test_area_bh[0, :, ocm], bins=np.arange(min(test_area_bh[0, :, ocm]), max(test_area_bh[0, :, ocm]) + binwidth, binwidth), label='bef3', alpha=0.5)
    ax0.hist(test_area_bh[1, :, ocm], bins=np.arange(min(test_area_bh[1, :, ocm]), max(test_area_bh[1, :, ocm]) + binwidth, binwidth), label='bef4', alpha=0.5)
    ax0.hist(test_area_bh[2, :, ocm], bins=np.arange(min(test_area_bh[2, :, ocm]), max(test_area_bh[2, :, ocm]) + binwidth, binwidth), label='bef5', alpha=0.5)
    ax0.set_title('Area, OCM0, Before')
    ax0.set_xlabel('Area out of envelop')
    ax0.set_ylabel('Frequency')
    ax0.set_xlim(0, Xmax_0)
    ax0.set_ylim(0, 50)
    plt.legend(loc='best')
    print('4')

    ax0 = fig2.add_subplot(232)
    ocm = 1
    ax0.hist(test_area_bh[0, :, ocm], bins=np.arange(min(test_area_bh[0, :, ocm]), max(test_area_bh[0, :, ocm]) + binwidth, binwidth), label='bef3', alpha=0.5)
    ax0.hist(test_area_bh[1, :, ocm], bins=np.arange(min(test_area_bh[1, :, ocm]), max(test_area_bh[1, :, ocm]) + binwidth, binwidth), label='bef4', alpha=0.5)
    ax0.hist(test_area_bh[2, :, ocm], bins=np.arange(min(test_area_bh[2, :, ocm]), max(test_area_bh[2, :, ocm]) + binwidth, binwidth), label='bef5', alpha=0.5)
    ax0.set_title('Area, OCM1, Before')
    ax0.set_xlabel('Area out of envelop')
    ax0.set_ylabel('Frequency')
    #ax0.set_yscale('log')
    ax0.set_xlim(0, Xmax_1)
    ax0.set_ylim(0, 50)
    plt.legend(loc='best')
    print('5')

    ax0 = fig2.add_subplot(233)
    ocm = 2
    ax0.hist(test_area_bh[0, :, ocm], bins=np.arange(min(test_area_bh[0, :, ocm]), max(test_area_bh[0, :, ocm]) + binwidth, binwidth), label='bef3', alpha=0.5)
    ax0.hist(test_area_bh[1, :, ocm], bins=np.arange(min(test_area_bh[1, :, ocm]), max(test_area_bh[1, :, ocm]) + binwidth, binwidth), label='bef4', alpha=0.5)
    ax0.hist(test_area_bh[2, :, ocm], bins=np.arange(min(test_area_bh[2, :, ocm]), max(test_area_bh[2, :, ocm]) + binwidth, binwidth), label='bef5', alpha=0.5)
    ax0.set_title('Area, OCM2, Before')
    ax0.set_xlabel('Area out of envelop')
    ax0.set_ylabel('Frequency')
    #ax0.set_yscale('log')
    ax0.set_xlim(0, Xmax_2)
    ax0.set_ylim(0, 50)
    plt.legend(loc='best')
    print('6')

    fig2.tight_layout()
    #fig2.show()
    f_name = 'Histogram_cnt_Area_' + Sub_run + '.png'
    plt.savefig(f_name)
    print('Data:', Sub_run, ' has finished')
    '''

    '''
    ## Visualize mean+-SD of "difference"
    d = np.linspace(0, ocm0_all.shape[0] - 1, ocm0_all.shape[0])
    fig = plt.figure(figsize=(18, 4))
    # OCM0
    ocm = 0
    ax1 = fig.add_subplot(131)
    a0 = ax1.plot(d, train_diff_m[:, ocm], 'b', linewidth=2, label='train')
    plt.fill_between(range(train_diff_m.shape[0]), train_diff_m[:, ocm]-3*train_diff_sd[:, ocm], train_diff_m[:, ocm]+3*train_diff_sd[:, ocm], alpha=.2)
    a1 = ax1.plot(d, ocm_test_diff[0, :, ocm], linewidth=2, label='test0')
    a2 = ax1.plot(d, ocm_test_diff[1000, :, ocm], linewidth=2, label='test1')
    a3 = ax1.plot(d, ocm_test_diff[2000, :, ocm], linewidth=2, label='test2')
    ax1.set_title('OCM0, 3SD, Training')
    ax1.set_ylim(-1,1)
    ax1.set_xlabel('Depth')
    ax1.set_ylabel('Difference')
    plt.legend(loc='lower right')
    # OCM1
    ocm = 1
    ax1 = fig.add_subplot(132)
    a0 = ax1.plot(d, train_diff_m[:, ocm], 'b', linewidth=2, label='train')
    plt.fill_between(range(train_diff_m.shape[0]), train_diff_m[:, ocm] - 3 * train_diff_sd[:, ocm], train_diff_m[:, ocm] + 3 * train_diff_sd[:, ocm], alpha=.2)
    a1 = ax1.plot(d, ocm_test_diff[0, :, ocm], linewidth=2, label='test0')
    a2 = ax1.plot(d, ocm_test_diff[100, :, ocm], linewidth=2, label='test1')
    a3 = ax1.plot(d, ocm_test_diff[2000, :, ocm], linewidth=2, label='test2')
    ax1.set_title('OCM1, 3SD, Training')
    ax1.set_ylim(-1, 1)
    ax1.set_xlabel('Depth')
    ax1.set_ylabel('Difference')
    plt.legend(loc='lower right')
    # OCM2
    ocm = 2
    ax1 = fig.add_subplot(133)
    a0 = ax1.plot(d, train_diff_m[:, ocm], 'b', linewidth=2, label='train')
    plt.fill_between(range(train_diff_m.shape[0]), train_diff_m[:, ocm] - 3 * train_diff_sd[:, ocm], train_diff_m[:, ocm] + 3 * train_diff_sd[:, ocm], alpha=.2)
    a1 = ax1.plot(d, ocm_test_diff[0, :, ocm], linewidth=2, label='test0')
    a2 = ax1.plot(d, ocm_test_diff[1000, :, ocm], linewidth=2, label='test1')
    a3 = ax1.plot(d, ocm_test_diff[2000, :, ocm], linewidth=2, label='test2')
    ax1.set_title('OCM2, 3SD, Training')
    ax1.set_ylim(-1, 1)
    ax1.set_xlabel('Depth')
    ax1.set_ylabel('Difference')
    plt.legend(loc='lower right')
    fig.tight_layout()
    fig.show()
    f_name = ('Diff_meanSD_Train.png')
    plt.savefig(f_name)
    '''

    ## Output begins
    print('TP.shape:', TP.shape)
    print('fidx:', Sub_run)
    print('training set')
    print('TN,FP: ', '{:.3f}'.format(bh*bh_test - outside_train[0]), ' ', '{:.3f}'.format(outside_train[0]),
          '{:.3f}'.format(bh*bh_test - outside_train[1]), ' ', '{:.3f}'.format(outside_train[1]),
          '{:.3f}'.format(bh*bh_test - outside_train[2]), ' ', '{:.3f}'.format(outside_train[2]))

    print('TNR,FPR: ', '{:.3f}'.format((bh*bh_test - outside_train[0]) / (bh*bh_test)), ' ', '{:.3f}'.format(outside_train[0] / (bh*bh_test)),
          '{:.3f}'.format((bh*bh_test - outside_train[1]) / (bh*bh_test)), ' ', '{:.3f}'.format(outside_train[1] / (bh*bh_test)),
          '{:.3f}'.format((bh*bh_test - outside_train[2]) / (bh*bh_test)), ' ', '{:.3f}'.format(outside_train[2] / (bh*bh_test)))
    print('End of training data')

    # test result with "before" data (TN and FP)
    for bh_cnt in range(0, bh_test-5):
        print('bh=', bh_cnt + bh_train + 1)
        print('TN,FP: ', '{:.3f}'.format(TN[bh_cnt][0]), ' ', '{:.3f}'.format(FP[bh_cnt][0])
              , ' ', '{:.3f}'.format(TN[bh_cnt][1]), ' ', '{:.3f}'.format(FP[bh_cnt][1])
              , ' ', '{:.3f}'.format(TN[bh_cnt][2]), ' ', '{:.3f}'.format(FP[bh_cnt][2]))

        print('TNR,FPR: ', '{:.3f}'.format(TN[bh_cnt][0] / bh), ' ', '{:.3f}'.format(FP[bh_cnt][0] / bh)
              , ' ', '{:.3f}'.format(TN[bh_cnt][1] / bh), ' ', '{:.3f}'.format(FP[bh_cnt][1] / bh)
              , ' ', '{:.3f}'.format(TN[bh_cnt][2] / bh), ' ', '{:.3f}'.format(FP[bh_cnt][2] / bh))
    print('End of Before water')

    # test result with "after" data (TP and FN)
    for bh_cnt in range(bh_test-5, bh_test):
        print('bh=', bh_cnt + bh_train + 1)
        print('TP,FN: ', '{:.3f}'.format(TP[bh_cnt][0]), ' ', '{:.3f}'.format(FN[bh_cnt][0])
              , ' ', '{:.3f}'.format(TP[bh_cnt][1]), ' ', '{:.3f}'.format(FN[bh_cnt][1])
              , ' ', '{:.3f}'.format(TP[bh_cnt][2]), ' ', '{:.3f}'.format(FN[bh_cnt][2]))

        print('TPR,FNR: ', '{:.3f}'.format(TP[bh_cnt][0] / bh), ' ', '{:.3f}'.format(FN[bh_cnt][0] / bh)
              , ' ', '{:.3f}'.format(TP[bh_cnt][1] / bh), ' ', '{:.3f}'.format(FN[bh_cnt][1] / bh)
              , ' ', '{:.3f}'.format(TP[bh_cnt][2] / bh), ' ', '{:.3f}'.format(FN[bh_cnt][2] / bh))
    print('End of After water')

    f_name = 'result2_' + Sub_run + '_' + str(area_level) + '.csv'
    with open(f_name, mode='w',newline='') as file:
        writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        writer.writerow(['TNR,FPR_train:', '{:.3f}'.format((bh*bh_test - outside_train[0]) / (bh*bh_test)), '{:.3f}'.format(outside_train[0] / (bh*bh_test)),
                             '{:.3f}'.format((bh*bh_test - outside_train[1]) / (bh*bh_test)), '{:.3f}'.format(outside_train[1] / (bh*bh_test)),
                             '{:.3f}'.format((bh*bh_test - outside_train[2]) / (bh*bh_test)), '{:.3f}'.format(outside_train[2] / (bh*bh_test))])

        writer.writerow(['TNR,FPR_test:'])
        for bh_cnt in range(0, bh_test-5):
            writer.writerow([bh_cnt + bh_train + 1, '{:.3f}'.format(TN[bh_cnt][0] / bh), '{:.3f}'.format(FP[bh_cnt][0] / bh),
                                 '{:.3f}'.format(TN[bh_cnt][1] / bh), '{:.3f}'.format(FP[bh_cnt][1] / bh),
                                 '{:.3f}'.format(TN[bh_cnt][2] / bh), '{:.3f}'.format(FP[bh_cnt][2] / bh)])

        writer.writerow(['TPR,FNR:'])
        for bh_cnt in range(bh_test-5, bh_test):
            writer.writerow([bh_cnt + bh_train + 1, '{:.3f}'.format(TP[bh_cnt][0] / bh), '{:.3f}'.format(FN[bh_cnt][0] / bh),
                                 '{:.3f}'.format(TP[bh_cnt][1] / bh), '{:.3f}'.format(FN[bh_cnt][1] / bh),
                                 '{:.3f}'.format(TP[bh_cnt][2] / bh), '{:.3f}'.format(FN[bh_cnt][2] / bh)])
