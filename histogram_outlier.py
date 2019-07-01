'''
This code draw histogram for outlier frequency
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import pickle
import seaborn as sns
import csv

plt.rcParams['font.family'] ='sans-serif'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

sr_list = ['s1r1', 's1r2', 's2r1', 's2r2', 's3r1', 's3r2']
#sr_list = ['s1r1']
tole = 10  # tolerance level

bh_train = 2
bh_test = 10 - bh_train



for fidx in range(0, np.size(sr_list)):
    Sub_run = sr_list[fidx]
    sr_name = 'Raw_det_ocm012_' + Sub_run + '.pkl'  # raw

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

    # Split data to "train" and "test". Here, only bh=1 is "train".
    ocm_train = np.zeros([depth, bh * bh_train, 3])
    ocm_test = np.zeros([depth, bh * bh_test, 3])

    ocm_train = ocm_ba[:, 0:bh * bh_train, :]
    ocm_test = ocm_ba[:, bh * bh_train:bh * 10, :]
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
    ocm_test_m = np.zeros([depth, 3])
    ocm_test_sd = np.zeros([depth, 3])

    # Calculate mean of "train"
    for ocm in range(0, 3):
        ocm_train_m[:, ocm] = np.mean(ocm_train[:, :, ocm], axis=0)
        ocm_train_sd[:, ocm] = np.std(ocm_train[:, :, ocm], axis=0)

    # Calculate mean of "test" (each bh separately)
    ocm_bh_test = np.zeros([bh, depth, 3])
    total = np.zeros([bh_test, bh * bh_train, 3])
    for bh_cnt in range(0, bh_test):
        output_test = [0, 0, 0]
        for ocm in range(0, 3):
            ocm_bh_test[:, :, ocm] = ocm_test[bh_cnt * bh:(bh_cnt + 1) * bh, :, ocm]
            ocm_test_m[:, ocm] = np.mean(ocm_test[bh_cnt * bh:(bh_cnt + 1) * bh, :, ocm], axis=0)
            ocm_test_sd[:, ocm] = np.std(ocm_test[bh_cnt * bh:(bh_cnt + 1) * bh, :, ocm], axis=0)

        ## Check performance of "test" set
        for num in range(0, bh):
            # If any of the depth is out of the envelope, flag will be 1.
            flag = [0, 0, 0]
            # Detect out of envelope
            for d in range(0, depth):
                for ocm in range(0, 3):
                    mean = ocm_train_m[d, ocm]
                    sd = ocm_train_sd[d, ocm]
                    # if (before < mean-3SD) or (mean+3SD < before)
                    if ((ocm_bh_test[num, d, ocm] < (mean - 3 * sd)) or ((mean + 3 * sd) < ocm_bh_test[num, d, ocm])):
                        total[bh_cnt, num, ocm] = total[bh_cnt, num, ocm] + 1

    # ========================Visualize Distribution ==============================================
    binwidth = 0.03
    fig2 = plt.figure(figsize=(12, 8))
    ax0 = fig2.add_subplot(331)
    ocm = 0
    ax0.hist(total[2, :, ocm], bins=np.arange(min(total[2, :, ocm]), max(total[2, :, ocm]) + binwidth, binwidth), color='Blue', label="bh=3", alpha=0.5)
    ax0.hist(total[3, :, ocm], bins=np.arange(min(total[3, :, ocm]), max(total[3, :, ocm]) + binwidth, binwidth), color='Red', label="bh=4", alpha=0.5)
    ax0.set_title('OCM0, bh3,4')
    ax0.set_xlabel('Number of outliers')
    ax0.set_ylabel('Frequency')
    ax0.set_yscale('log')
    plt.legend(loc='best')

    ax0 = fig2.add_subplot(332)
    ocm = 1
    ax0.hist(total[2, :, ocm], bins=np.arange(min(total[2, :, ocm]), max(total[2, :, ocm]) + binwidth, binwidth), color='Blue', label="bh=3", alpha=0.5)
    ax0.hist(total[3, :, ocm], bins=np.arange(min(total[3, :, ocm]), max(total[3, :, ocm]) + binwidth, binwidth), color='Red', label="bh=4", alpha=0.5)
    ax0.set_title('OCM1, bh3,4')
    ax0.set_xlabel('Number of outliers')
    ax0.set_ylabel('Frequency')
    ax0.set_yscale('log')
    plt.legend(loc='best')

    ax0 = fig2.add_subplot(333)
    ocm = 2
    ax0.hist(total[2, :, ocm], bins=np.arange(min(total[2, :, ocm]), max(total[2, :, ocm]) + binwidth, binwidth), color='Blue', label="bh=3", alpha=0.5)
    ax0.hist(total[3, :, ocm], bins=np.arange(min(total[3, :, ocm]), max(total[3, :, ocm]) + binwidth, binwidth), color='Red', label="bh=4", alpha=0.5)
    ax0.set_title('OCM2, bh3,4')
    ax0.set_xlabel('Number of outliers')
    ax0.set_ylabel('Frequency')
    ax0.set_yscale('log')
    plt.legend(loc='best')

    ax0 = fig2.add_subplot(334)
    ocm = 0
    ax0.hist(total[4, :, ocm], bins=np.arange(min(total[4, :, ocm]), max(total[4, :, ocm]) + binwidth, binwidth), color='Blue', label="bh=5", alpha=0.5)
    ax0.hist(total[5, :, ocm], bins=np.arange(min(total[5, :, ocm]), max(total[5, :, ocm]) + binwidth, binwidth), color='Red', label="bh=6", alpha=0.5)
    ax0.set_title('OCM0, bh5,6')
    ax0.set_xlabel('Number of outliers')
    ax0.set_ylabel('Frequency')
    ax0.set_yscale('log')
    plt.legend(loc='best')

    ax0 = fig2.add_subplot(335)
    ocm = 1
    ax0.hist(total[4, :, ocm], bins=np.arange(min(total[4, :, ocm]), max(total[4, :, ocm]) + binwidth, binwidth), color='Blue', label="bh=5", alpha=0.5)
    ax0.hist(total[5, :, ocm], bins=np.arange(min(total[5, :, ocm]), max(total[5, :, ocm]) + binwidth, binwidth), color='Red', label="bh=6", alpha=0.5)
    ax0.set_title('OCM1, bh5,6')
    ax0.set_xlabel('Number of outliers')
    ax0.set_ylabel('Frequency')
    ax0.set_yscale('log')
    plt.legend(loc='best')

    ax0 = fig2.add_subplot(336)
    ocm = 2
    ax0.hist(total[4, :, ocm], bins=np.arange(min(total[4, :, ocm]), max(total[4, :, ocm]) + binwidth, binwidth), color='Blue', label="bh=5", alpha=0.5)
    ax0.hist(total[5, :, ocm], bins=np.arange(min(total[5, :, ocm]), max(total[5, :, ocm]) + binwidth, binwidth), color='Red', label="bh=6", alpha=0.5)
    ax0.set_title('OCM2, bh5,6')
    ax0.set_xlabel('Number of outliers')
    ax0.set_ylabel('Frequency')
    ax0.set_yscale('log')
    plt.legend(loc='best')

    ax0 = fig2.add_subplot(337)
    ocm = 0
    ax0.hist(total[6, :, ocm], bins=np.arange(min(total[6, :, ocm]), max(total[6, :, ocm]) + binwidth, binwidth), color='Blue', label="bh=7", alpha=0.5)
    ax0.hist(total[7, :, ocm], bins=np.arange(min(total[7, :, ocm]), max(total[7, :, ocm]) + binwidth, binwidth), color='Red', label="bh=8", alpha=0.5)
    ax0.set_title('OCM0, bh7,8')
    ax0.set_xlabel('Number of outliers')
    ax0.set_ylabel('Frequency')
    ax0.set_yscale('log')
    plt.legend(loc='best')

    ax0 = fig2.add_subplot(338)
    ocm = 1
    ax0.hist(total[6, :, ocm], bins=np.arange(min(total[6, :, ocm]), max(total[6, :, ocm]) + binwidth, binwidth), color='Blue', label="bh=7", alpha=0.5)
    ax0.hist(total[7, :, ocm], bins=np.arange(min(total[7, :, ocm]), max(total[7, :, ocm]) + binwidth, binwidth), color='Red', label="bh=8", alpha=0.5)
    ax0.set_title('OCM1, bh7,8')
    ax0.set_xlabel('Number of outliers')
    ax0.set_ylabel('Frequency')
    ax0.set_yscale('log')
    plt.legend(loc='best')

    ax0 = fig2.add_subplot(339)
    ocm = 2
    ax0.hist(total[6, :, ocm], bins=np.arange(min(total[6, :, ocm]), max(total[6, :, ocm]) + binwidth, binwidth), color='Blue', label="bh=7", alpha=0.5)
    ax0.hist(total[7, :, ocm], bins=np.arange(min(total[7, :, ocm]), max(total[7, :, ocm]) + binwidth, binwidth), color='Red', label="bh=8", alpha=0.5)
    ax0.set_title('OCM2, bh7,8')
    ax0.set_xlabel('Number of outliers')
    ax0.set_ylabel('Frequency')
    ax0.set_yscale('log')
    plt.legend(loc='best')



    fig2.tight_layout()
    fig2.show()
    f_name = 'Histogram_' + Sub_run + '.png'
    plt.savefig(f_name)