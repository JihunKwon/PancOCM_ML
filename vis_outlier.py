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

#sr_list = ['s1r1', 's1r2', 's2r1', 's2r2', 's3r1', 's3r2']
sr_list = ['s1r1']
tole = 10  # tolerance level

bh_train = 2
bh_test = 10 - bh_train



for fidx in range(0, np.size(sr_list)):
    Sub_run = sr_list[fidx]
    #sr_name = 'Raw_det_ocm012_' + Sub_run + '.pkl'  # raw
    sr_name = 'ocm012_' + Sub_run + '.pkl'  # filtered

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
        count = [0, 0, 0]
        out_bh_test_new = np.zeros([bh, depth, 3])
        for num in range(0, bh):
            # If any of the depth is out of the envelope, flag will be 1.
            flag = [0, 0, 0]
            # Detect out of envelope
            for d in range(0, depth):
                for ocm in range(0, 3):
                    mean = ocm_train_m[d, ocm]
                    sd = ocm_train_sd[d, ocm]
                    if flag[ocm] < 1:
                        # if (before < mean-3SD) or (mean+3SD < before)
                        if ((ocm_bh_test[num, d, ocm] < (mean - 3 * sd)) or ((mean + 3 * sd) < ocm_bh_test[num, d, ocm])):
                            out_bh_test_new[count[ocm], :, ocm] = ocm_bh_test[num, :, ocm]
                            count[ocm] = count[ocm] + 1
                            flag[ocm] = 1

        # ========================Visualize Outlier trace ==============================================
        fig = plt.figure(figsize=(18, 3))
        d = np.linspace(0, ocm0_all.shape[0] - 1, ocm0_all.shape[0])
        # This part shows raw signals
        ## Mean+3SD
        # OCM0
        ocm = 0
        ax0 = fig.add_subplot(131)
        a0 = ax0.plot(d, ocm_train_m[:, ocm], 'b', linewidth=2, label="Before")  # Before water
        plt.fill_between(range(ocm_train_m.shape[0]), ocm_train_m[:, ocm] - 3 * ocm_train_sd[:, ocm], ocm_train_m[:, ocm] + 3 * ocm_train_sd[:, ocm], alpha=.3)
        a1 = ax0.plot(d, ocm_test_m[:, ocm], 'r', linewidth=2, label="After")  # After water
        plt.fill_between(range(ocm_test_m.shape[0]), ocm_test_m[:, ocm] - 3 * ocm_test_sd[:, ocm], ocm_test_m[:, ocm] + 3 * ocm_test_sd[:, ocm], alpha=.3)
        for i in range(0, 100):
            a1_1 = ax0.plot(d, out_bh_test_new[i, :, ocm], 'r', linewidth=0.5)  # Outlier traces
        ax0.set_title('OCM0, 3SD')
        ax0.set_ylim(-2, 2)
        ax0.set_xlabel('Depth')
        ax0.set_ylabel('Intensity')
        plt.legend(loc='best')
        # OCM1
        ocm = 1
        ax1 = fig.add_subplot(132)
        a0 = ax1.plot(d, ocm_train_m[:, ocm], 'b', linewidth=2, label="Before")  # Before water
        plt.fill_between(range(ocm_train_m.shape[0]), ocm_train_m[:, ocm] - 3 * ocm_train_sd[:, ocm], ocm_train_m[:, ocm] + 3 * ocm_train_sd[:, ocm], alpha=.3)
        a1 = ax1.plot(d, ocm_test_m[:, ocm], 'r', linewidth=2, label="After")  # After water
        plt.fill_between(range(ocm_test_m.shape[0]), ocm_test_m[:, ocm] - 3 * ocm_test_sd[:, ocm], ocm_test_m[:, ocm] + 3 * ocm_test_sd[:, ocm], alpha=.3)
        for i in range(0, 100):
            a1_1 = ax1.plot(d, out_bh_test_new[i, :, ocm], 'r', linewidth=0.5)  # Outlier traces
        ax1.set_title('OCM1, 3SD')
        ax1.set_ylim(-2, 2)
        ax1.set_xlabel('Depth')
        # OCM2
        ocm = 2
        ax2 = fig.add_subplot(133)
        a0 = ax2.plot(d, ocm_train_m[:, ocm], 'b', linewidth=2, label="Before")  # Before water
        plt.fill_between(range(ocm_train_m.shape[0]), ocm_train_m[:, ocm] - 3 * ocm_train_sd[:, ocm], ocm_train_m[:, ocm] + 3 * ocm_train_sd[:, ocm], alpha=.3)
        a1 = ax2.plot(d, ocm_test_m[:, ocm], 'r', linewidth=2, label="After")  # After water
        plt.fill_between(range(ocm_test_m.shape[0]), ocm_test_m[:, ocm] - 3 * ocm_test_sd[:, ocm], ocm_test_m[:, ocm] + 3 * ocm_test_sd[:, ocm], alpha=.3)
        for i in range(0, 100):
            a1_1 = ax2.plot(d, out_bh_test_new[i, :, ocm], 'r', linewidth=0.5)  # Outlier traces
        ax2.set_title('OCM2, 3SD')
        ax2.set_ylim(-2, 2)
        ax2.set_xlabel('Depth')

        fig.tight_layout()
        fig.show()
        f_name = 'Mean_3SD_Outlier_filt_' + Sub_run + '_bh' + str(bh_cnt)+ '.png'
        plt.savefig(f_name)