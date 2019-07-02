'''
This code show mean+SD for all bh before water intake
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
#sr_list = ['s1r1']
sr_list = ['s2r1']

for fidx in range(0, np.size(sr_list)):
    Sub_run = sr_list[fidx]
    sr_name = 'Raw_det_ocm012_' + Sub_run + '.pkl'  # raw
    # sr_name = 'ocm012_' + Sub_run + '.pkl'  # filtered

    with open(sr_name, 'rb') as f:
        ocm0_all, ocm1_all, ocm2_all = pickle.load(f)

    print(ocm0_all.shape)
    d = np.linspace(0, ocm0_all.shape[0] - 1, ocm0_all.shape[0])
    depth = ocm0_all.shape[0]
    num_max = ocm0_all.shape[1]  # Total num of traces
    bh = int(num_max // 5)

    bh_train = 2
    bh_test = 10 - bh_train

    # Remove "10min after" component
    ocm0_ba = np.zeros([depth, 2 * ocm0_all.shape[1]])  # 2nd dim is "before" and "after"
    ocm1_ba = np.zeros([depth, 2 * ocm0_all.shape[1]])  # 2nd dim is "before" and "after"
    ocm2_ba = np.zeros([depth, 2 * ocm0_all.shape[1]])  # 2nd dim is "before" and "after"

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

    # Split data to "bh1, bh2, bh3, bh4, and bh5"
    ocm_train = np.zeros([depth, bh*bh_train, 3])
    ocm_ba_6 = np.zeros([depth, bh, 3])

    ocm_train = ocm_ba[:, 0:bh*bh_train, :]
    ocm_ba_6 = ocm_ba[:, bh*5:bh*6, :]
    print('ocm_train', ocm_train.shape)

    # Transpose
    ocm_train = np.einsum('abc->bac', ocm_train)
    ocm_ba_6 = np.einsum('abc->bac', ocm_ba_6)
    print('ocm_train', ocm_train.shape)

    # Initialize mean and SD
    ocm_train_m = np.zeros([depth, 3])
    ocm_6_m = np.zeros([depth, 3])
    ocm_train_sd = np.zeros([depth, 3])
    ocm_6_sd = np.zeros([depth, 3])

    # Calculate mean of each bh
    for ocm in range(0, 3):
        ocm_train_m[:, ocm] = np.mean(ocm_train[:, :, ocm], axis=0)
        ocm_6_m[:, ocm] = np.mean(ocm_ba_6[:, :, ocm], axis=0)
        ocm_train_sd[:, ocm] = np.std(ocm_train[:, :, ocm], axis=0)
        ocm_6_sd[:, ocm] = np.std(ocm_ba_6[:, :, ocm], axis=0)
        '''
        # ========================Visualize==============================================
        d = np.linspace(0, ocm0_all.shape[0] - 1, ocm0_all.shape[0])
        fig = plt.figure(figsize=(18, 3))
        # This part shows raw signals
        ## Before
        # OCM0
        ocm = 0
        ax1 = fig.add_subplot(131)
        a0 = ax1.plot(d, ocm_train_m[:, ocm], 'b', linewidth=2, label="Train")  # Train
        plt.fill_between(range(ocm_train_m.shape[0]), ocm_train_m[:,ocm]-3*ocm_train_sd[:,ocm], ocm_train_m[:,ocm]+3*ocm_train_sd[:,ocm], alpha=.4)
        a1 = ax1.plot(d, ocm_6_m[:, ocm], 'r', linewidth=2, label="Test")  # BH6
        plt.fill_between(range(ocm_6_m.shape[0]), ocm_6_m[:,ocm]-3*ocm_6_sd[:,ocm], ocm_6_m[:,ocm]+3*ocm_6_sd[:,ocm], alpha=.4)
        ax1.set_title('OCM0, Train vs bh6')
        ax1.set_ylim(-2,2)
        ax1.set_xlabel('Depth')
        ax1.set_ylabel('Intensity')
        plt.legend(loc='lower right')
        # OCM1
        ocm = 1
        ax2 = fig.add_subplot(132)
        a0 = ax2.plot(d, ocm_train_m[:, ocm], 'b', linewidth=2, label="Train")  # Train
        plt.fill_between(range(ocm_train_m.shape[0]), ocm_train_m[:,ocm]-3*ocm_train_sd[:,ocm], ocm_train_m[:,ocm]+3*ocm_train_sd[:,ocm], alpha=.4)
        a1 = ax2.plot(d, ocm_6_m[:, ocm], 'r', linewidth=2, label="Test")  # BH6
        plt.fill_between(range(ocm_6_m.shape[0]), ocm_6_m[:,ocm]-3*ocm_6_sd[:,ocm], ocm_6_m[:,ocm]+3*ocm_6_sd[:,ocm], alpha=.4)
        ax2.set_title('OCM1, Train vs bh6')
        ax2.set_ylim(-2,2)
        ax2.set_xlabel('Depth')
        plt.legend(loc='lower right')
        # OCM2
        ocm = 2
        ax3 = fig.add_subplot(133)
        a0 = ax3.plot(d, ocm_train_m[:, ocm], 'b', linewidth=2, label="Train")  # Train
        plt.fill_between(range(ocm_train_m.shape[0]), ocm_train_m[:,ocm]-3*ocm_train_sd[:,ocm], ocm_train_m[:,ocm]+3*ocm_train_sd[:,ocm], alpha=.4)
        a1 = ax3.plot(d, ocm_6_m[:, ocm], 'r', linewidth=2, label="Test")  # BH6
        plt.fill_between(range(ocm_6_m.shape[0]), ocm_6_m[:,ocm]-3*ocm_6_sd[:,ocm], ocm_6_m[:,ocm]+3*ocm_6_sd[:,ocm], alpha=.4)
        ax3.set_title('OCM2, Train vs bh6')
        ax3.set_ylim(-2,2)
        ax3.set_xlabel('Depth')
        plt.legend(loc='lower right')

        fig.tight_layout()
        fig.show()
        f_name = 'Mean_3SD_Train_Test_' + Sub_run + '_.png'
        plt.savefig(f_name)
        '''



    # ========================Fig for AAPM==============================================
    d = np.linspace(0, ocm0_all.shape[0] - 1, ocm0_all.shape[0])
    d_cm = np.linspace(2.3, 4.875, depth)
    fig = plt.figure(figsize=(7, 3))
    ax1 = fig.add_subplot(111)

    ax2 = ax1.twiny()

    ocm = 1  # OCM1
    ax1.plot(d_cm, ocm_train_m[:, ocm], 'b', linewidth=2.5, label="Train, mean")  # Train
    ax2.fill_between(range(ocm_train_m.shape[0]), ocm_train_m[:,ocm]-3*ocm_train_sd[:,ocm], ocm_train_m[:,ocm]+3*ocm_train_sd[:,ocm], alpha=.4)
    ax1.plot(d_cm, ocm_6_m[:, ocm], 'r', linewidth=2.5, label="Breath-hold 6, mean")  # BH6
    ax2.fill_between(range(ocm_6_m.shape[0]), ocm_6_m[:,ocm]-3*ocm_6_sd[:,ocm], ocm_6_m[:,ocm]+3*ocm_6_sd[:,ocm], alpha=.4)
    ax1.set_title('OCM1, Train vs bh6')
    ax1.set_ylim(-2,2)
    ax2.set_ylim(-2,2)
    ax1.set_ylabel('Intensity (a.u.)')
    ax1.set_xlabel('Depth (cm)')
    ax1.legend(loc='upper right')
    ax2.axis('off')

    fig.tight_layout()
    fig.show()
    f_name = 'Mean_3SD_AAPM_' + Sub_run + '.png'
    plt.savefig(f_name)