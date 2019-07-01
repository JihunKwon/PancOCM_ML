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

sr_list = ['s1r1', 's1r2', 's2r1', 's2r2', 's3r1', 's3r2']
#sr_list = ['s1r1']

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
    ocm_ba_1 = np.zeros([depth, bh, 3])
    ocm_ba_2 = np.zeros([depth, bh, 3])
    ocm_ba_3 = np.zeros([depth, bh, 3])
    ocm_ba_4 = np.zeros([depth, bh, 3])
    ocm_ba_5 = np.zeros([depth, bh, 3])

    ocm_ba_1 = ocm_ba[:, bh*0:bh*1, :]
    ocm_ba_2 = ocm_ba[:, bh*1:bh*2, :]
    ocm_ba_3 = ocm_ba[:, bh*2:bh*3, :]
    ocm_ba_4 = ocm_ba[:, bh*3:bh*4, :]
    ocm_ba_5 = ocm_ba[:, bh*4:bh*5, :]

    ocm_ba_6 = ocm_ba[:, bh*5:bh*6, :]
    ocm_ba_7 = ocm_ba[:, bh*6:bh*7, :]
    ocm_ba_8 = ocm_ba[:, bh*7:bh*8, :]
    ocm_ba_9 = ocm_ba[:, bh*8:bh*9, :]
    ocm_ba_10 = ocm_ba[:, bh*9:bh*10, :]
    print('ocm_ba_1', ocm_ba_1.shape)

    # Transpose
    ocm_ba_1 = np.einsum('abc->bac', ocm_ba_1)
    ocm_ba_2 = np.einsum('abc->bac', ocm_ba_2)
    ocm_ba_3 = np.einsum('abc->bac', ocm_ba_3)
    ocm_ba_4 = np.einsum('abc->bac', ocm_ba_4)
    ocm_ba_5 = np.einsum('abc->bac', ocm_ba_5)
    ocm_ba_6 = np.einsum('abc->bac', ocm_ba_6)
    ocm_ba_7 = np.einsum('abc->bac', ocm_ba_7)
    ocm_ba_8 = np.einsum('abc->bac', ocm_ba_8)
    ocm_ba_9 = np.einsum('abc->bac', ocm_ba_9)
    ocm_ba_10 = np.einsum('abc->bac', ocm_ba_10)
    print('ocm_ba_1', ocm_ba_1.shape)

    # Initialize mean and SD
    ocm_1_m = np.zeros([depth, 3])
    ocm_2_m = np.zeros([depth, 3])
    ocm_3_m = np.zeros([depth, 3])
    ocm_4_m = np.zeros([depth, 3])
    ocm_5_m = np.zeros([depth, 3])
    ocm_6_m = np.zeros([depth, 3])
    ocm_7_m = np.zeros([depth, 3])
    ocm_8_m = np.zeros([depth, 3])
    ocm_9_m = np.zeros([depth, 3])
    ocm_10_m = np.zeros([depth, 3])
    ocm_1_sd = np.zeros([depth, 3])
    ocm_2_sd = np.zeros([depth, 3])
    ocm_3_sd = np.zeros([depth, 3])
    ocm_4_sd = np.zeros([depth, 3])
    ocm_5_sd = np.zeros([depth, 3])
    ocm_6_sd = np.zeros([depth, 3])
    ocm_7_sd = np.zeros([depth, 3])
    ocm_8_sd = np.zeros([depth, 3])
    ocm_9_sd = np.zeros([depth, 3])
    ocm_10_sd = np.zeros([depth, 3])

    # Calculate mean of each bh
    for ocm in range(0, 3):
        ocm_1_m[:, ocm] = np.mean(ocm_ba_1[:, :, ocm], axis=0)
        ocm_2_m[:, ocm] = np.mean(ocm_ba_2[:, :, ocm], axis=0)
        ocm_3_m[:, ocm] = np.mean(ocm_ba_3[:, :, ocm], axis=0)
        ocm_4_m[:, ocm] = np.mean(ocm_ba_4[:, :, ocm], axis=0)
        ocm_5_m[:, ocm] = np.mean(ocm_ba_5[:, :, ocm], axis=0)
        ocm_6_m[:, ocm] = np.mean(ocm_ba_6[:, :, ocm], axis=0)
        ocm_7_m[:, ocm] = np.mean(ocm_ba_7[:, :, ocm], axis=0)
        ocm_8_m[:, ocm] = np.mean(ocm_ba_8[:, :, ocm], axis=0)
        ocm_9_m[:, ocm] = np.mean(ocm_ba_9[:, :, ocm], axis=0)
        ocm_10_m[:, ocm] = np.mean(ocm_ba_10[:, :, ocm], axis=0)
        ocm_1_sd[:, ocm] = np.std(ocm_ba_1[:, :, ocm], axis=0)
        ocm_2_sd[:, ocm] = np.std(ocm_ba_2[:, :, ocm], axis=0)
        ocm_3_sd[:, ocm] = np.std(ocm_ba_3[:, :, ocm], axis=0)
        ocm_4_sd[:, ocm] = np.std(ocm_ba_4[:, :, ocm], axis=0)
        ocm_5_sd[:, ocm] = np.std(ocm_ba_5[:, :, ocm], axis=0)
        ocm_6_sd[:, ocm] = np.std(ocm_ba_6[:, :, ocm], axis=0)
        ocm_7_sd[:, ocm] = np.std(ocm_ba_7[:, :, ocm], axis=0)
        ocm_8_sd[:, ocm] = np.std(ocm_ba_8[:, :, ocm], axis=0)
        ocm_9_sd[:, ocm] = np.std(ocm_ba_9[:, :, ocm], axis=0)
        ocm_10_sd[:, ocm] = np.std(ocm_ba_10[:, :, ocm], axis=0)


        # ========================Visualize==============================================
        d = np.linspace(0, ocm0_all.shape[0] - 1, ocm0_all.shape[0])
        fig = plt.figure(figsize=(18, 7))
        # This part shows raw signals
        ## Before
        # OCM0
        ocm = 0
        ax1 = fig.add_subplot(231)
        a0 = ax1.plot(d, ocm_1_m[:, ocm], 'b', linewidth=2, label="bh1")  # BH1
        plt.fill_between(range(ocm_1_m.shape[0]), ocm_1_m[:,ocm]-3*ocm_1_sd[:,ocm], ocm_1_m[:,ocm]+3*ocm_1_sd[:,ocm], alpha=.2)
        a1 = ax1.plot(d, ocm_2_m[:, ocm], 'g', linewidth=2, label="bh2")  # BH2
        plt.fill_between(range(ocm_2_m.shape[0]), ocm_2_m[:,ocm]-3*ocm_2_sd[:,ocm], ocm_2_m[:,ocm]+3*ocm_2_sd[:,ocm], alpha=.2)
        a2 = ax1.plot(d, ocm_3_m[:, ocm], 'r', linewidth=2, label="bh3")  # BH3
        plt.fill_between(range(ocm_3_m.shape[0]), ocm_3_m[:,ocm]-3*ocm_3_sd[:,ocm], ocm_3_m[:,ocm]+3*ocm_3_sd[:,ocm], alpha=.2)
        a3 = ax1.plot(d, ocm_4_m[:, ocm], 'c', linewidth=2, label="bh4")  # BH4
        plt.fill_between(range(ocm_4_m.shape[0]), ocm_4_m[:,ocm]-3*ocm_4_sd[:,ocm], ocm_4_m[:,ocm]+3*ocm_4_sd[:,ocm], alpha=.2)
        a4 = ax1.plot(d, ocm_5_m[:, ocm], 'm', linewidth=2, label="bh5")  # BH5
        plt.fill_between(range(ocm_5_m.shape[0]), ocm_5_m[:,ocm]-3*ocm_5_sd[:,ocm], ocm_5_m[:,ocm]+3*ocm_5_sd[:,ocm], alpha=.2)
        ax1.set_title('OCM0, 3SD, Before water')
        ax1.set_ylim(-2,2)
        ax1.set_xlabel('Depth')
        ax1.set_ylabel('Intensity')
        plt.legend(loc='lower right')
        plt.legend(loc='lower right')
        # OCM1
        ocm = 1
        ax2 = fig.add_subplot(232)
        a0 = ax2.plot(d, ocm_1_m[:, ocm], 'b', linewidth=2, label="bh1")  # BH1
        plt.fill_between(range(ocm_1_m.shape[0]), ocm_1_m[:,ocm]-3*ocm_1_sd[:,ocm], ocm_1_m[:,ocm]+3*ocm_1_sd[:,ocm], alpha=.2)
        a1 = ax2.plot(d, ocm_2_m[:, ocm], 'g', linewidth=2, label="bh2")  # BH2
        plt.fill_between(range(ocm_2_m.shape[0]), ocm_2_m[:,ocm]-3*ocm_2_sd[:,ocm], ocm_2_m[:,ocm]+3*ocm_2_sd[:,ocm], alpha=.2)
        a2 = ax2.plot(d, ocm_3_m[:, ocm], 'r', linewidth=2, label="bh3")  # BH3
        plt.fill_between(range(ocm_3_m.shape[0]), ocm_3_m[:,ocm]-3*ocm_3_sd[:,ocm], ocm_3_m[:,ocm]+3*ocm_3_sd[:,ocm], alpha=.2)
        a3 = ax2.plot(d, ocm_4_m[:, ocm], 'c', linewidth=2, label="bh4")  # BH4
        plt.fill_between(range(ocm_4_m.shape[0]), ocm_4_m[:,ocm]-3*ocm_4_sd[:,ocm], ocm_4_m[:,ocm]+3*ocm_4_sd[:,ocm], alpha=.2)
        a4 = ax2.plot(d, ocm_5_m[:, ocm], 'm', linewidth=2, label="bh5")  # BH5
        plt.fill_between(range(ocm_5_m.shape[0]), ocm_5_m[:,ocm]-3*ocm_5_sd[:,ocm], ocm_5_m[:,ocm]+3*ocm_5_sd[:,ocm], alpha=.2)
        ax2.set_title('OCM1, 3SD, Before water')
        ax2.set_ylim(-2,2)
        ax2.set_xlabel('Depth')
        plt.legend(loc='lower right')
        # OCM2
        ocm = 2
        ax3 = fig.add_subplot(233)
        a0 = ax3.plot(d, ocm_1_m[:, ocm], 'b', linewidth=2, label="bh1")  # BH1
        plt.fill_between(range(ocm_1_m.shape[0]), ocm_1_m[:,ocm]-3*ocm_1_sd[:,ocm], ocm_1_m[:,ocm]+3*ocm_1_sd[:,ocm], alpha=.2)
        a1 = ax3.plot(d, ocm_2_m[:, ocm], 'g', linewidth=2, label="bh2")  # BH2
        plt.fill_between(range(ocm_2_m.shape[0]), ocm_2_m[:,ocm]-3*ocm_2_sd[:,ocm], ocm_2_m[:,ocm]+3*ocm_2_sd[:,ocm], alpha=.2)
        a2 = ax3.plot(d, ocm_3_m[:, ocm], 'r', linewidth=2, label="bh3")  # BH3
        plt.fill_between(range(ocm_3_m.shape[0]), ocm_3_m[:,ocm]-3*ocm_3_sd[:,ocm], ocm_3_m[:,ocm]+3*ocm_3_sd[:,ocm], alpha=.2)
        a3 = ax3.plot(d, ocm_4_m[:, ocm], 'c', linewidth=2, label="bh4")  # BH4
        plt.fill_between(range(ocm_4_m.shape[0]), ocm_4_m[:,ocm]-3*ocm_4_sd[:,ocm], ocm_4_m[:,ocm]+3*ocm_4_sd[:,ocm], alpha=.2)
        a4 = ax3.plot(d, ocm_5_m[:, ocm], 'm', linewidth=2, label="bh5")  # BH5
        plt.fill_between(range(ocm_5_m.shape[0]), ocm_5_m[:,ocm]-3*ocm_5_sd[:,ocm], ocm_5_m[:,ocm]+3*ocm_5_sd[:,ocm], alpha=.2)
        ax3.set_title('OCM2, 3SD, Before water')
        ax3.set_ylim(-2,2)
        ax3.set_xlabel('Depth')
        plt.legend(loc='lower right')

        ## After
        # OCM0
        ocm = 0
        ax4 = fig.add_subplot(234)
        a0 = ax4.plot(d, ocm_6_m[:, ocm], 'b', linewidth=2, label="bh6")
        plt.fill_between(range(ocm_6_m.shape[0]), ocm_6_m[:,ocm]-3*ocm_6_sd[:,ocm], ocm_6_m[:,ocm]+3*ocm_6_sd[:,ocm], alpha=.2)
        a1 = ax4.plot(d, ocm_7_m[:, ocm], 'g', linewidth=2, label="bh7")
        plt.fill_between(range(ocm_7_m.shape[0]), ocm_7_m[:,ocm]-3*ocm_7_sd[:,ocm], ocm_7_m[:,ocm]+3*ocm_7_sd[:,ocm], alpha=.2)
        a2 = ax4.plot(d, ocm_8_m[:, ocm], 'r', linewidth=2, label="bh8")
        plt.fill_between(range(ocm_8_m.shape[0]), ocm_8_m[:,ocm]-3*ocm_8_sd[:,ocm], ocm_8_m[:,ocm]+3*ocm_8_sd[:,ocm], alpha=.2)
        a3 = ax4.plot(d, ocm_9_m[:, ocm], 'c', linewidth=2, label="bh9")
        plt.fill_between(range(ocm_9_m.shape[0]), ocm_9_m[:,ocm]-3*ocm_9_sd[:,ocm], ocm_9_m[:,ocm]+3*ocm_9_sd[:,ocm], alpha=.2)
        a4 = ax4.plot(d, ocm_10_m[:, ocm], 'm', linewidth=2, label="bh10")
        plt.fill_between(range(ocm_10_m.shape[0]), ocm_10_m[:,ocm]-3*ocm_10_sd[:,ocm], ocm_10_m[:,ocm]+3*ocm_10_sd[:,ocm], alpha=.2)
        ax4.set_title('OCM0, 3SD, After water')
        ax4.set_ylim(-2,2)
        ax4.set_xlabel('Depth')
        ax4.set_ylabel('Intensity')
        plt.legend(loc='lower right')
        # OCM1
        ocm = 1
        ax5 = fig.add_subplot(235)
        a0 = ax5.plot(d, ocm_6_m[:, ocm], 'b', linewidth=2, label="bh6")
        plt.fill_between(range(ocm_6_m.shape[0]), ocm_6_m[:,ocm]-3*ocm_6_sd[:,ocm], ocm_6_m[:,ocm]+3*ocm_6_sd[:,ocm], alpha=.2)
        a1 = ax5.plot(d, ocm_7_m[:, ocm], 'g', linewidth=2, label="bh7")
        plt.fill_between(range(ocm_7_m.shape[0]), ocm_7_m[:,ocm]-3*ocm_7_sd[:,ocm], ocm_7_m[:,ocm]+3*ocm_7_sd[:,ocm], alpha=.2)
        a2 = ax5.plot(d, ocm_8_m[:, ocm], 'r', linewidth=2, label="bh8")
        plt.fill_between(range(ocm_8_m.shape[0]), ocm_8_m[:,ocm]-3*ocm_8_sd[:,ocm], ocm_8_m[:,ocm]+3*ocm_8_sd[:,ocm], alpha=.2)
        a3 = ax5.plot(d, ocm_9_m[:, ocm], 'c', linewidth=2, label="bh9")
        plt.fill_between(range(ocm_9_m.shape[0]), ocm_9_m[:,ocm]-3*ocm_9_sd[:,ocm], ocm_9_m[:,ocm]+3*ocm_9_sd[:,ocm], alpha=.2)
        a4 = ax5.plot(d, ocm_10_m[:, ocm], 'm', linewidth=2, label="bh10")
        plt.fill_between(range(ocm_10_m.shape[0]), ocm_10_m[:,ocm]-3*ocm_10_sd[:,ocm], ocm_10_m[:,ocm]+3*ocm_10_sd[:,ocm], alpha=.2)
        ax5.set_title('OCM1, 3SD, After water')
        ax5.set_ylim(-2,2)
        ax5.set_xlabel('Depth')
        plt.legend(loc='lower right')
        # OCM2
        ocm = 2
        ax6 = fig.add_subplot(236)
        a0 = ax6.plot(d, ocm_6_m[:, ocm], 'b', linewidth=2, label="bh6")
        plt.fill_between(range(ocm_6_m.shape[0]), ocm_6_m[:,ocm]-3*ocm_6_sd[:,ocm], ocm_6_m[:,ocm]+3*ocm_6_sd[:,ocm], alpha=.2)
        a1 = ax6.plot(d, ocm_7_m[:, ocm], 'g', linewidth=2, label="bh7")
        plt.fill_between(range(ocm_7_m.shape[0]), ocm_7_m[:,ocm]-3*ocm_7_sd[:,ocm], ocm_7_m[:,ocm]+3*ocm_7_sd[:,ocm], alpha=.2)
        a2 = ax6.plot(d, ocm_8_m[:, ocm], 'r', linewidth=2, label="bh8")
        plt.fill_between(range(ocm_8_m.shape[0]), ocm_8_m[:,ocm]-3*ocm_8_sd[:,ocm], ocm_8_m[:,ocm]+3*ocm_8_sd[:,ocm], alpha=.2)
        a3 = ax6.plot(d, ocm_9_m[:, ocm], 'c', linewidth=2, label="bh9")
        plt.fill_between(range(ocm_9_m.shape[0]), ocm_9_m[:,ocm]-3*ocm_9_sd[:,ocm], ocm_9_m[:,ocm]+3*ocm_9_sd[:,ocm], alpha=.2)
        a4 = ax6.plot(d, ocm_10_m[:, ocm], 'm', linewidth=2, label="bh10")
        plt.fill_between(range(ocm_10_m.shape[0]), ocm_10_m[:,ocm]-3*ocm_10_sd[:,ocm], ocm_10_m[:,ocm]+3*ocm_10_sd[:,ocm], alpha=.2)
        ax6.set_title('OCM2, 3SD, After water')
        ax6.set_ylim(-2,2)
        ax6.set_xlabel('Depth')
        plt.legend(loc='lower right')

        fig.tight_layout()
        fig.show()
        f_name = 'Mean_3SD_All_bh_' + Sub_run + '_.png'
        plt.savefig(f_name)