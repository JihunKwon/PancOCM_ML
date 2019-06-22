'''
This code check raw traces and remove if outlier is detected.
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import pickle
import seaborn as sns

plt.rcParams['font.family'] ='sans-serif'#使用するフォント
plt.rcParams['xtick.direction'] = 'in'#x軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
plt.rcParams['ytick.direction'] = 'in'#y軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')

sr_list = ['s1r1', 's1r2', 's2r1', 's2r2', 's3r1', 's3r2']
for fidx in range(0,np.size(sr_list)):
    Sub_run = sr_list[fidx]
    para_name = 'ocm012_' + Sub_run + '.pkl'

    with open(para_name, 'rb') as f:
        ocm0_all_r1, ocm1_all_r1, ocm2_all_r1 = pickle.load(f)

    print(ocm0_all_r1.shape)
    depth = np.linspace(0, ocm0_all_r1.shape[0] - 1, ocm0_all_r1.shape[0])

    # Calculate mean and sd
    ocm0_m = np.zeros([ocm0_all_r1.shape[0], 2]) # 2rd dimension is before and after water
    ocm0_sd = np.zeros([ocm0_all_r1.shape[0], 2])
    ocm1_m = np.zeros([ocm0_all_r1.shape[0], 2])
    ocm1_sd = np.zeros([ocm0_all_r1.shape[0], 2])
    ocm2_m = np.zeros([ocm0_all_r1.shape[0], 2])
    ocm2_sd = np.zeros([ocm0_all_r1.shape[0], 2])

    ocm0_m = np.mean(ocm0_all_r1[:,:,:], axis=1)
    ocm0_sd = np.std(ocm0_all_r1[:,:,:], axis=1)
    ocm1_m = np.mean(ocm1_all_r1[:,:,:], axis=1)
    ocm1_sd = np.std(ocm1_all_r1[:,:,:], axis=1)
    ocm2_m = np.mean(ocm2_all_r1[:,:,:], axis=1)
    ocm2_sd = np.std(ocm2_all_r1[:,:,:], axis=1)

    fig = plt.figure(figsize=(18, 6))
    # ========================Visualize==============================================
    # This part shows raw signals
    ## Mean+1SD
    # OCM0
    ax0 = fig.add_subplot(231)
    a0 = ax0.plot(depth, ocm0_m[:, 0], 'b', linewidth=2, label="Before")  # Before water
    plt.fill_between(range(ocm0_all_r1.shape[0]), ocm0_m[:,0]-ocm0_sd[:,0], ocm0_m[:,0]+ocm0_sd[:,0], alpha=.3)
    a1 = ax0.plot(depth, ocm0_m[:, 1], 'r', linewidth=2, label="After")  # After water
    plt.fill_between(range(ocm0_all_r1.shape[0]), ocm0_m[:,1]-ocm0_sd[:,1], ocm0_m[:,1]+ocm0_sd[:,1], alpha=.3)
    ax0.set_title('OCM0, 1SD')
    ax0.set_ylim(0,1)
    ax0.set_xlabel('Depth')
    ax0.set_ylabel('Intensity')
    plt.legend(loc='best')
    # OCM1
    ax1 = fig.add_subplot(232)
    a0 = ax1.plot(depth, ocm1_m[:, 0], 'b', linewidth=2, label="Before")  # Before water
    plt.fill_between(range(ocm1_all_r1.shape[0]), ocm1_m[:,0]-ocm1_sd[:,0], ocm1_m[:,0]+ocm1_sd[:,0], alpha=.3)
    a1 = ax1.plot(depth, ocm1_m[:, 1], 'r', linewidth=2, label="After")  # After water
    plt.fill_between(range(ocm1_all_r1.shape[0]), ocm1_m[:,1]-ocm1_sd[:,1], ocm1_m[:,1]+ocm1_sd[:,1], alpha=.3)
    ax1.set_title('OCM1, 1SD')
    ax1.set_ylim(0,1)
    # OCM2
    ax2 = fig.add_subplot(233)
    a0 = ax2.plot(depth, ocm2_m[:, 0], 'b', linewidth=2, label="Before")  # Before water
    plt.fill_between(range(ocm2_all_r1.shape[0]), ocm2_m[:,0]-ocm2_sd[:,0], ocm2_m[:,0]+ocm2_sd[:,0], alpha=.3)
    a1 = ax2.plot(depth, ocm2_m[:, 1], 'r', linewidth=2, label="After")  # After water
    plt.fill_between(range(ocm2_all_r1.shape[0]), ocm2_m[:,1]-ocm2_sd[:,1], ocm2_m[:,1]+ocm2_sd[:,1], alpha=.3)
    ax2.set_title('OCM2, 1SD')
    ax2.set_ylim(0,1)

    ## Mean+3SD
    # OCM0
    ax0 = fig.add_subplot(234)
    a0 = ax0.plot(depth, ocm0_m[:, 0], 'b', linewidth=2, label="Before")  # Before water
    plt.fill_between(range(ocm0_all_r1.shape[0]), ocm0_m[:,0]-3*ocm0_sd[:,0], ocm0_m[:,0]+3*ocm0_sd[:,0], alpha=.3)
    a1 = ax0.plot(depth, ocm0_m[:, 1], 'r', linewidth=2, label="After")  # After water
    plt.fill_between(range(ocm0_all_r1.shape[0]), ocm0_m[:,1]-3*ocm0_sd[:,1], ocm0_m[:,1]+3*ocm0_sd[:,1], alpha=.3)
    ax0.set_title('OCM0, 3SD')
    ax0.set_ylim(0,1)
    ax0.set_xlabel('Depth')
    ax0.set_ylabel('Intensity')
    plt.legend(loc='best')
    # OCM1
    ax1 = fig.add_subplot(235)
    a0 = ax1.plot(depth, ocm1_m[:, 0], 'b', linewidth=2, label="Before")  # Before water
    plt.fill_between(range(ocm1_all_r1.shape[0]), ocm1_m[:,0]-3*ocm1_sd[:,0], ocm1_m[:,0]+3*ocm1_sd[:,0], alpha=.3)
    a1 = ax1.plot(depth, ocm1_m[:, 1], 'r', linewidth=2, label="After")  # After water
    plt.fill_between(range(ocm1_all_r1.shape[0]), ocm1_m[:,1]-3*ocm1_sd[:,1], ocm1_m[:,1]+3*ocm1_sd[:,1], alpha=.3)
    ax1.set_title('OCM1, 3SD')
    ax1.set_ylim(0,1)
    # OCM2
    ax2 = fig.add_subplot(236)
    a0 = ax2.plot(depth, ocm2_m[:, 0], 'b', linewidth=2, label="Before")  # Before water
    plt.fill_between(range(ocm2_all_r1.shape[0]), ocm2_m[:,0]-3*ocm2_sd[:,0], ocm2_m[:,0]+3*ocm2_sd[:,0], alpha=.3)
    a1 = ax2.plot(depth, ocm2_m[:, 1], 'r', linewidth=2, label="After")  # After water
    plt.fill_between(range(ocm2_all_r1.shape[0]), ocm2_m[:,1]-3*ocm2_sd[:,1], ocm2_m[:,1]+3*ocm2_sd[:,1], alpha=.3)
    ax2.set_title('OCM2, 3SD')
    ax2.set_ylim(0,1)

    fig.tight_layout()
    fig.show()
    f_name = 'Mean_SD_' + Sub_run + '.png'
    plt.savefig(f_name)
    # =============================================================================