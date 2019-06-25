'''
This code check raw traces and remove if outlier is detected.
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import pickle
import seaborn as sns

plt.rcParams['font.family'] ='sans-serif'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

sr_list = ['s1r1', 's1r2', 's2r1', 's2r2', 's3r1', 's3r2']
#sr_list = ['s1r1']
for fidx in range(0,np.size(sr_list)):
    Sub_run = sr_list[fidx]
    #sr_name = 'ocm012_' + Sub_run + '.pkl'  # filtered
    sr_name = 'Raw_det_ocm012_' + Sub_run + '.pkl'  # raw

    with open(sr_name, 'rb') as f:
        ocm0_all_r1, ocm1_all_r1, ocm2_all_r1 = pickle.load(f)

    print(ocm0_all_r1.shape)
    depth = np.linspace(0, ocm0_all_r1.shape[0] - 1, ocm0_all_r1.shape[0])
    num_max = ocm0_all_r1.shape[1]  # Total num of traces


    # Calculate mean and sd
    ocm0_m = np.zeros([ocm0_all_r1.shape[0], 2]) # 2rd dimension is before and after water
    ocm0_sd = np.zeros([ocm0_all_r1.shape[0], 2])
    ocm1_m = np.zeros([ocm0_all_r1.shape[0], 2])
    ocm1_sd = np.zeros([ocm0_all_r1.shape[0], 2])
    ocm2_m = np.zeros([ocm0_all_r1.shape[0], 2])
    ocm2_sd = np.zeros([ocm0_all_r1.shape[0], 2])
    
    # Case1. Use all bh to calculate mean
    ocm0_m = np.mean(ocm0_all_r1[:,:,:], axis=1)
    ocm0_sd = np.std(ocm0_all_r1[:,:,:], axis=1)
    ocm1_m = np.mean(ocm1_all_r1[:,:,:], axis=1)
    ocm1_sd = np.std(ocm1_all_r1[:,:,:], axis=1)
    ocm2_m = np.mean(ocm2_all_r1[:,:,:], axis=1)
    ocm2_sd = np.std(ocm2_all_r1[:,:,:], axis=1)

    D_bef_100 = np.zeros([3, ocm0_all_r1.shape[1]])  # store traces at depth 100
    D_aft_100 = np.zeros([3, ocm0_all_r1.shape[1]])  # store traces at depth 100
    D_bef_200 = np.zeros([3, ocm0_all_r1.shape[1]])  # store traces at depth 200
    D_aft_200 = np.zeros([3, ocm0_all_r1.shape[1]])  # store traces at depth 200
    D_bef_300 = np.zeros([3, ocm0_all_r1.shape[1]])  # store traces at depth 200
    D_aft_300 = np.zeros([3, ocm0_all_r1.shape[1]])  # store traces at depth 200

    for num in range(0, ocm0_all_r1.shape[1]):
        D_bef_100[0, num] = ocm0_all_r1[100, num, 0]
        D_bef_100[1, num] = ocm1_all_r1[100, num, 0]
        D_bef_100[2, num] = ocm2_all_r1[100, num, 0]

        D_aft_100[0, num] = ocm0_all_r1[100, num, 1]
        D_aft_100[1, num] = ocm1_all_r1[100, num, 1]
        D_aft_100[2, num] = ocm2_all_r1[100, num, 1]

        D_bef_200[0, num] = ocm0_all_r1[200, num, 0]
        D_bef_200[1, num] = ocm1_all_r1[200, num, 0]
        D_bef_200[2, num] = ocm2_all_r1[200, num, 0]

        D_aft_200[0, num] = ocm0_all_r1[200, num, 1]
        D_aft_200[1, num] = ocm1_all_r1[200, num, 1]
        D_aft_200[2, num] = ocm2_all_r1[200, num, 1]

        D_bef_300[0, num] = ocm0_all_r1[300, num, 0]
        D_bef_300[1, num] = ocm1_all_r1[300, num, 0]
        D_bef_300[2, num] = ocm2_all_r1[300, num, 0]

        D_aft_300[0, num] = ocm0_all_r1[300, num, 1]
        D_aft_300[1, num] = ocm1_all_r1[300, num, 1]
        D_aft_300[2, num] = ocm2_all_r1[300, num, 1]

    '''

    # Calculate mean of "before"
    ocm0_m_2 = np.zeros([1, ocm0_all_r1.shape[0]])  # 2rd dimension is before and after water
    ocm0_sd_2 = np.zeros([1, ocm0_all_r1.shape[0]])
    ocm1_m_2 = np.zeros([1, ocm0_all_r1.shape[0]])
    ocm1_sd_2 = np.zeros([1, ocm0_all_r1.shape[0]])
    ocm2_m_2 = np.zeros([1, ocm0_all_r1.shape[0]])
    ocm2_sd_2 = np.zeros([1, ocm0_all_r1.shape[0]])

    ocm0_m_2 = np.mean(ocm0_all_r1[:, 0:num_max, 0], axis=1)
    ocm0_sd_2 = np.std(ocm0_all_r1[:, 0:num_max, 0], axis=1)
    ocm1_m_2 = np.mean(ocm1_all_r1[:, 0:num_max, 0], axis=1)
    ocm1_sd_2 = np.std(ocm1_all_r1[:, 0:num_max, 0], axis=1)
    ocm2_m_2 = np.mean(ocm2_all_r1[:, 0:num_max, 0], axis=1)
    ocm2_sd_2 = np.std(ocm2_all_r1[:, 0:num_max, 0], axis=1)
'''

    '''
    # ========================Visualize==============================================

    fig = plt.figure(figsize=(18, 6))
    # This part shows raw signals
    ## Mean+1SD
    # OCM0
    ax0 = fig.add_subplot(231)
    a0 = ax0.plot(depth, ocm0_m[:, 0], 'b', linewidth=2, label="Before")  # Before water
    plt.fill_between(range(ocm0_all_r1.shape[0]), ocm0_m[:,0]-ocm0_sd[:,0], ocm0_m[:,0]+ocm0_sd[:,0], alpha=.3)
    a1 = ax0.plot(depth, ocm0_m[:, 1], 'r', linewidth=2, label="After")  # After water
    plt.fill_between(range(ocm0_all_r1.shape[0]), ocm0_m[:,1]-ocm0_sd[:,1], ocm0_m[:,1]+ocm0_sd[:,1], alpha=.3)
    ax0.set_title('OCM0, 1SD')
    ax0.set_ylim(-2,2)
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
    ax1.set_ylim(-2,2)
    # OCM2
    ax2 = fig.add_subplot(233)
    a0 = ax2.plot(depth, ocm2_m[:, 0], 'b', linewidth=2, label="Before")  # Before water
    plt.fill_between(range(ocm2_all_r1.shape[0]), ocm2_m[:,0]-ocm2_sd[:,0], ocm2_m[:,0]+ocm2_sd[:,0], alpha=.3)
    a1 = ax2.plot(depth, ocm2_m[:, 1], 'r', linewidth=2, label="After")  # After water
    plt.fill_between(range(ocm2_all_r1.shape[0]), ocm2_m[:,1]-ocm2_sd[:,1], ocm2_m[:,1]+ocm2_sd[:,1], alpha=.3)
    ax2.set_title('OCM2, 1SD')
    ax2.set_ylim(-2,2)

    ## Mean+3SD
    # OCM0
    ax0 = fig.add_subplot(234)
    a0 = ax0.plot(depth, ocm0_m[:, 0], 'b', linewidth=2, label="Before")  # Before water
    plt.fill_between(range(ocm0_all_r1.shape[0]), ocm0_m[:,0]-3*ocm0_sd[:,0], ocm0_m[:,0]+3*ocm0_sd[:,0], alpha=.3)
    a1 = ax0.plot(depth, ocm0_m[:, 1], 'r', linewidth=2, label="After")  # After water
    plt.fill_between(range(ocm0_all_r1.shape[0]), ocm0_m[:,1]-3*ocm0_sd[:,1], ocm0_m[:,1]+3*ocm0_sd[:,1], alpha=.3)
    ax0.set_title('OCM0, 3SD')
    ax0.set_ylim(-2,2)
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
    ax1.set_ylim(-2,2)
    # OCM2
    ax2 = fig.add_subplot(236)
    a0 = ax2.plot(depth, ocm2_m[:, 0], 'b', linewidth=2, label="Before")  # Before water
    plt.fill_between(range(ocm2_all_r1.shape[0]), ocm2_m[:,0]-3*ocm2_sd[:,0], ocm2_m[:,0]+3*ocm2_sd[:,0], alpha=.3)
    a1 = ax2.plot(depth, ocm2_m[:, 1], 'r', linewidth=2, label="After")  # After water
    plt.fill_between(range(ocm2_all_r1.shape[0]), ocm2_m[:,1]-3*ocm2_sd[:,1], ocm2_m[:,1]+3*ocm2_sd[:,1], alpha=.3)
    ax2.set_title('OCM2, 3SD')
    ax2.set_ylim(-2,2)

    fig.tight_layout()
    fig.show()
    f_name = 'Mean_SD_' + Sub_run + '_test.png'
    plt.savefig(f_name)
    # =============================================================================
    '''


    # ========================Visualize Distribution ==============================================
    binwidth = 0.01
    fig2 = plt.figure(figsize=(12, 8))
    ax0 = fig2.add_subplot(331)
    ax0.hist(D_bef_100[0,:], bins=np.arange(min(D_bef_100[0,:]), max(D_bef_100[0,:]) + binwidth, binwidth), color='Blue', label="Before", alpha=0.5)
    ax0.hist(D_aft_100[0,:], bins=np.arange(min(D_aft_100[0,:]), max(D_aft_100[0,:]) + binwidth, binwidth), color='Red', label="After", alpha=0.5)
    ax0.set_title('OCM0, Depth100')
    ax0.set_xlabel('Intensity of OCM signal')
    ax0.set_ylabel('Number of traces')
    ax0.set_xlim(-1, 1)
    plt.legend(loc='best')

    ax0 = fig2.add_subplot(332)
    ax0.hist(D_bef_100[1,:], bins=np.arange(min(D_bef_100[1,:]), max(D_bef_100[1,:]) + binwidth, binwidth), color='Blue', label="Before", alpha=0.5)
    ax0.hist(D_aft_100[1,:], bins=np.arange(min(D_aft_100[1,:]), max(D_aft_100[1,:]) + binwidth, binwidth), color='Red', label="After", alpha=0.5)
    ax0.set_title('OCM1, Depth100')
    ax0.set_xlabel('Intensity of OCM signal')
    ax0.set_xlim(-1, 1)
    plt.legend(loc='best')

    ax0 = fig2.add_subplot(333)
    ax0.hist(D_bef_100[2,:], bins=np.arange(min(D_bef_100[2,:]), max(D_bef_100[2,:]) + binwidth, binwidth), color='Blue', label="Before", alpha=0.5)
    ax0.hist(D_aft_100[2,:], bins=np.arange(min(D_aft_100[2,:]), max(D_aft_100[2,:]) + binwidth, binwidth), color='Red', label="After", alpha=0.5)
    ax0.set_title('OCM2, Depth100')
    ax0.set_xlabel('Intensity of OCM signal')
    ax0.set_xlim(-1, 1)
    plt.legend(loc='best')

    # Depth 200
    ax0 = fig2.add_subplot(334)
    ax0.hist(D_bef_200[0,:], bins=np.arange(min(D_bef_200[0,:]), max(D_bef_200[0,:]) + binwidth, binwidth), color='Blue', label="Before", alpha=0.5)
    ax0.hist(D_aft_200[0,:], bins=np.arange(min(D_aft_200[0,:]), max(D_aft_200[0,:]) + binwidth, binwidth), color='Red', label="After", alpha=0.5)
    ax0.set_title('OCM0, Depth200')
    ax0.set_xlabel('Intensity of OCM signal')
    ax0.set_ylabel('Number of traces')
    ax0.set_xlim(-1, 1)
    plt.legend(loc='best')

    ax0 = fig2.add_subplot(335)
    ax0.hist(D_bef_200[1,:], bins=np.arange(min(D_bef_200[1,:]), max(D_bef_200[1,:]) + binwidth, binwidth), color='Blue', label="Before", alpha=0.5)
    ax0.hist(D_aft_200[1,:], bins=np.arange(min(D_aft_200[1,:]), max(D_aft_200[1,:]) + binwidth, binwidth), color='Red', label="After", alpha=0.5)
    ax0.set_title('OCM1, Depth200')
    ax0.set_xlabel('Intensity of OCM signal')
    ax0.set_xlim(-1, 1)
    plt.legend(loc='best')

    ax0 = fig2.add_subplot(336)
    ax0.hist(D_bef_200[2,:], bins=np.arange(min(D_bef_200[2,:]), max(D_bef_200[2,:]) + binwidth, binwidth), color='Blue', label="Before", alpha=0.5)
    ax0.hist(D_aft_200[2,:], bins=np.arange(min(D_aft_200[2,:]), max(D_aft_200[2,:]) + binwidth, binwidth), color='Red', label="After", alpha=0.5)
    ax0.set_title('OCM2, Depth200')
    ax0.set_xlabel('Intensity of OCM signal')
    ax0.set_xlim(-1, 1)
    plt.legend(loc='best')

    # Depth 300
    ax0 = fig2.add_subplot(337)
    ax0.hist(D_bef_300[0,:], bins=np.arange(min(D_bef_300[0,:]), max(D_bef_300[0,:]) + binwidth, binwidth), color='Blue', label="Before", alpha=0.5)
    ax0.hist(D_aft_300[0,:], bins=np.arange(min(D_aft_300[0,:]), max(D_aft_300[0,:]) + binwidth, binwidth), color='Red', label="After", alpha=0.5)
    ax0.set_title('OCM0, Depth300')
    ax0.set_xlabel('Intensity of OCM signal')
    ax0.set_ylabel('Number of traces')
    ax0.set_xlim(-1, 1)
    plt.legend(loc='best')

    ax0 = fig2.add_subplot(338)
    ax0.hist(D_bef_300[1,:], bins=np.arange(min(D_bef_300[1,:]), max(D_bef_300[1,:]) + binwidth, binwidth), color='Blue', label="Before", alpha=0.5)
    ax0.hist(D_aft_300[1,:], bins=np.arange(min(D_aft_300[1,:]), max(D_aft_300[1,:]) + binwidth, binwidth), color='Red', label="After", alpha=0.5)
    ax0.set_title('OCM1, Depth300')
    ax0.set_xlabel('Intensity of OCM signal')
    ax0.set_xlim(-1, 1)
    plt.legend(loc='best')

    ax0 = fig2.add_subplot(339)
    ax0.hist(D_bef_300[2,:], bins=np.arange(min(D_bef_300[2,:]), max(D_bef_300[2,:]) + binwidth, binwidth), color='Blue', label="Before", alpha=0.5)
    ax0.hist(D_aft_300[2,:], bins=np.arange(min(D_aft_300[2,:]), max(D_aft_300[2,:]) + binwidth, binwidth), color='Red', label="After", alpha=0.5)
    ax0.set_title('OCM2, Depth300')
    ax0.set_xlabel('Intensity of OCM signal')
    ax0.set_xlim(-1, 1)
    plt.legend(loc='best')

    fig2.tight_layout()
    fig2.show()
    f_name = 'Histogram_' + Sub_run + '.png'
    plt.savefig(f_name)
    # =============================================================================
