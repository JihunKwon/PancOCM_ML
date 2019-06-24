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
    sr_name = 'ocm012_' + Sub_run + '.pkl'  # filtered
    #sr_name = 'Raw_det_ocm012_' + Sub_run + '.pkl'  # raw

    with open(sr_name, 'rb') as f:
        ocm0_all_r1, ocm1_all_r1, ocm2_all_r1 = pickle.load(f)

    print(ocm0_all_r1.shape)
    depth = np.linspace(0, ocm0_all_r1.shape[0] - 1, ocm0_all_r1.shape[0])
    num_max = ocm0_all_r1.shape[1]  # Total num of traces

    '''
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
    '''

    # Case2. Use bh 1~4 to calculate mean (bh5 is used to )
    ocm0_m_2 = np.zeros([1, ocm0_all_r1.shape[0]])  # 2rd dimension is before and after water
    ocm0_sd_2 = np.zeros([1, ocm0_all_r1.shape[0]])
    ocm1_m_2 = np.zeros([1, ocm0_all_r1.shape[0]])
    ocm1_sd_2 = np.zeros([1, ocm0_all_r1.shape[0]])
    ocm2_m_2 = np.zeros([1, ocm0_all_r1.shape[0]])
    ocm2_sd_2 = np.zeros([1, ocm0_all_r1.shape[0]])

    # Calculate mean of bh1~4 for "before" water phase
    '''
    ocm0_m_2 = np.mean(ocm0_all_r1[:, 0:int(num_max/5*4), 0], axis=1)
    ocm0_sd_2 = np.std(ocm0_all_r1[:, 0:int(num_max/5*4), 0], axis=1)
    ocm1_m_2 = np.mean(ocm1_all_r1[:, 0:int(num_max/5*4), 0], axis=1)
    ocm1_sd_2 = np.std(ocm1_all_r1[:, 0:int(num_max/5*4), 0], axis=1)
    ocm2_m_2 = np.mean(ocm2_all_r1[:, 0:int(num_max/5*4), 0], axis=1)
    ocm2_sd_2 = np.std(ocm2_all_r1[:, 0:int(num_max/5*4), 0], axis=1)
    '''
    ocm0_m_2 = np.mean(ocm0_all_r1[:, 0:num_max, 0], axis=1)
    ocm0_sd_2 = np.std(ocm0_all_r1[:, 0:num_max, 0], axis=1)
    ocm1_m_2 = np.mean(ocm1_all_r1[:, 0:num_max, 0], axis=1)
    ocm1_sd_2 = np.std(ocm1_all_r1[:, 0:num_max, 0], axis=1)
    ocm2_m_2 = np.mean(ocm2_all_r1[:, 0:num_max, 0], axis=1)
    ocm2_sd_2 = np.std(ocm2_all_r1[:, 0:num_max, 0], axis=1)

    # ========================Visualize==============================================
    '''
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
    f_name = 'Mean_SD_' + Sub_run + '.png'
    plt.savefig(f_name)
    # =============================================================================
    '''

    # ======================== Count num of traces outside of mean+3SD ==============================================
    ocm_bef_out = np.zeros([3, 6])
    ocm_aft_out = np.zeros([3, 6])
    for num in range(0, num_max):  # each traces
        # If any of the depth is out of the envelope, flag will be 1.
        flag_0_bef = 0
        flag_1_bef = 0
        flag_2_bef = 0
        flag_0_aft = 0
        flag_1_aft = 0
        flag_2_aft = 0

        # Detect out of envelope

        for d in range(0, ocm0_all_r1.shape[0]):
            ocm = 0
            phase = 0  # Before
            if flag_0_bef == 0:  # if no change has been detected in shallower region
                if (ocm0_all_r1[d, num, phase] < (ocm0_m_2[d] - 3 * ocm0_sd_2[d])) or ((ocm0_m_2[d] + 3 * ocm0_sd_2[d]) < ocm0_all_r1[d, num, phase]):
                    ocm_bef_out[ocm][fidx] = ocm_bef_out[ocm][fidx] + 1  # False Positive
                    flag_0_bef = 1  # Change detected! This trace is done.
            phase = 1 # After
            if flag_0_aft == 0:
                if (ocm0_all_r1[d, num, phase] < (ocm0_m_2[d] - 3 * ocm0_sd_2[d])) or ((ocm0_m_2[d] + 3 * ocm0_sd_2[d]) < ocm0_all_r1[d, num, phase]):
                    ocm_aft_out[ocm][fidx] = ocm_aft_out[ocm][fidx] + 1  # True Positive
                    flag_0_aft = 1  # Change detected! This trace is done.
            ocm = 1
            phase = 0  # Before
            if flag_1_bef == 0:
                if (ocm1_all_r1[d, num, phase] < (ocm1_m_2[d] - 3 * ocm1_sd_2[d])) or ((ocm1_m_2[d] + 3 * ocm1_sd_2[d]) < ocm1_all_r1[d, num, phase]):
                    ocm_bef_out[ocm][fidx] = ocm_bef_out[ocm][fidx] + 1  # False Positive
                    flag_1_bef = 1  # Change detected! This trace is done.
            phase = 1 # After
            if flag_1_aft == 0:
                if (ocm1_all_r1[d, num, phase] < (ocm1_m_2[d] - 3 * ocm1_sd_2[d])) or ((ocm1_m_2[d] + 3 * ocm1_sd_2[d]) < ocm1_all_r1[d, num, phase]):
                    ocm_aft_out[ocm][fidx] = ocm_aft_out[ocm][fidx] + 1  # True Positive
                    flag_1_aft = 1  # Change detected! This trace is done.
            ocm = 2
            phase = 0  # Before
            if flag_2_bef == 0:
                if (ocm2_all_r1[d, num, phase] < (ocm2_m_2[d] - 3 * ocm2_sd_2[d])) or ((ocm2_m_2[d] + 3 * ocm2_sd_2[d]) < ocm2_all_r1[d, num, phase]):
                    ocm_bef_out[ocm][fidx] = ocm_bef_out[ocm][fidx] + 1  # False Positive
                    flag_2_bef = 1  # Change detected! This trace is done.
            phase = 1 # After
            if flag_2_aft == 0:
                if (ocm2_all_r1[d, num, phase] < (ocm2_m_2[d] - 3 * ocm2_sd_2[d])) or ((ocm2_m_2[d] + 3 * ocm2_sd_2[d]) < ocm2_all_r1[d, num, phase]):
                    ocm_aft_out[ocm][fidx] = ocm_aft_out[ocm][fidx] + 1  # True Positive
                    flag_2_aft = 1  # Change detected! This trace is done.

    print('fidx:', Sub_run)
    TP = [0, 0, 0]
    FN = [0, 0, 0]
    TN = [0, 0, 0]
    FP = [0, 0, 0]

    for ocm in range(0,3):
        TP[ocm] = ocm_aft_out[ocm][fidx]  # After, change detected
        FN[ocm] = num_max - ocm_aft_out[ocm][fidx]  # After, change not detected
        TN[ocm] = int(num_max / 1) - ocm_bef_out[ocm][fidx]  # Before, change not detected
        FP[ocm] = ocm_bef_out[ocm][fidx]  # Before, change detected

    print(Sub_run,', OCM', str(ocm))
    #print('TP,FN,TN,FP: ', '{:.3f}'.format(TP), ' ', '{:.3f}'.format(FN), ' ', '{:.3f}'.format(TN), ' ', '{:.3f}'.format(FP))
    print('TPR,FNR,Recall: ', '{:.3f}'.format(TP[0]/num_max), ' ', '{:.3f}'.format(FN[0]/num_max),
          ' ', '{:.3f}'.format(TP[1]/num_max), ' ', '{:.3f}'.format(FN[1]/num_max),
          ' ', '{:.3f}'.format(TP[2]/num_max), ' ', '{:.3f}'.format(FN[2]/num_max))
    print('')