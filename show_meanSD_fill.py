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
    '''

    # ======================== Count num of traces outside of mean+3SD ==============================================
    num_max = ocm0_all_r1.shape[1]  # Total num of traces
    ocm0_yes = [0,0,0,0,0,0]
    ocm1_yes = [0,0,0,0,0,0]
    ocm2_yes = [0,0,0,0,0,0]
    for num in range(0, num_max):  # each traces
        # If any of the depth is out of the envelope, flag will be 1.
        flag_0 = 0
        flag_1 = 0
        flag_2 = 0
        for d in range(0, ocm0_all_r1.shape[0]):
            if flag_0 == 0:  # if no change has been detected in shallower region
                # if anatomy changed (after is smaller than mean-3SD or after is greater than mean+3SD)
                # OCM0
                if (ocm0_all_r1[d, num, 1] < (ocm0_m[d, 0] - 3*ocm0_sd[d, 0])) or ((ocm0_m[d, 0] + 3*ocm0_sd[d, 0]) < ocm0_all_r1[d, num, 1]):
                    ocm0_yes[fidx] = ocm0_yes[fidx] + 1  # Anatomy changed!
                    flag_0 = 1  # Change detected! This depth is done.
            # OCM1
            if flag_1 == 0:
                if (ocm1_all_r1[d, num, 1] < (ocm1_m[d, 0] - 3*ocm1_sd[d, 0])) or ((ocm1_m[d, 0] + 3*ocm1_sd[d, 0]) < ocm1_all_r1[d, num, 1]):
                    ocm1_yes[fidx] = ocm1_yes[fidx] + 1  # Anatomy changed!
                    flag_1 = 1  # Change detected! This depth is done.
            # OCM2
            if flag_2 == 0:
                if (ocm2_all_r1[d, num, 1] < (ocm2_m[d, 0] - 3*ocm2_sd[d, 0])) or ((ocm2_m[d, 0] + 3*ocm2_sd[d, 0]) < ocm2_all_r1[d, num, 1]):
                    ocm2_yes[fidx] = ocm2_yes[fidx] + 1  # Anatomy changed!
                    flag_2 = 1  # Change detected! This depth is done.

    print('fidx:', Sub_run)
    print('OCM0 changed:', ocm0_yes[fidx])
    print('Not changed:', num_max - ocm0_yes[fidx])
    print('TPR_0:', '{:.3f}'.format(ocm0_yes[fidx]/num_max))
    print('FNR_0:', '{:.3f}'.format((num_max - ocm0_yes[fidx])/num_max))
    print('OCM1 changed:', ocm1_yes[fidx])
    print('Not changed:', num_max - ocm1_yes[fidx])
    print('TPR_1:', '{:.3f}'.format(ocm1_yes[fidx]/num_max))
    print('FNR_1:', '{:.3f}'.format((num_max - ocm1_yes[fidx])/num_max))
    print('OCM2 changed:', ocm2_yes[fidx])
    print('Not changed:', num_max - ocm2_yes[fidx])
    print('TPR_2:', '{:.3f}'.format(ocm2_yes[fidx]/num_max))
    print('FNR_2:', '{:.3f}'.format((num_max - ocm2_yes[fidx])/num_max))


