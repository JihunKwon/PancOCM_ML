'''
This code continuously check TPR every T seconds.
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
bh_train = 1
bh_test = 10 - bh_train

for fidx in range(0,np.size(sr_list)):
    Sub_run = sr_list[fidx]
    sr_name = 'Raw_det_ocm012_' + Sub_run + '.pkl'  # raw
    #sr_name = 'ocm012_' + Sub_run + '.pkl'  # filtered

    with open(sr_name, 'rb') as f:
        ocm0_all, ocm1_all, ocm2_all = pickle.load(f)

    print(ocm0_all.shape)
    d = np.linspace(0, ocm0_all.shape[0] - 1, ocm0_all.shape[0])
    depth = ocm0_all.shape[0]
    num_max = ocm0_all.shape[1]  # Total num of traces
    bh = int(num_max//5)

    # Remove "10min after" component
    ocm0_ba = np.zeros([depth, 2 * ocm0_all.shape[1]])  # 2nd dim is "before" and "after"
    ocm1_ba = np.zeros([depth, 2 * ocm0_all.shape[1]])  # 2nd dim is "before" and "after"
    ocm2_ba = np.zeros([depth, 2 * ocm0_all.shape[1]])  # 2nd dim is "before" and "after"

    ocm0_ba[:, 0:num_max] = ocm0_all[:,:,0]  # add "before"
    ocm1_ba[:, 0:num_max] = ocm1_all[:,:,0]
    ocm2_ba[:, 0:num_max] = ocm2_all[:,:,0]
    ocm0_ba[:, num_max:2*num_max] = ocm0_all[:,:,1]  # add "after"
    ocm1_ba[:, num_max:2*num_max] = ocm1_all[:,:,1]
    ocm2_ba[:, num_max:2*num_max] = ocm2_all[:,:,1]

    # Add to one variable
    ocm_ba = np.zeros([depth, 2*num_max, 3])
    ocm_ba[:, :, 0] = ocm0_ba[:, :]
    ocm_ba[:, :, 1] = ocm1_ba[:, :]
    ocm_ba[:, :, 2] = ocm2_ba[:, :]

    # Split data to "train" and "test". Here, only bh=1 is "train".
    ocm_train = np.zeros([depth, bh*bh_train, 3])
    ocm_test = np.zeros([depth, bh*bh_test, 3])

    ocm_train = ocm_ba[:, 0:bh*bh_train, :]
    ocm_test = ocm_ba[:, bh*bh_train:bh*10, :]
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

    ## Check performance of "train" set
    output_train = [0, 0, 0]
    for num in range(0, bh*bh_train):  # each traces
        # If any of the depth is out of the envelope, flag will be 1.
        flag = [0, 0, 0]
        # Detect out of envelope
        for d in range(0, depth):
            for ocm in range(0, 3):
                mean = ocm_train_m[d, ocm]
                sd = ocm_train_sd[d, ocm]
                if flag[ocm] == 0:  # if no change has been detected in shallower region
                    # if (before < mean-3SD) or (mean+3SD < before)
                    if ((ocm_train[num, d, ocm] < (mean - 3 * sd)) or ((mean + 3 * sd) < ocm_train[num, d, ocm])):
                        output_train[ocm] = output_train[ocm] + 1  # False Positive
                        flag[ocm] = 1  # Out of envelope!

    TP = np.zeros([9, 3])  # [bh_test, ocm]
    FN = np.zeros([9, 3])
    TN = np.zeros([9, 3])
    FP = np.zeros([9, 3])
    ocm_bh_test = np.zeros([bh, depth, 3])

    # Calculate mean of "test" (each bh separately)
    for bh_cnt in range(0, bh_test):
        output_test = [0, 0, 0]
        for ocm in range(0,3):
            ocm_bh_test[:, :, ocm] = ocm_test[bh_cnt*bh:(bh_cnt+1)*bh, :, ocm]
            ocm_test_m[:, ocm] = np.mean(ocm_test[bh_cnt*bh:(bh_cnt+1)*bh, :, ocm], axis=0)
            ocm_test_sd[:, ocm] = np.std(ocm_test[bh_cnt*bh:(bh_cnt+1)*bh, :, ocm], axis=0)

        '''
        # ========================Visualize==============================================
        d = np.linspace(0, ocm0_all.shape[0] - 1, ocm0_all.shape[0])
        fig = plt.figure(figsize=(18, 3))
        # This part shows raw signals
        ## Mean+1SD
        # OCM0
        ocm = 0
        ax0 = fig.add_subplot(131)
        a0 = ax0.plot(d, ocm_train_m[:, ocm], 'b', linewidth=2, label="Train")  # Before water
        plt.fill_between(range(ocm_train_m.shape[0]), ocm_train_m[:,ocm]-3*ocm_train_sd[:,ocm], ocm_train_m[:,ocm]+3*ocm_train_sd[:,ocm], alpha=.3)
        a1 = ax0.plot(d, ocm_test_m[:, ocm], 'r', linewidth=2, label="Test")  # After water
        plt.fill_between(range(ocm_test_m.shape[0]), ocm_test_m[:,ocm]-3*ocm_test_sd[:,ocm], ocm_test_m[:,ocm]+3*ocm_test_sd[:,ocm], alpha=.3)
        ax0.set_title('OCM0, 3SD')
        ax0.set_ylim(-2,2)
        ax0.set_xlabel('Depth')
        ax0.set_ylabel('Intensity')
        plt.legend(loc='best')
        # OCM1
        ocm = 1
        ax1 = fig.add_subplot(132)
        a0 = ax1.plot(d, ocm_train_m[:, ocm], 'b', linewidth=2, label="Train")  # Before water
        plt.fill_between(range(ocm_train_m.shape[0]), ocm_train_m[:,ocm]-3*ocm_train_sd[:,ocm], ocm_train_m[:,ocm]+3*ocm_train_sd[:,ocm], alpha=.3)
        a1 = ax1.plot(d, ocm_test_m[:, ocm], 'r', linewidth=2, label="Test")  # After water
        plt.fill_between(range(ocm_test_m.shape[0]), ocm_test_m[:,ocm]-3*ocm_test_sd[:,ocm], ocm_test_m[:,ocm]+3*ocm_test_sd[:,ocm], alpha=.3)
        ax1.set_title('OCM1, 3SD')
        ax1.set_ylim(-2,2)
        ax1.set_xlabel('Depth')
        # OCM2
        ocm = 2
        ax2 = fig.add_subplot(133)
        a0 = ax2.plot(d, ocm_train_m[:, ocm], 'b', linewidth=2, label="Train")  # Before water
        plt.fill_between(range(ocm_train_m.shape[0]), ocm_train_m[:,ocm]-3*ocm_train_sd[:,ocm], ocm_train_m[:,ocm]+3*ocm_train_sd[:,ocm], alpha=.3)
        a1 = ax2.plot(d, ocm_test_m[:, ocm], 'r', linewidth=2, label="Test")  # After water
        plt.fill_between(range(ocm_test_m.shape[0]), ocm_test_m[:,ocm]-3*ocm_test_sd[:,ocm], ocm_test_m[:,ocm]+3*ocm_test_sd[:,ocm], alpha=.3)
        ax2.set_title('OCM2, 3SD')
        ax2.set_ylim(-2,2)
        ax2.set_xlabel('Depth')
        fig.tight_layout()
        fig.show()
        plt.savefig('test.png')
        '''

            
        ## Check performance of "test" set
        for num in range(0, bh):
            # If any of the depth is out of the envelope, flag will be 1.
            flag = [0, 0, 0]
            # Detect out of envelope
            for d in range(0, depth):
                for ocm in range(0, 3):
                    mean = ocm_train_m[d, ocm]
                    sd = ocm_train_sd[d, ocm]
                    if flag[ocm] == 0:  # if no change has been detected in shallower region
                        # if (before < mean-3SD) or (mean+3SD < before)
                        if ((ocm_bh_test[num, d, ocm] < (mean - 3 * sd)) or ((mean + 3 * sd) < ocm_bh_test[num, d, ocm])):
                            output_test[ocm] = output_test[ocm] + 1
                            flag[ocm] = 1

        # Gather each bh data
        for ocm in range(0, 3):
            TP[bh_cnt][ocm] = output_test[ocm]  # Test, change detected
            FN[bh_cnt][ocm] = bh - output_test[ocm]  # Test, change not detected
            TN[bh_cnt][ocm] = bh*bh_test - output_train[ocm]  # Train, change not detected
            FP[bh_cnt][ocm] = output_train[ocm]  # Train, change detected

    print('fidx:', Sub_run)
    print('training set')
    print('TN,FP: ', '{:.3f}'.format(TN[bh_cnt][0]), ' ', '{:.3f}'.format(FP[bh_cnt][0]),
          '{:.3f}'.format(TN[bh_cnt][1]), ' ', '{:.3f}'.format(FP[bh_cnt][1]),
          '{:.3f}'.format(TN[bh_cnt][2]), ' ', '{:.3f}'.format(FP[bh_cnt][2]))
    print('TNR,FPR: ', '{:.3f}'.format(TN[bh_cnt][0] / bh), ' ', '{:.3f}'.format(FP[bh_cnt][0] / bh),
          '{:.3f}'.format(TN[bh_cnt][1] / bh), ' ', '{:.3f}'.format(FP[bh_cnt][1] / bh),
          '{:.3f}'.format(TN[bh_cnt][2] / bh), ' ', '{:.3f}'.format(FP[bh_cnt][2] / bh))
    print('')
    for bh_cnt in range(0, bh_test):
        print('bh=', bh_cnt+bh_train+1)
        print('TP,FN: ', '{:.3f}'.format(TP[bh_cnt][0]), ' ', '{:.3f}'.format(FN[bh_cnt][0])
              , ' ', '{:.3f}'.format(TP[bh_cnt][1]), ' ', '{:.3f}'.format(FN[bh_cnt][1])
              , ' ', '{:.3f}'.format(TP[bh_cnt][2]), ' ', '{:.3f}'.format(FN[bh_cnt][2]))

        print('TPR,FNR: ', '{:.3f}'.format(TP[bh_cnt][0] / bh), ' ', '{:.3f}'.format(FN[bh_cnt][0] / bh)
              , ' ', '{:.3f}'.format(TP[bh_cnt][1] / bh), ' ', '{:.3f}'.format(FN[bh_cnt][1] / bh)
              , ' ', '{:.3f}'.format(TP[bh_cnt][2] / bh), ' ', '{:.3f}'.format(FN[bh_cnt][2] / bh))
