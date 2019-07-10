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

bh_train = 3
bh_test = 10 - bh_train
batch = 1000  # subset of OCM traces used to get local average
area_level = 0.5  # What FPR you want to get for first bh of test set.
area_level_idx = [0.5, 0.7, 0.9]

for area_idx in range(0, len(area_level_idx)):
    area_level = area_level_idx[area_idx]
    print('level:', area_level)
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
        area_cumulat = np.zeros([bh * bh_train, 3])
        area_sort = np.zeros(outside_train_area.shape)

        print('###### Get Area Threshold ######')
        '''
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
        '''

        print('outside_test_area', outside_train_area.shape)
        # start searching threshold here
        flag = [0, 0, 0]
        for ocm in range(0, 3):
            area_sort[:, ocm] = sorted(outside_train_area[:, ocm])  # sort
            area_total[ocm] = sum(outside_train_area[:, ocm])
            print('area_total', ocm, ':', area_total[0])

            for a in range(0, outside_train_area.shape[0]):
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
