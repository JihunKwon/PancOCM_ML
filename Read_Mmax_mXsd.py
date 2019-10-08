'''
When we set FPR=0% and still we want to decrease the outlier detection rate, we extract max m0, m1, m2 and multiply it.
Scaling the threshold above the maximum m.
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import csv
import statistics
import pickle
from trace_outlier_check import outlier_remove

plt.close('all')
out_list = []

plt.rcParams['font.family'] ='sans-serif'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

out_list.append("C:\\Users\\Kwon\\Documents\\Panc_OCM\\Subject_01_20180928\\run1.npy")  #Before water
out_list.append("C:\\Users\\Kwon\\Documents\\Panc_OCM\\Subject_01_20180928\\run2.npy")  #After water
out_list.append("C:\\Users\\Kwon\\Documents\\Panc_OCM\\Subject_01_20181102\\run1.npy")
out_list.append("C:\\Users\\Kwon\\Documents\\Panc_OCM\\Subject_01_20181102\\run2.npy")
out_list.append("C:\\Users\\Kwon\\Documents\\Panc_OCM\\Subject_02_20181102\\run1.npy")
out_list.append("C:\\Users\\Kwon\\Documents\\Panc_OCM\\Subject_02_20181102\\run2.npy")
out_list.append("C:\\Users\\Kwon\\Documents\\Panc_OCM\\Subject_02_20181220\\run1.npy")
out_list.append("C:\\Users\\Kwon\\Documents\\Panc_OCM\\Subject_02_20181220\\run2.npy")
out_list.append("C:\\Users\\Kwon\\Documents\\Panc_OCM\\Subject_03_20190228\\run1.npy")
out_list.append("C:\\Users\\Kwon\\Documents\\Panc_OCM\\Subject_03_20190228\\run2.npy")
out_list.append("C:\\Users\\Kwon\\Documents\\Panc_OCM\\Subject_03_20190320\\run1.npy")
out_list.append("C:\\Users\\Kwon\\Documents\\Panc_OCM\\Subject_03_20190320\\run2.npy")

sr_list = ['s1r1', 's1r1', 's1r2', 's1r2', 's2r1', 's2r1', 's2r2', 's2r2', 's3r1', 's3r1', 's3r2', 's3r2']
rep_list = [8196, 8196, 8192, 8192, 6932, 6932, 3690, 3690, 3401, 3401, 3690, 3690]

'''
out_list.append("C:\\Users\\Kwon\\Documents\\Panc_OCM\\Subject_01_20180928\\run1.npy")  #Before water
out_list.append("C:\\Users\\Kwon\\Documents\\Panc_OCM\\Subject_01_20180928\\run2.npy")  #After water
sr_list = ['s1r1', 's1r1']
rep_list = [8192, 8192]
'''

num_train = 3
num_test = 10 - num_train
num_ocm = 3
num_bh = 5 # number of bh in each state
bin = 1000
scale = 50  # number divides m

m_list = [1.1, 1.2, 1.3, 1.4]

for y in range(0, np.size(m_list)):
    for fidx in range(0, np.size(rep_list)):
    #for fidx in range(6, 8):
        m_scale = m_list[y]
        Sub_run_name = sr_list[fidx]
        if fidx%2==0:
            print('Status: Mscale'+str(m_scale)+'_train'+str(num_train)+'_'+Sub_run_name)
        plt.rcParams["font.size"] = 11
        in_filename = out_list[fidx]
        ocm = np.load(in_filename)

        #crop data
        ocm = ocm[300:650, :]
        s, t = np.shape(ocm)

        # variables initialization
        median0 = np.zeros([s, num_bh])  # median
        median1 = np.zeros([s, num_bh])
        median2 = np.zeros([s, num_bh])
        if fidx % 2 == 0:
            thr0 = np.zeros([s])  # threshold
            thr1 = np.zeros([s])
            thr2 = np.zeros([s])
            out0_state0 = np.zeros([num_bh])  # output test result for state 0
            out1_state0 = np.zeros([num_bh])
            out2_state0 = np.zeros([num_bh])
        out0 = np.zeros([num_bh])  # output test result
        out1 = np.zeros([num_bh])
        out2 = np.zeros([num_bh])

        # divide the data into each OCM and store absolute value
        b = np.linspace(0, t-1, t)
        b0 = np.mod(b,4) == 0
        ocm0 = ocm[:, b0]
        b1 = np.mod(b,4) == 1
        ocm1 = ocm[:, b1]
        b2 = np.mod(b,4) == 2
        ocm2 = ocm[:, b2]
        s, c0 = np.shape(ocm0)
        #print('ocm0:', ocm0.shape)

        # first few traces are strange. Remove them
        ocm0 = ocm0[:, c0 - num_bh * rep_list[fidx]:]
        ocm1 = ocm1[:, c0 - num_bh * rep_list[fidx]:]
        ocm2 = ocm2[:, c0 - num_bh * rep_list[fidx]:]

        #print('ocm0 new:', ocm0.shape)
        s, c0_new = np.shape(ocm0)
        t_sub = int(c0_new / num_bh)

        # Manually remove outlier. (OCM0 contains about 0.5% of outlier)
        ocm0_new = outlier_remove(Sub_run_name, c0_new, ocm0)
        s, c0_new_removed = np.shape(ocm0_new)
        t_sub_removed = int(c0_new_removed / num_bh)
        dif = ocm0.shape[1] - ocm0_new.shape[1]

        #print('ocm0 new_removed:', ocm0_new.shape)
        #print(str(dif), 'was removed. It is', '{:.2f}'.format(dif * 100 / ocm0.shape[1]), '%')

        ocm0_filt = np.zeros([s, c0_new_removed])  # filtered signal (median based filtering)
        ocm1_filt = np.zeros([s, c0_new])
        ocm2_filt = np.zeros([s, c0_new])
        ocm0_low = np.zeros([s, c0_new_removed])  # low pass
        ocm1_low = np.zeros([s, c0_new])
        ocm2_low = np.zeros([s, c0_new])
        median0_low = np.zeros([s, num_bh])
        median1_low = np.zeros([s, num_bh])
        median2_low = np.zeros([s, num_bh])
        f1 = np.ones([10])  # low pass kernel

        # Median-based filtering
        for bh in range(0, num_bh):
            for depth in range(0, s):
                # get median for each bh
                median0[depth, bh] = statistics.median(ocm0_new[depth, bh * t_sub_removed:(bh + 1) * t_sub_removed])
                median1[depth, bh] = statistics.median(ocm1[depth, bh * t_sub:(bh + 1) * t_sub])
                median2[depth, bh] = statistics.median(ocm2[depth, bh * t_sub:(bh + 1) * t_sub])

        # Filter the median
        length = 50  # size of low pass filter
        f1_medi = np.ones([length])
        for bh in range(0, num_bh):
            tr0_medi = median0[:, bh]
            tr1_medi = median1[:, bh]
            tr2_medi = median2[:, bh]
            median0_low[:, bh] = np.convolve(tr0_medi, f1_medi, 'same') / length
            median1_low[:, bh] = np.convolve(tr1_medi, f1_medi, 'same') / length
            median2_low[:, bh] = np.convolve(tr2_medi, f1_medi, 'same') / length

        # filtering all traces with median trace
        # The size of ocm0 is different with ocm1 and ocm2. Has to be filtered separately.
        ## OCM0
        bh = -1
        for p in range(0, c0_new_removed):
            # have to consider the number of traces removed
            if p % rep_list[fidx] - dif // num_bh == 0:
                bh = bh + 1
            for depth in range(0, s):
                # filter the signal (subtract median from each trace of corresponding bh)
                ocm0_filt[depth, p] = ocm0_new[depth, p] - median0_low[depth, bh]
            tr0 = ocm0_filt[:, p]
            ocm0_low[:, p] = np.convolve(np.sqrt(np.square(tr0)), f1, 'same')

        ## OCM1 and 2
        bh = -1
        for p in range(0, c0_new):
            if p % rep_list[fidx] == 0:
                bh = bh + 1
            for depth in range(0, s):
                # filter the signal (subtract median from each trace of corresponding bh)
                ocm1_filt[depth, p] = ocm1[depth, p] - median1_low[depth, bh]
                ocm2_filt[depth, p] = ocm2[depth, p] - median2_low[depth, bh]
            tr1 = ocm1_filt[:, p]
            tr2 = ocm2_filt[:, p]
            ocm1_low[:, p] = np.convolve(np.sqrt(np.square(tr1)), f1, 'same')
            ocm2_low[:, p] = np.convolve(np.sqrt(np.square(tr2)), f1, 'same')


        #### Threshold generation ####
        # if state 1
        if fidx % 2 == 0:
            #### Get parameter m ####
            # m and OoE (out of envelop) distribution
            fname = 'm012_' + str(Sub_run_name) + '_bin' + str(bin) + '_scale' + str(scale) + '_train' + str(num_train) + '_pr.pkl'
            with open(fname, 'rb') as f:
                count0, count1, count2, median0_base, median1_base, median2_base, sd0, sd1, sd2 = pickle.load(f)

            # search the maximum m in the distribution
            flag0 = 0
            flag1 = 0
            flag2 = 0
            for x in range(bin-1, -1, -1):
                if flag0 == 0:
                    if count0[x] > 0:
                        m0_max = x / scale
                        flag0 = 1
                if flag1 == 0:
                    if count1[x] > 0:
                        m1_max = x / scale
                        flag1 = 1
                if flag2 == 0:
                    if count2[x] > 0:
                        m2_max = x / scale
                        flag2 = 1

            m0 = m0_max * m_scale
            m1 = m1_max * m_scale
            m2 = m2_max * m_scale

            print('m0_max:', m0_max, 'm1_max:', m1_max, 'm2_max:', m2_max)
            print('m0:', m0, 'm1:', m1, 'm2:', m2)

        #### Performance Evaluation with remaining data####
        # set threshold
        thr0[:] = m0 * sd0[:]
        thr1[:] = m1 * sd1[:]
        thr2[:] = m2 * sd2[:]

        # Check test data
        ## OCM0
        for bh in range(0, num_bh):
            for p in range(0, t_sub_removed):
                flag0 = 0
                for depth in range(0, s):
                    # if not detected yet
                    if flag0 < 1:  # OCM0
                        # check every depth and count if it's larger than the threshold
                        if np.abs(ocm0_low[depth, bh * t_sub_removed + p] - median0_base[depth]) > thr0[depth]:
                            out0[bh] = out0[bh] + 1
                            flag0 = 1


        ## OCM1 and OCM2
        for bh in range(0, num_bh):
            for p in range(0, t_sub):
                flag1 = 0
                flag2 = 0
                for depth in range(0, s):
                    # if not detected yet
                    if flag1 < 1:  # OCM1
                        if np.abs(ocm1_low[depth, bh * t_sub + p] - median1_base[depth]) > thr1[depth]:
                            out1[bh] = out1[bh] + 1
                            flag1 = 1
                    if flag2 < 1:  # OCM2
                        if np.abs(ocm2_low[depth, bh * t_sub + p] - median2_base[depth]) > thr2[depth]:
                            out2[bh] = out2[bh] + 1
                            flag2 = 1


        # if state 1, bh=1 (baseline) has to be subtracted
        total0 = 100 / (rep_list[fidx] - dif//num_bh)
        total12 = 100 / rep_list[fidx]

        if fidx % 2 == 0:
            # store out from state 0
            out0_state0 = out0
            out1_state0 = out1
            out2_state0 = out2
            '''
            print('OCM0 number:', out0[0], out0[1], 'bh3:', out0[2], 'bh4:',
                  out0[3], 'bh5:', out0[4])
            print('OCM1 number:', out1[0], out1[1], 'bh3:', out1[2], 'bh4:',
                  out1[3], 'bh5:', out1[4])
            print('OCM2 number:', out2[0], out2[1], 'bh3:', out2[2], 'bh4:',
                  out2[3], 'bh5:', out2[4])
            '''
        else:
            '''
            print('OCM0 number:', out0[0], out0[1], 'bh8:', out0[2],
                  out0[3], out0[4])
            print('OCM1 number:', out1[0], out1[1], 'bh8:', out1[2],
                  out1[3], out1[4])
            print('OCM2 number:', out2[0], out2[1], 'bh8:', out2[2],
                  out2[3], out2[4])
            '''
            print('bh1, bh2, bh3, bh4, bh5, bh6, bh7, bh8, bh9, bh10')
            print('OCM0 rate:',
                  '{:.3f}'.format(out0_state0[0] * total0), '{:.3f}'.format(out0_state0[1] * total0),
                  '{:.3f}'.format(out0_state0[2] * total0), '{:.3f}'.format(out0_state0[3] * total0),
                  '{:.3f}'.format(out0_state0[4] * total0),
                  '{:.3f}'.format(out0[0] * total0), '{:.3f}'.format(out0[1] * total0),
                  '{:.3f}'.format(out0[2] * total0), '{:.3f}'.format(out0[3] * total0),
                  '{:.3f}'.format(out0[4] * total0))

            print('OCM1 rate:',
                  '{:.3f}'.format(out1_state0[0] * total12), '{:.3f}'.format(out1_state0[1] * total12),
                  '{:.3f}'.format(out1_state0[2] * total12), '{:.3f}'.format(out1_state0[3] * total12),
                  '{:.3f}'.format(out1_state0[4] * total12),
                  '{:.3f}'.format(out1[0] * total12), '{:.3f}'.format(out1[1] * total12),
                  '{:.3f}'.format(out1[2] * total12), '{:.3f}'.format(out1[3] * total12),
                  '{:.3f}'.format(out1[4] * total12))

            print('OCM2 rate:',
                  '{:.3f}'.format(out2_state0[0] * total12), '{:.3f}'.format(out2_state0[1] * total12),
                  '{:.3f}'.format(out2_state0[2] * total12), '{:.3f}'.format(out2_state0[3] * total12),
                  '{:.3f}'.format(out2_state0[4] * total12),
                  '{:.3f}'.format(out2[0] * total12), '{:.3f}'.format(out2[1] * total12),
                  '{:.3f}'.format(out2[2] * total12), '{:.3f}'.format(out2[3] * total12),
                  '{:.3f}'.format(out2[4] * total12))