# This code draws figures which show each process of signal processing and threshold generation

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

#Jihun Local
out_list.append("C:\\Users\\Kwon\\Documents\\Panc_OCM\\Subject_01_20180928\\run1.npy") #Before water
out_list.append("C:\\Users\\Kwon\\Documents\\Panc_OCM\\Subject_01_20180928\\run2.npy") #After water
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

rep_list = [8196, 8196, 8192, 8192, 6932, 6932, 3690, 3690, 3401, 3401, 3690, 3690]
sr_list = ['s1r1', 's1r1', 's1r2', 's1r2', 's2r1', 's2r1', 's2r2', 's2r2', 's3r1', 's3r1', 's3r2', 's3r2']

#these are where the runs end in each OCM file
num_train = 3
num_test = 10 - num_train
num_ocm = 3
num_bh = 5 # number of bh in each state
bin = 1000
scale = 50  # number divides m

tole_list = [0.000001]

for y in range(0, np.size(tole_list)):
    #for fidx in range(0, np.size(rep_list)):
    for fidx in range(2, 4):
        tole = tole_list[y]
        Sub_run_name = sr_list[fidx]
        if fidx%2==0:
            print('Status: tole'+str(tole)+'_train'+str(num_train)+'_'+Sub_run_name)
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
        '''
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
        '''
        

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
                fname = 'm012_' + str(Sub_run_name) + '_bin' + str(bin) + '_scale' + str(scale) + '_train' + str(
                    num_train) + '_pr.pkl'
                with open(fname, 'rb') as f:
                    count0, count1, count2, median0_base, median1_base, median2_base, sd0, sd1, sd2 = pickle.load(f)


        if fidx%2 == 0:
            # ===============OCM1===========================================================
            depth_my = np.linspace(2.3, 4.9, s)  # My Depth
            fig = plt.figure(figsize=(6, 5))
            ax1 = plt.subplot2grid((3, 1), (0, 0), colspan=2, rowspan=1)
            ax2 = plt.subplot2grid((3, 1), (1, 0), colspan=2, sharex=ax1)

            plt.subplots_adjust(hspace=0)
            ax1 = plt.subplot(211)
            ax2 = plt.subplot(212, sharex=ax1)
            target = 3  # target=1 is very good, but chose the data from the different bh may be better

            ax1.plot(depth_my, median1_base[:], 'k', linewidth=1, linestyle='solid', label="Baseline ($\it{M_{B}}$)")
            #ax1.plot(depth_my, ocm1_low[:, num_train*t_sub+target], 'b', linewidth=1, linestyle='dashed', label="Test set ($\it{M_{n}}$)")
            for p in range(0, t_sub):
                if p%100==0:
                    ax1.plot(depth_my, ocm1_low[:, num_train*t_sub+p], linewidth=1, linestyle='dashed', label="Test set ($\it{M_{n}}$)")
            ax1.set_title("Model Generation, State 1")
            ax1.set_ylabel("Magnitude (a.u.)")
            #ax1.legend(loc='upper right')
            #ax1.set_ylim(-700, 100000)

            #ax2.plot(depth_my, np.abs(ocm1_low[:, num_train*t_sub+target] - median1_base[:]), 'r', linewidth=1, linestyle='solid', label="Absolute Difference")
            for p in range(0, t_sub):
                if p%100==0:
                    ax2.plot(depth_my, np.abs(ocm1_low[:, num_train*t_sub+p] - median1_base[:]), linewidth=1, linestyle='solid', label="Absolute Difference")
            ax2.plot(depth_my, sd1[:], 'g', linewidth=1, linestyle='dashed', label="Threshold (m*SD, m=1)")
            ax2.plot(depth_my, 5 * sd1[:], 'g', linewidth=1.5, linestyle='dashdot', label="Threshold (m*SD, m=5)")
            ax2.set_xlabel("Depth (cm)")
            ax2.set_ylabel("Absolute Difference")
            #ax2.legend(loc='upper right')

            fig.subplots_adjust(hspace=0)
            # fig.tight_layout()

            xticklabels = ax1.get_xticklabels()
            plt.setp(xticklabels, visible=False)
            plt.subplots_adjust(left=0.15, right=0.98, top=0.92)
            # plt.show()
            f_name = 'trace_' + str(fidx // 2) + '_state1_ocm1_multi.png'
            plt.savefig(f_name)
            # =============================================================================

    
        else:
            # ===============OCM1===========================================================
            depth_my = np.linspace(2.3, 4.9, s)  # My Depth
            fig = plt.figure(figsize=(6, 5))
            ax1 = plt.subplot2grid((3, 1), (0, 0), colspan=2, rowspan=1)
            ax2 = plt.subplot2grid((3, 1), (1, 0), colspan=2, sharex=ax1)

            plt.subplots_adjust(hspace=0)
            ax1 = plt.subplot(211)
            ax2 = plt.subplot(212, sharex=ax1)
            target = 3  # target=1 is very good, but chose the data from the different bh may be better

            ax1.plot(depth_my, median1_base[:], 'k', linewidth=1, linestyle='solid',
                     label="Baseline ($\it{M_{B}}$)")
            ax1.plot(depth_my, ocm1_low[:, target], 'b', linewidth=1, linestyle='dashed',
                     label="Test set ($\it{M_{n}}$)")
            ax1.set_title("Model Generation, State 2")
            ax1.set_ylabel("Magnitude (a.u.)")
            ax1.legend(loc='upper right')
            #ax1.set_ylim(-700, 100000)

            ax2.plot(depth_my, np.abs(ocm1_low[:, target] - median1_base[:]), 'r', linewidth=1, linestyle='solid',
                     label="Absolute Difference")
            ax2.plot(depth_my, sd1[:], 'g', linewidth=1, linestyle='dashed', label="Threshold (m*SD, m=1)")
            ax2.plot(depth_my, 5 * sd1[:], 'g', linewidth=1.5, linestyle='dashdot', label="Threshold (m*SD, m=5)")
            ax2.set_xlabel("Depth (cm)")
            ax2.set_ylabel("Absolute Difference")
            ax2.legend(loc='upper right')

            fig.subplots_adjust(hspace=0)
            # fig.tight_layout()

            xticklabels = ax1.get_xticklabels()
            plt.setp(xticklabels, visible=False)
            plt.subplots_adjust(left=0.15, right=0.98, top=0.92)
            # plt.show()
            f_name = 'trace_' + str(fidx // 2) + '_state2_ocm1.png'
            plt.savefig(f_name)
            # =============================================================================
        print(str(fidx) + 'finished!')