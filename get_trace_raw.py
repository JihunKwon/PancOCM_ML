'''
This code aims get signals which preserves information as much as we can.
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import pickle
import time

start = time.time()
plt.close('all')
out_list = []

'''
Note: file name run1, run2 and run3 means: before, shortly after and 10 minutes after water, respectively.
      This run name is confusing because we also use only three OCM out of four in this study. 
'''

# Jihun Local
'''
# s1r1
out_list.append("C:\\Users\\Kwon\\Documents\\Panc_OCM\\Subject_01_20180928\\run1.npy")  # Before water
out_list.append("C:\\Users\\Kwon\\Documents\\Panc_OCM\\Subject_01_20180928\\run2.npy")  # After water
out_list.append("C:\\Users\\Kwon\\Documents\\Panc_OCM\\Subject_01_20180928\\run3.npy")  # 10min After water
rep_list = [8196, 8196, 8196]
#rep_list = [820, 820, 820]

# s1r2
out_list.append("C:\\Users\\Kwon\\Documents\\Panc_OCM\\Subject_01_20181102\\run1.npy")
out_list.append("C:\\Users\\Kwon\\Documents\\Panc_OCM\\Subject_01_20181102\\run2.npy")
out_list.append("C:\\Users\\Kwon\\Documents\\Panc_OCM\\Subject_01_20181102\\run3.npy")
rep_list = [8192, 8192, 8192]
#rep_list = [1000, 1000, 1000]


# s2r1
out_list.append("C:\\Users\\Kwon\\Documents\\Panc_OCM\\Subject_02_20181102\\run1.npy")
out_list.append("C:\\Users\\Kwon\\Documents\\Panc_OCM\\Subject_02_20181102\\run2.npy")
out_list.append("C:\\Users\\Kwon\\Documents\\Panc_OCM\\Subject_02_20181102\\run3.npy")
rep_list = [6932, 6932, 6932]

# s2r2
out_list.append("C:\\Users\\Kwon\\Documents\\Panc_OCM\\Subject_02_20181220\\run1.npy")
out_list.append("C:\\Users\\Kwon\\Documents\\Panc_OCM\\Subject_02_20181220\\run2.npy")
out_list.append("C:\\Users\\Kwon\\Documents\\Panc_OCM\\Subject_02_20181220\\run3.npy")
rep_list = [3690, 3690, 3690]


# s3r1
out_list.append("C:\\Users\\Kwon\\Documents\\Panc_OCM\\Subject_03_20190228\\run1.npy")
out_list.append("C:\\Users\\Kwon\\Documents\\Panc_OCM\\Subject_03_20190228\\run2.npy")
out_list.append("C:\\Users\\Kwon\\Documents\\Panc_OCM\\Subject_03_20190228\\run3.npy")
rep_list = [3401, 3401, 3401]
'''
# s3r2
out_list.append("C:\\Users\\Kwon\\Documents\\Panc_OCM\\Subject_03_20190320\\run1.npy")
out_list.append("C:\\Users\\Kwon\\Documents\\Panc_OCM\\Subject_03_20190320\\run2.npy")
out_list.append("C:\\Users\\Kwon\\Documents\\Panc_OCM\\Subject_03_20190320\\run3.npy")
rep_list = [3690, 3690, 3690]


# these are where the runs end in each OCM file
num_subject = 1  # This number has to be the number of total run (number of subjects * number of runs)
init = 300
depth = 384
s_rate = 4  # Under sampling rate
print(np.size(rep_list))
print(rep_list[0] * 1)

# stores mean squared difference
d0 = np.zeros([5, np.size(rep_list)])
d1 = np.zeros([5, np.size(rep_list)])
d2 = np.zeros([5, np.size(rep_list)])
d3 = np.zeros([5, np.size(rep_list)])

# these store data for each transducer, 5 breath holds, 15 runs
ocm0_all = np.zeros([depth, 5 * rep_list[0], np.size(rep_list)])
ocm1_all = np.zeros([depth, 5 * rep_list[0], np.size(rep_list)])
ocm2_all = np.zeros([depth, 5 * rep_list[0], np.size(rep_list)])

# Undersampling of ocm0_all
ocm0_all_udr = np.zeros([depth, 5 * rep_list[0] // s_rate, np.size(rep_list)])
ocm1_all_udr = np.zeros([depth, 5 * rep_list[0] // s_rate, np.size(rep_list)])
ocm2_all_udr = np.zeros([depth, 5 * rep_list[0] // s_rate, np.size(rep_list)])

for fidx in range(0, np.size(rep_list)):
    # fidx = 16
    in_filename = out_list[fidx]
    ocm = np.load(in_filename)

    # crop data
    ocm = ocm[300:684, :]  # Original code.

    # s=# of samples per trace
    # t=# of total traces
    s, t = np.shape(ocm)

    # ============================1: INITIAL CODES=====================================
    # filter the data
    offset = np.ones([s, t])  # offset correction
    hptr = np.ones([s, t])  # high pass filter
    lptr = np.ones([s, t])  # low pass filter
    lptra = np.ones([s, t])
    lptr_norm = np.ones([s, t])  # Normalized
    f1 = np.ones([5])
    f2 = np.ones([10])
    max_p = 0

    for p in range(0, t):
        # high pass then low pass filter the data
        tr1 = ocm[:, p]
        offset = signal.detrend(tr1)
        '''
        hptr[:, p] = np.convolve(offset, [1, -1], 'same')
        tr2 = hptr[:, p]
        lptra[:, p] = np.convolve(tr2, f1, 'same')
        tr3 = lptra[:, p]
        # square and envelope detect
        lptr[:, p] = np.convolve(np.sqrt(np.square(tr3)), f2, 'same')
        # normalize
        max_temp = np.max(lptr[:, p])
        if max_p < max_temp:
            max_p = max_temp
        '''

        #lptr_norm[:, p] = np.divide(lptr[:, p], np.max(lptr[:, p]))
        #lptr_norm[:, p] = np.divide(ocm[:, p], np.max(ocm[:, p]))  # raw
        lptr_norm[:, p] = np.divide(offset, np.max(offset))  # raw detrend

        '''
        # ========================Visualize==============================================
        # This part shows how the signal changed after the filtering.
        depth = np.linspace(0, s - 1, s)
        fig = plt.figure(figsize=(12,8))

        ax0 = fig.add_subplot(311)
        a0 = ax0.plot(depth,ocm[:,p])
        a0off = ax0.plot(depth,offset[:])
        ax0.set_title('Raw and Offset')

        ax1 = fig.add_subplot(312)
        a1 = ax1.plot(depth,lptr[:,p])
        ax1.set_title('Low pass 5')
        ax2 = fig.add_subplot(313)
        a2 = ax2.plot(depth,lptr_norm[:,p])
        ax2.set_title('Low pass 5')

        fig.tight_layout()
        fig.show()
        plt.savefig('Filtered_wave_my.png')
        # =============================================================================
        '''

    ocm = lptr_norm

    b = np.linspace(0, t - 1, t)
    b0 = np.mod(b, 4) == 0
    ocm0 = ocm[:, b0]
    b1 = np.mod(b, 4) == 1
    ocm1 = ocm[:, b1]
    b2 = np.mod(b, 4) == 2
    ocm2 = ocm[:, b2]

    s, c0 = np.shape(ocm0)
    s, c1 = np.shape(ocm1)
    s, c2 = np.shape(ocm2)

    # collect all the data so far
    for i in range(0, 5):  # Distribute ocm signal from end to start
        ocm0_all[:, 5 * rep_list[fidx] - rep_list[fidx] * (i + 1): 5 * rep_list[fidx] - rep_list[fidx] * i, fidx] = \
            ocm0[:, ocm0.shape[1] - rep_list[fidx] * (i + 1) - 1: ocm0.shape[1] - rep_list[fidx] * i - 1]
        ocm1_all[:, 5 * rep_list[fidx] - rep_list[fidx] * (i + 1): 5 * rep_list[fidx] - rep_list[fidx] * i, fidx] = \
            ocm1[:, ocm1.shape[1] - rep_list[fidx] * (i + 1) - 1: ocm1.shape[1] - rep_list[fidx] * i - 1]
        ocm2_all[:, 5 * rep_list[fidx] - rep_list[fidx] * (i + 1): 5 * rep_list[fidx] - rep_list[fidx] * i, fidx] = \
            ocm2[:, ocm2.shape[1] - rep_list[fidx] * (i + 1) - 1: ocm2.shape[1] - rep_list[fidx] * i - 1]

    ### Under sampling here
    devide = s_rate  # dividing number
    for t in range(0, ocm0_all_udr.shape[1]):
        for r in range(s_rate):
            num = s_rate*t+r
            if num > ocm0_all.shape[1]:
                devide = devide - 1
            ocm0_all_udr[:, t, fidx] += ocm0_all[:, s_rate*t+r, fidx]
            ocm1_all_udr[:, t, fidx] += ocm1_all[:, s_rate*t+r, fidx]
            ocm2_all_udr[:, t, fidx] += ocm2_all[:, s_rate*t+r, fidx]
            if devide == 0:
                print('here!')

        ocm0_all_udr[:, t, fidx] = ocm0_all_udr[:, t, fidx] / devide
        ocm1_all_udr[:, t, fidx] = ocm1_all_udr[:, t, fidx] / devide
        ocm2_all_udr[:, t, fidx] = ocm2_all_udr[:, t, fidx] / devide

    print('fidx No.', fidx, ' has finished')

'''
# ========================Visualize==============================================
# This part shows how the signal changed after the filtering.
depth = np.linspace(0, s - 1, s)
fig = plt.figure(figsize=(12, 8))

ax0 = fig.add_subplot(311)
a1 = ax0.plot(depth, ocm0_all[:, 0, 0])
a2 = ax0.plot(depth, ocm0_all[:, 1, 0])
a3 = ax0.plot(depth, ocm0_all[:, 2, 0])
ave3 = ax0.plot(depth, ocm0_all_udr[:, 0, 0])
ax0.set_title('fidx 0')

ax1 = fig.add_subplot(312)
a1 = ax1.plot(depth, ocm0_all[:, 0, 1])
a2 = ax1.plot(depth, ocm0_all[:, 1, 1])
a3 = ax1.plot(depth, ocm0_all[:, 2, 1])
ave3 = ax1.plot(depth, ocm0_all_udr[:, 0, 1])
ax1.set_title('fidx 1')

ax2 = fig.add_subplot(313)
a1 = ax2.plot(depth, ocm0_all[:, 0, 2])
a2 = ax2.plot(depth, ocm0_all[:, 1, 2])
a3 = ax2.plot(depth, ocm0_all[:, 2, 2])
ave3 = ax2.plot(depth, ocm0_all_udr[:, 0, 2])
ax2.set_title('fidx 2')

fig.tight_layout()
fig.show()
plt.savefig('Filtered_wave_my.png')
# =============================================================================
'''

with open('Raw_det_ocm012_d384.pkl', 'wb') as f:
    pickle.dump([ocm0_all, ocm1_all, ocm2_all], f)

#fname = 'Raw_det_ocm012_undr' + str(s_rate) + '.pkl'
#with open(fname, 'wb') as f:
#    pickle.dump([ocm0_all_udr, ocm1_all_udr, ocm2_all_udr], f)

print(time.time() - start)