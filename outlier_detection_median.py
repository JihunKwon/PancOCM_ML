import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import statistics
import matplotlib.animation as animation
from sklearn import metrics

plt.close('all')

out_list = []

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

#these are where the runs end in each OCM file
num_bh = 5
batch = 3 # Divide each bh into three separate groups
num_ocm = 3 # # of OCM
#rep_list = [8196, 8196]
rep_list = [8196, 8196, 8192, 8192, 6932, 6932, 3690, 3690, 3401, 3401, 3690, 3690]

#these store data for each transducer, 5 breath holds, 15 runs
t0 = np.zeros([20,5,np.size(rep_list)])
t1 = np.zeros([20,5,np.size(rep_list)])
t2 = np.zeros([20,5,np.size(rep_list)])
t3 = np.zeros([20,5,np.size(rep_list)])

#stores mean squared difference
d0 = np.zeros([5,np.size(rep_list)])
d1 = np.zeros([5,np.size(rep_list)])
d2 = np.zeros([5,np.size(rep_list)])
d3 = np.zeros([5,np.size(rep_list)])

# stores TPR and FPR3
thr_max = 100  # of threshold bin
tpr_0 = np.zeros([thr_max])
tpr_1 = np.zeros([thr_max])
tpr_2 = np.zeros([thr_max])
fpr_0 = np.zeros([thr_max])
fpr_1 = np.zeros([thr_max])
fpr_2 = np.zeros([thr_max])

for fidx in range(0,np.size(rep_list)):
    in_filename = out_list[fidx]
    ocm = np.load(in_filename)

    #crop data
    ocm = ocm[300:650,:] #Original code.

    #s=# of samples per trace
    #t2=# of total traces
    s, t = np.shape(ocm)

    # filter the data
    hptr = np.ones([s,t])  # high pass filter
    lptr = np.ones([s,t])  # low pass filter
    mag = np.ones([s,t])   # magnitude
    mag_norm = np.ones([s,t])  # Normalized
    mag_norm_medi0 = np.ones([s, num_bh*batch])  # Normalized
    mag_norm_medi1 = np.ones([s, num_bh*batch])  # Normalized
    mag_norm_medi2 = np.ones([s, num_bh*batch])  # Normalized
    diff0 = np.zeros([num_bh*batch])
    diff1 = np.zeros([num_bh*batch])
    diff2 = np.zeros([num_bh*batch])

    f1 = np.ones([5])
    max_p = 0
    ocm_mag = np.ones([s, t])  # store the magnitude of the filtered signal
    median_sub = np.ones([s, batch*num_bh])  # store the magnitude of the filtered signal

    #for sub in range(0, batch * num_bh):  # loop from 0 to 15 for each state
    for p in range(0,t):  # loop every t_sub
        # high pass then low pass filter the data
        tr1 = ocm[:,p]
        hptr[:,p] = np.convolve(tr1,[1,-1],'same')
        tr2 = hptr[:,p]
        lptr[:,p] = np.convolve(tr2,f1,'same')
        tr3 = lptr[:,p]
        # get magnitude
        mag[:,p] = np.abs(tr3)
        # normalize
        max_temp = np.max(mag[:,p])
        if max_p < max_temp:
            max_p = max_temp

        mag_norm[:,p] = np.divide(mag[:,p],np.max(mag[:,p]))

    print('mag_norm:', mag_norm.shape)  # mag_norm: (350, 166211)

    # Divide into each OCM
    b = np.linspace(0,t-1,t)
    b0 = np.mod(b,4)==0
    ocm0 = mag_norm[:,b0]
    b1 = np.mod(b,4)==1
    ocm1 = mag_norm[:,b1]
    b2 = np.mod(b,4)==2
    ocm2 = mag_norm[:,b2]

    s, c0 = np.shape(ocm0)
    s, c1 = np.shape(ocm1)
    s, c2 = np.shape(ocm2)
    print('ocm0:', ocm0.shape)
    t_sub = int(c0 / batch / num_bh)  # t_sub: # of traces in sub-region of the bh
    Thr0 = np.ones([t_sub])
    Thr1 = np.ones([t_sub])
    Thr2 = np.ones([t_sub])
    d = np.linspace(0, ocm0.shape[0] - 1, ocm0.shape[0])

    print('ocm0_value:', ocm0[10,:])

    # Divide each OCM into subregion (Each bh and each batch)
    for sub in range(0, batch*num_bh):
        for depth in range(0, s):
            mag_norm_medi0[depth, sub] = statistics.median(ocm0[depth, sub*t_sub:(sub+1)*t_sub])
            mag_norm_medi1[depth, sub] = statistics.median(ocm1[depth, sub*t_sub:(sub+1)*t_sub])
            mag_norm_medi2[depth, sub] = statistics.median(ocm2[depth, sub*t_sub:(sub+1)*t_sub])

    # if state 1, calculate baseline difference and SD. SD is used later for thresholding
    if fidx%2==0:
        base0 = mag_norm_medi0
        base1 = mag_norm_medi1
        base2 = mag_norm_medi2

        # State 1 and 2 use the same baseline. So initialized when state 1 is called,
        base_diff0 = np.ones([s, t_sub])
        base_diff1 = np.ones([s, t_sub])
        base_diff2 = np.ones([s, t_sub])
        base_diff0_sum = np.ones([t_sub])
        base_diff1_sum = np.ones([t_sub])
        base_diff2_sum = np.ones([t_sub])

        # Calculate "absolute" difference between median of baseline and each trace in baseline
        for depth in range(0, s):
            for p in range(0, t_sub):
                base_diff0[depth, p] = abs(base0[depth, 0] - ocm0[depth, p])
                base_diff1[depth, p] = abs(base1[depth, 0] - ocm1[depth, p])
                base_diff2[depth, p] = abs(base2[depth, 0] - ocm2[depth, p])

        # Get total difference of each trace
        for p in range(0, t_sub):
            base_diff0_sum[p] = np.sum(base_diff0[:, p], axis=0)
            base_diff1_sum[p] = np.sum(base_diff1[:, p], axis=0)
            base_diff2_sum[p] = np.sum(base_diff2[:, p], axis=0)

        # Calculate SD of total difference
        base_sd0 = np.std(base_diff0_sum[:])
        base_sd1 = np.std(base_diff1_sum[:])
        base_sd2 = np.std(base_diff2_sum[:])

    ##### Get area outisde of the envelope #####
    out_area0 = np.zeros([num_bh*batch])  # store the cummulative area outside of the envelope
    out_area1 = np.zeros([num_bh*batch])
    out_area2 = np.zeros([num_bh*batch])
    flag0 = np.zeros(([thr_max]))
    flag1 = np.zeros(([thr_max]))
    flag2 = np.zeros(([thr_max]))

    # Compute the absolute difference between Medi_base and Medi_test
    for n in range(0, num_bh*batch):
        for d in range(0, s):
            diff0[n] = diff0[n] + abs(mag_norm_medi0[d, n] - base0[d, 0])
            diff1[n] = diff1[n] + abs(mag_norm_medi1[d, n] - base1[d, 0])
            diff2[n] = diff2[n] + abs(mag_norm_medi2[d, n] - base2[d, 0])

    # Count the number of sub-bh above threshold
    for m in range(0, thr_max):
        for n in range(0, num_bh * batch):
            if diff0[n] > m:  # check outlier with different thresholds
                flag0[m] = flag0[m] + 1
            if diff1[n] > m:
                flag1[m] = flag1[m] + 1
            if diff2[n] > m:
                flag2[m] = flag2[m] + 1

    # Count the number of sub-bh above "SD based" threshold
    width = 3  # tolerance
    flag0_thr = 0
    flag1_thr = 0
    flag2_thr = 0
    for n in range(0, num_bh * batch):
        if diff0[n] > width * base_sd0:  # check outlier with different thresholds
            flag0_thr = flag0_thr + 1
        if diff1[n] > width * base_sd1:
            flag1_thr = flag1_thr + 1
        if diff2[n] > width * base_sd2:
            flag2_thr = flag2_thr + 1

    print('fidx:', fidx, 'base_sd0:', base_sd0, 'base_sd1:', base_sd1, 'base_sd2:', base_sd2)
    print('fidx:', fidx, 'width * base_sd0:', width * base_sd0, 'width * base_sd1:', width * base_sd1, 'width * base_sd2:', width * base_sd2)
    print('fidx:', fidx, 'flag0_thr:', flag0_thr, 'flag1_thr:', flag1_thr, 'flag2_thr:', flag2_thr)

    # =============== draw out_area ===========================================================
    fig = plt.figure()
    total_bh = np.linspace(0, num_bh*batch-1, num_bh*batch)
    ax1 = fig.add_subplot(111)
    ax1.plot(total_bh, diff0[:], 'r', linewidth=2, linestyle='solid', label="OCM0")
    ax1.plot(total_bh, diff1[:], 'g', linewidth=2, linestyle='solid', label="OCM1")
    ax1.plot(total_bh, diff2[:], 'b', linewidth=2, linestyle='solid', label="OCM2")
    ax1.set_title("Absolute difference with baseline")
    ax1.set_xlabel("BH")
    ax1.set_ylabel("Area (a.u.)")
    plt.legend(loc='lower right')

    fig.show()
    f_name = 'Out_area' + str(fidx) + '.png'
    plt.savefig(f_name)
    # =============================================================================

    cnt = fidx // 2
    for m in range(0, thr_max):  # add fidx=0,1 to cnt=0, fidx=2,3 to cnt=1, ...
        if fidx % 2 == 0:  # if fidx is an even number, it is state 1 (before water). i.e. flag is FP
            fpr_0[m] = flag0[m] / (num_bh*batch-1)  # if state 1, first sub_bh is used for training
            fpr_1[m] = flag1[m] / (num_bh*batch-1)
            fpr_2[m] = flag2[m] / (num_bh*batch-1)
            FPR0 = flag0_thr / (num_bh*batch-1)
            FPR1 = flag1_thr / (num_bh*batch-1)
            FPR2 = flag2_thr / (num_bh*batch-1)
            FP0 = flag0_thr
            FP1 = flag1_thr
            FP2 = flag2_thr
        else:  # state 2 (after water)
            tpr_0[m] = flag0[m] / (num_bh*batch)
            tpr_1[m] = flag1[m] / (num_bh*batch)
            tpr_2[m] = flag2[m] / (num_bh*batch)
            TPR0 = flag0_thr / (num_bh*batch)
            TPR1 = flag1_thr / (num_bh*batch)
            TPR2 = flag2_thr / (num_bh*batch)
            TP0 = flag0_thr
            TP1 = flag1_thr
            TP2 = flag2_thr

        if m==thr_max-1 and fidx % 2 == 1:
            fig = plt.figure()
            ax1 = fig.add_subplot(111)
            if fidx != 11:
                ax1.plot(fpr_0, tpr_0, 'r', linewidth=1.5, linestyle='solid', label="OCM0 (AUC = {:.3f})".format(metrics.auc(fpr_0, tpr_0)))
                ax1.plot(fpr_1, tpr_1, 'g', linewidth=1.5, linestyle='solid', label="OCM1 (AUC = {:.3f})".format(metrics.auc(fpr_1, tpr_1)))
                ax1.plot(fpr_2, tpr_2, 'b', linewidth=1.5, linestyle='solid', label="OCM2 (AUC = {:.3f})".format(metrics.auc(fpr_2, tpr_2)))
            else:  # signal from OCM0 for fidx=11 (s3r2) is strange and we neglect this.
                ax1.plot(fpr_1, tpr_1, 'g', linewidth=1.5, linestyle='solid', label="OCM1 (AUC = {:.3f})".format(metrics.auc(fpr_1, tpr_1)))
                ax1.plot(fpr_2, tpr_2, 'b', linewidth=1.5, linestyle='solid', label="OCM2 (AUC = {:.3f})".format(metrics.auc(fpr_2, tpr_2)))
            ax1.set_title("ROC curve")
            ax1.set_xlabel("FPR")
            ax1.set_ylabel("TPR")
            ax1.set_aspect('equal','box')
            plt.plot([0, 1], [0, 1], 'k--')
            #ax1.set_xlim(0, 1)
            #ax1.set_ylim(0, 1)
            plt.legend(loc='lower right')
            fig.show()
            f_name = 'roc_subject' + str(cnt) + '.png'
            plt.savefig(f_name)

'''
            print('fidx:', fidx)
            print('width:', width)
            print('TPR0:', TPR0, 'TPR1:', TPR1, 'TPR2:', TPR2, 'FPR0:', FPR0, 'FPR1:', FPR1, 'FPR2:', FPR2)
            print('Accuracy0:', (TP0+(num_bh*batch-1)-FP0)/((num_bh*batch)+(num_bh*batch-1)))
            print('Accuracy1:', (TP1+(num_bh*batch-1)-FP1)/((num_bh*batch)+(num_bh*batch-1)))
            print('Accuracy2:', (TP2+(num_bh*batch-1)-FP2)/((num_bh*batch)+(num_bh*batch-1)))
            print('Precision0:', (TP0/(TP0+FP0)))
            print('Precision1:', (TP1/(TP1+FP1)))
            print('Precision2:', (TP2/(TP2+FP2)))
            print('Recall0:', (TP0/(num_bh*batch)))
            print('Recall1:', (TP1/(num_bh*batch)))
            print('Recall2:', (TP2/(num_bh*batch)))
            print('F-score0:', (2*TP0/(num_bh*batch)*(TP0/(TP0+FP0)) / (TP0/(num_bh*batch)+TP0/(TP0+FP0))))
            print('F-score1:', (2*TP1/(num_bh*batch)*(TP1/(TP1+FP1)) / (TP1/(num_bh*batch)+TP1/(TP1+FP1))))
            print('F-score2:', (2*TP2/(num_bh*batch)*(TP2/(TP2+FP2)) / (TP2/(num_bh*batch)+TP2/(TP2+FP2))))

'''