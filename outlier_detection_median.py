import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import statistics
import matplotlib.animation as animation

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
num_subject = 1
num_bh = 5
batch = 3 # Divide each bh into three separate groups
num_ocm = 3 # # of OCM
rep_list = [8196, 8196]
#rep_list = [8196, 8196, 8192, 8192, 6932, 6932, 3690, 3690, 3401, 3401, 3690, 3690]

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
thr_max = 1000  # of threshold bin
TPR0 = np.zeros([thr_max - 1, np.size(rep_list) // 2])
TPR1 = np.zeros([thr_max - 1, np.size(rep_list) // 2])
TPR2 = np.zeros([thr_max - 1, np.size(rep_list) // 2])
FPR0 = np.zeros([thr_max - 1, np.size(rep_list) // 2])
FPR1 = np.zeros([thr_max - 1, np.size(rep_list) // 2])
FPR2 = np.zeros([thr_max - 1, np.size(rep_list) // 2])
tpr_0 = np.zeros([thr_max - 1])
tpr_1 = np.zeros([thr_max - 1])
tpr_2 = np.zeros([thr_max - 1])
fpr_0 = np.zeros([thr_max - 1])
fpr_1 = np.zeros([thr_max - 1])
fpr_2 = np.zeros([thr_max - 1])

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

    print('mag_norm_medi:', mag_norm_medi0[10, 0])
    print('mag_norm_medi.shape:', mag_norm_medi0.shape)

    ###### Get Threshold ######
    base_diff0 = np.ones([s, t_sub])
    base_diff1 = np.ones([s, t_sub])
    base_diff2 = np.ones([s, t_sub])
    base_sd0 = np.ones([s])
    base_sd1 = np.ones([s])
    base_sd2 = np.ones([s])

    # Calculate difference between median of baseline and each trace in baseline
    for depth in range(0, s):
        for p in range(0, t_sub):
            base_diff0[depth, p] = mag_norm_medi0[depth, 0] - ocm0[depth, p]
            base_diff1[depth, p] = mag_norm_medi1[depth, 0] - ocm1[depth, p]
            base_diff2[depth, p] = mag_norm_medi2[depth, 0] - ocm2[depth, p]

    # Calculate SD of difference
    base_sd0[:] = np.std(base_diff0[:, :], axis=1)
    base_sd1[:] = np.std(base_diff1[:, :], axis=1)
    base_sd2[:] = np.std(base_diff2[:, :], axis=1)

    # Get Threshold
    width = 3  # tolerance
    target = 2
    '''
    # ===============OCM0===========================================================
    # Plot Baseline
    fig = plt.figure()
    ims = []
    plt.plot(d, mag_norm_medi0[:, 0], 'g', linewidth=2, linestyle='dashed', label="OCM0, baseline")
    plt.plot(d, mag_norm_medi0[:, 0] + width * base_sd0[:], 'r', linewidth=2, linestyle='dashed', label="OCM0, baseline")
    plt.title("Median signal, Baseline")
    plt.xlabel("Depth")
    plt.ylabel("Magnitude (a.u.)")
    for p in range(0, 100):
        im = plt.plot(d, ocm0[:, p], 'b', linewidth=1, linestyle='solid')
        ims.append(im)
    ani = animation.ArtistAnimation(fig, ims, interval=100)
    ani.save("OCM0_baseline.gif", writer="imagemagic")

    # Plot Test set
    fig = plt.figure()
    ims = []
    plt.plot(d, mag_norm_medi0[:, 0], 'g', linewidth=2, linestyle='dashed', label="OCM0, baseline")
    plt.plot(d, mag_norm_medi0[:, 0] + width * base_sd0[:], 'r', linewidth=2, linestyle='dashed', label="OCM0, baseline")
    plt.title("Median signal, State 2")
    plt.xlabel("Depth")
    plt.ylabel("Magnitude (a.u.)")
    for p in range(t_sub*target, t_sub*target+100):
        im = plt.plot(d, ocm0[:, p], 'b', linewidth=1, linestyle='solid')
        ims.append(im)
    ani = animation.ArtistAnimation(fig, ims, interval=100)
    ani.save("OCM0_test.gif", writer="imagemagic")
    # =============================================================================

    # ===============OCM1===========================================================
    # Plot Baseline
    fig = plt.figure()
    ims = []
    plt.plot(d, mag_norm_medi1[:, 0], 'g', linewidth=2, linestyle='dashed', label="OCM0, baseline")
    plt.plot(d, mag_norm_medi1[:, 0] + width * base_sd1[:], 'r', linewidth=2, linestyle='dashed', label="OCM1, baseline")
    plt.title("Median signal, Baseline")
    plt.xlabel("Depth")
    plt.ylabel("Magnitude (a.u.)")
    for p in range(0, 100):
        im = plt.plot(d, ocm1[:, p], 'b', linewidth=1, linestyle='solid')
        ims.append(im)
    ani = animation.ArtistAnimation(fig, ims, interval=100)
    ani.save("OCM1_baseline.gif", writer="imagemagic")

    # Plot Test set
    fig = plt.figure()
    ims = []
    plt.plot(d, mag_norm_medi1[:, 0], 'g', linewidth=2, linestyle='dashed', label="OCM0, baseline")
    plt.plot(d, mag_norm_medi1[:, 0] + width * base_sd1[:], 'r', linewidth=2, linestyle='dashed', label="OCM1, baseline")
    plt.title("Median signal, State 2")
    plt.xlabel("Depth")
    plt.ylabel("Magnitude (a.u.)")
    for p in range(t_sub*target, t_sub*target+100):
        im = plt.plot(d, ocm1[:, p], 'b', linewidth=1, linestyle='solid')
        ims.append(im)
    ani = animation.ArtistAnimation(fig, ims, interval=100)
    ani.save("OCM1_test.gif", writer="imagemagic")
    # =============================================================================

    # ===============OCM2===========================================================
    # Plot Baseline
    fig = plt.figure()
    ims = []
    plt.plot(d, mag_norm_medi2[:, 0], 'g', linewidth=2, linestyle='dashed', label="OCM0, baseline")
    plt.plot(d, mag_norm_medi2[:, 0] + width * base_sd2[:], 'r', linewidth=2, linestyle='dashed', label="OCM2, baseline")
    plt.title("Median signal, Baseline")
    plt.xlabel("Depth")
    plt.ylabel("Magnitude (a.u.)")
    for p in range(0, 100):
        im = plt.plot(d, ocm2[:, p], 'b', linewidth=1, linestyle='solid')
        ims.append(im)
    ani = animation.ArtistAnimation(fig, ims, interval=100)
    ani.save("OCM2_baseline.gif", writer="imagemagic")

    # Plot Test set
    fig = plt.figure()
    ims = []
    plt.plot(d, mag_norm_medi2[:, 0], 'g', linewidth=2, linestyle='dashed', label="OCM0, baseline")
    plt.plot(d, mag_norm_medi2[:, 0] + width * base_sd2[:], 'r', linewidth=2, linestyle='dashed', label="OCM2, baseline")
    plt.title("Median signal, State 2")
    plt.xlabel("Depth")
    plt.ylabel("Magnitude (a.u.)")
    for p in range(t_sub*target, t_sub*target+100):
        im = plt.plot(d, ocm2[:, p], 'b', linewidth=1, linestyle='solid')
        ims.append(im)
    ani = animation.ArtistAnimation(fig, ims, interval=100)
    ani.save("OCM2_test.gif", writer="imagemagic")
    # =============================================================================
    '''

    ##### Get area outisde of the envelope #####
    out_area0 = np.zeros([num_bh*batch])  # store the cummulative area outside of the envelope
    out_area1 = np.zeros([num_bh*batch])
    out_area2 = np.zeros([num_bh*batch])
    flag0 = np.zeros(([thr_max]))
    flag1 = np.zeros(([thr_max]))
    flag2 = np.zeros(([thr_max]))

    # Compute the area above the envelope
    for n in range(0, num_bh*batch):
        for d in range(0, s):
            if mag_norm_medi0[d, n] > (mag_norm_medi0[d, 0] + width * base_sd0[d]):
                out_area0[n] = out_area0[n] + mag_norm_medi0[d, n] - width * base_sd0[d]
            if mag_norm_medi1[d, n] > (mag_norm_medi1[d, 0] + width * base_sd1[d]):
                out_area1[n] = out_area1[n] + mag_norm_medi1[d, n] - width * base_sd1[d]
            if mag_norm_medi2[d, n] > (mag_norm_medi2[d, 0] + width * base_sd2[d]):
                out_area2[n] = out_area2[n] + mag_norm_medi2[d, n] - width * base_sd2[d]

    # Count the number of sub-bh above threshold
    out_max = 100
    for m in range(1, thr_max):
        for n in range(0, num_bh*batch):
            if out_area0[n] >= out_max/m:  # check outlier with different thresholds
                flag0[m] = flag0[m] + 1
            if out_area1[n] >= out_max/m:
                flag1[m] = flag1[m] + 1
            if out_area2[n] >= out_max/m:
                flag2[m] = flag2[m] + 1

    ### draw out_area and threshold
    # ===============OCM0===========================================================
    fig = plt.figure()
    total_bh = np.linspace(0, num_bh*batch-1, num_bh*batch)
    ax1 = fig.add_subplot(121)
    ax1.plot(total_bh, out_area0[:], 'r', linewidth=2, linestyle='solid', label="OCM0, out_area")
    ax1.set_title("Area outisde of envelope")
    ax1.set_xlabel("BH")
    ax1.set_ylabel("Area (a.u.)")

    total_thr = np.linspace(1, thr_max-1, thr_max)
    ax2 = fig.add_subplot(122)
    ax2.plot(total_thr, flag0[:], 'r', linewidth=2, linestyle='solid', label="OCM0, count")
    ax2.set_title("# of BH higher than threshold")
    ax2.set_xlabel("Num of threshold")
    ax2.set_ylabel("Num of outlier detected")

    fig.tight_layout()
    fig.show()
    f_name = 'Out_area' + str(fidx) + '.png'
    plt.savefig(f_name)
    # =============================================================================

    cnt = fidx // 2
    for m in range(1, thr_max-1):  # add fidx=0,1 to cnt=0, fidx=2,3 to cnt=1, ...
        if fidx % 2 == 0:  # if fidx is an even number, it is state 1 (before water). i.e. flag is FP
            fpr_0[m] = flag0[m] / (num_bh*batch)
            fpr_1[m] = flag1[m] / (num_bh*batch)
            fpr_2[m] = flag2[m] / (num_bh*batch)
        else:  # state 2 (after water)
            tpr_0[m] = flag0[m] / (num_bh*batch)
            tpr_1[m] = flag1[m] / (num_bh*batch)
            tpr_2[m] = flag2[m] / (num_bh*batch)

        if m==thr_max-2 and fidx % 2 == 1:
            fig = plt.figure()
            ax1 = fig.add_subplot(111)
            ax1.plot(fpr_0, tpr_0, 'r', linewidth=1, linestyle='solid', label="OCM0")
            ax1.plot(fpr_1, tpr_1, 'g', linewidth=1, linestyle='solid', label="OCM1")
            ax1.plot(fpr_2, tpr_2, 'b', linewidth=1, linestyle='solid', label="OCM2")
            ax1.set_title("ROC curve, OCM0")
            ax1.set_xlabel("FPR")
            ax1.set_ylabel("TPR")
            ax1.set_aspect('equal','box')
            ax1.set_xlim(0, 1)
            ax1.set_ylim(0, 1)
            fig.show()
            f_name = 'roc_subject' + str(cnt) + '_new.png'
            plt.savefig(f_name)