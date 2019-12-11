import numpy as np
import matplotlib.pyplot as plt
import csv
import statistics
import pickle
import mpl_toolkits.axes_grid1
from scipy.stats import chi2
from scipy import signal
from skimage.feature import greycomatrix, greycoprops
from skimage import util, exposure, data
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

sr_list = ['s1r1', 's1r1', 's1r2', 's1r2', 's2r1', 's2r1', 's2r2', 's2r2', 's3r1', 's3r1', 's3r2', 's3r2']
rep_list = [8196, 8196, 8192, 8192, 6932, 6932, 3690, 3690, 3401, 3401, 3690, 3690]
#rep_list = [200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200]

'''
out_list.append("C:\\Users\\Kwon\\Documents\\Panc_OCM\\Subject_01_20180928\\run1.npy") #Before water
out_list.append("C:\\Users\\Kwon\\Documents\\Panc_OCM\\Subject_01_20180928\\run2.npy") #After water
out_list.append("C:\\Users\\Kwon\\Documents\\Panc_OCM\\Subject_01_20181102\\run1.npy")
out_list.append("C:\\Users\\Kwon\\Documents\\Panc_OCM\\Subject_01_20181102\\run2.npy")
sr_list = ['s1r1', 's1r1', 's1r2', 's1r2']
rep_list = [200, 200, 200, 200]
'''

grayco_prop_list = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']

num_train = 3
num_test = 10 - num_train
num_ocm = 3
num_bh = 5 # number of bh in each state
#bin = 1000
#scale = 50  # number divides m
bin = 2000
scale = 20  # number divides m
s_new = 297  # 2.3cm to 4.5cm
interval = 20  # Interval for averaging

# Initialize output parameters of texture analysis
out_0 = np.zeros([len(rep_list)//2, len(grayco_prop_list), num_bh * 2, 1, 1])
out_1 = np.zeros([len(rep_list)//2, len(grayco_prop_list), num_bh * 2, 1, 1])
out_2 = np.zeros([len(rep_list)//2, len(grayco_prop_list), num_bh * 2, 1, 1])

for fidx in range(0, len(rep_list)):
    #fidx = 0
    #### Filtering ###
    Sub_run_name = sr_list[fidx]
    print('Status: train' + str(num_train) + '_' + Sub_run_name)
    plt.rcParams["font.size"] = 11
    in_filename = out_list[fidx]
    ocm = np.load(in_filename)

    # crop data
    ocm = ocm[300:650, :]
    s, t = np.shape(ocm)

    # variables initialization
    median0 = np.zeros([s_new, num_bh])  # median
    median1 = np.zeros([s_new, num_bh])
    median2 = np.zeros([s_new, num_bh])
    if fidx % 2 == 0:
        median0_base = np.zeros([s_new]) # median of filtered signal
        median1_base = np.zeros([s_new])
        median2_base = np.zeros([s_new])
        sd0 = np.zeros([s_new])  # sd of (median - train)
        sd1 = np.zeros([s_new])
        sd2 = np.zeros([s_new])

    # divide the data into each OCM and store absolute value
    b = np.linspace(0, t - 1, t)
    b0 = np.mod(b, 4) == 0
    ocm0 = ocm[:, b0]
    b1 = np.mod(b, 4) == 1
    ocm1 = ocm[:, b1]
    b2 = np.mod(b, 4) == 2
    ocm2 = ocm[:, b2]
    s, c0 = np.shape(ocm0)
    print('ocm0:', ocm0.shape)

    # first few traces are strange. Remove them
    ocm0 = ocm0[:, c0 - num_bh * rep_list[fidx]:]
    ocm1 = ocm1[:, c0 - num_bh * rep_list[fidx]:]
    ocm2 = ocm2[:, c0 - num_bh * rep_list[fidx]:]

    print('ocm0 new:', ocm0.shape)
    s, c0_new = np.shape(ocm0)
    t_sub = int(c0_new / num_bh)

    # Manually remove outlier. (OCM0 contains about 0.5% of outlier)
    ocm0_new = outlier_remove(Sub_run_name, c0_new, ocm0)
    s, c0_new_removed = np.shape(ocm0_new)
    t_sub_removed = int(c0_new_removed / num_bh)
    dif = ocm0.shape[1] - ocm0_new.shape[1]

    print('ocm0 new_removed:', ocm0_new.shape)
    print(str(dif), 'was removed. It is', '{:.2f}'.format(dif * 100 / ocm0.shape[1]), '%')

    ocm0_filt = np.zeros([s_new, c0_new_removed])  # filtered signal (median based filtering)
    ocm1_filt = np.zeros([s_new, c0_new])
    ocm2_filt = np.zeros([s_new, c0_new])
    ocm0_low = np.zeros([s_new, c0_new_removed])  # low pass
    ocm1_low = np.zeros([s_new, c0_new])
    ocm2_low = np.zeros([s_new, c0_new])
    # Abnormality
    A0 = np.zeros([s_new, c0_new_removed])
    A1 = np.zeros([s_new, c0_new])
    A2 = np.zeros([s_new, c0_new])
    median0_low = np.zeros([s_new, num_bh])
    median1_low = np.zeros([s_new, num_bh])
    median2_low = np.zeros([s_new, num_bh])
    f1 = np.ones([10])  # low pass kernel

    # Median-based filtering
    for bh in range(0, num_bh):
        for depth in range(0, s_new):
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
        for depth in range(0, s_new):
            # filter the signal (subtract median from each trace of corresponding bh)
            ocm0_filt[depth, p] = ocm0_new[depth, p] - median0_low[depth, bh]
        tr0 = ocm0_filt[:, p]
        ocm0_low[:, p] = np.convolve(np.sqrt(np.square(tr0)), f1, 'same')

    ## OCM1 and 2
    bh = -1
    for p in range(0, c0_new):
        if p % rep_list[fidx] == 0:
            bh = bh + 1
        for depth in range(0, s_new):
            # filter the signal (subtract median from each trace of corresponding bh)
            ocm1_filt[depth, p] = ocm1[depth, p] - median1_low[depth, bh]
            ocm2_filt[depth, p] = ocm2[depth, p] - median2_low[depth, bh]
        tr1 = ocm1_filt[:, p]
        tr2 = ocm2_filt[:, p]
        ocm1_low[:, p] = np.convolve(np.sqrt(np.square(tr1)), f1, 'same')
        ocm2_low[:, p] = np.convolve(np.sqrt(np.square(tr2)), f1, 'same')


    ## Get mean of the low pass-filtered traces
    count0 = 0
    count12 = 0
    if fidx%2 is 0:  # if state 1
        ocm0_low_pre = np.zeros([ocm0_low.shape[0], int(ocm0_low.shape[1] / interval) + 1])
        ocm1_low_pre = np.zeros([ocm1_low.shape[0], int(ocm1_low.shape[1] / interval)])
        ocm2_low_pre = np.zeros([ocm2_low.shape[0], int(ocm2_low.shape[1] / interval)])
        for p in range(0, c0_new_removed):
            if p % interval is 0:
                ocm0_low_pre[:, count0] = np.mean(ocm0_low[:, count0*interval:(count0+1)*interval], axis=1)
                count0+=1
        for p in range(0, c0_new):
            if p % interval is 0:
                ocm1_low_pre[:, count12] = np.mean(ocm1_low[:, count12*interval:(count12+1)*interval], axis=1)
                ocm2_low_pre[:, count12] = np.mean(ocm2_low[:, count12*interval:(count12+1)*interval], axis=1)
                count12+=1

    if fidx%2 is 1:  # if state 2
        ocm0_low_post = np.zeros([ocm0_low.shape[0], int(ocm0_low.shape[1]/interval)+1])
        ocm1_low_post = np.zeros([ocm1_low.shape[0], int(ocm1_low.shape[1]/interval)])
        ocm2_low_post = np.zeros([ocm2_low.shape[0], int(ocm2_low.shape[1]/interval)])
        for p in range(0, c0_new_removed):
            if p % interval is 0:
                ocm0_low_post[:, count0] = np.mean(ocm0_low[:, count0*interval:(count0+1)*interval], axis=1)
                count0+=1
        for p in range(0, c0_new):
            if p % interval is 0:
                ocm1_low_post[:, count12] = np.mean(ocm1_low[:, count12*interval:(count12+1)*interval], axis=1)
                ocm2_low_post[:, count12] = np.mean(ocm2_low[:, count12*interval:(count12+1)*interval], axis=1)
                count12+=1

        ## Followings are run when it is state 2.
        # Divide to bh
        ocm0_pre_bh = np.zeros([5, ocm0_low_pre.shape[0], int(ocm0_low_pre.shape[1]/5)])
        ocm1_pre_bh = np.zeros([5, ocm1_low_pre.shape[0], int(ocm1_low_pre.shape[1]/5)])
        ocm2_pre_bh = np.zeros([5, ocm2_low_pre.shape[0], int(ocm2_low_pre.shape[1]/5)])
        ocm0_post_bh = np.zeros([5, ocm0_low_post.shape[0], int(ocm0_low_post.shape[1]/5)])
        ocm1_post_bh = np.zeros([5, ocm1_low_post.shape[0], int(ocm1_low_post.shape[1]/5)])
        ocm2_post_bh = np.zeros([5, ocm2_low_post.shape[0], int(ocm2_low_post.shape[1]/5)])

        for bh in range(0, num_bh):
            ocm0_pre_bh[bh, :, :] = ocm0_low_pre[:, bh*int(ocm0_low_pre.shape[1]/5):(bh+1)*int(ocm0_low_pre.shape[1]/5)]
            ocm1_pre_bh[bh, :, :] = ocm1_low_pre[:, bh*int(ocm1_low_pre.shape[1]/5):(bh+1)*int(ocm1_low_pre.shape[1]/5)]
            ocm2_pre_bh[bh, :, :] = ocm2_low_pre[:, bh*int(ocm2_low_pre.shape[1]/5):(bh+1)*int(ocm2_low_pre.shape[1]/5)]
            ocm0_post_bh[bh, :, :] = ocm0_low_post[:, bh*int(ocm0_low_post.shape[1]/5):(bh+1)*int(ocm0_low_post.shape[1]/5)]
            ocm1_post_bh[bh, :, :] = ocm1_low_post[:, bh*int(ocm1_low_post.shape[1]/5):(bh+1)*int(ocm1_low_post.shape[1]/5)]
            ocm2_post_bh[bh, :, :] = ocm2_low_post[:, bh*int(ocm2_low_post.shape[1]/5):(bh+1)*int(ocm2_low_post.shape[1]/5)]

        ## Visualize traces (averaged by "interval")
        fig = plt.figure(figsize=(11, 18))
        vis = 0

        # Plot state 1
        for bh in range(0, num_bh):
            ax = fig.add_subplot(5, 2, bh * 2 + 1)
            plt.title('Bh=' + str(bh + 1))
            divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax)
            a0 = ax.imshow(ocm0_pre_bh[bh, :, :], vmin=0, vmax=np.max(ocm0_post_bh) - vis)
            cax = divider.append_axes('right', '5%', pad='3%')
            fig.colorbar(a0, cax=cax)

        # Plot state 2
        for bh in range(0, num_bh):
            ax = fig.add_subplot(5, 2, bh * 2 + 2)
            a1 = ax.imshow(ocm0_post_bh[bh, :, :], vmin=0, vmax=np.max(ocm0_post_bh) - vis)
            plt.title('Bh=' + str(bh + 6))
            divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax)
            cax = divider.append_axes('right', '5%', pad='3%')
            fig.colorbar(a0, cax=cax)
        plt.tight_layout
        f_name = 'Averaged_traces'+Sub_run_name+'.png'
        plt.savefig(f_name)

        ## Get co-occurrence matrix
        ocm0_pre_scaled = exposure.rescale_intensity(ocm0_pre_bh[:,:,:], out_range=(0, 1))
        ocm1_pre_scaled = exposure.rescale_intensity(ocm1_pre_bh[:,:,:], out_range=(0, 1))
        ocm2_pre_scaled = exposure.rescale_intensity(ocm2_pre_bh[:,:,:], out_range=(0, 1))
        ocm0_post_scaled = exposure.rescale_intensity(ocm0_post_bh[:,:,:], out_range=(0, 1))
        ocm1_post_scaled = exposure.rescale_intensity(ocm1_post_bh[:,:,:], out_range=(0, 1))
        ocm2_post_scaled = exposure.rescale_intensity(ocm2_post_bh[:,:,:], out_range=(0, 1))

        im_ocm0_pre_scaled = util.img_as_ubyte(ocm0_pre_scaled)
        im_ocm1_pre_scaled = util.img_as_ubyte(ocm1_pre_scaled)
        im_ocm2_pre_scaled = util.img_as_ubyte(ocm2_pre_scaled)
        im_ocm0_post_scaled = util.img_as_ubyte(ocm0_post_scaled)
        im_ocm1_post_scaled = util.img_as_ubyte(ocm1_post_scaled)
        im_ocm2_post_scaled = util.img_as_ubyte(ocm2_post_scaled)

        greycomatrix_pre_0 = np.zeros([num_bh, 256, 256, 1, 1])
        greycomatrix_pre_1 = np.zeros([num_bh, 256, 256, 1, 1])
        greycomatrix_pre_2 = np.zeros([num_bh, 256, 256, 1, 1])
        greycomatrix_post_0 = np.zeros([num_bh, 256, 256, 1, 1])
        greycomatrix_post_1 = np.zeros([num_bh, 256, 256, 1, 1])
        greycomatrix_post_2 = np.zeros([num_bh, 256, 256, 1, 1])

        for bh in range(0, 5):
            greycomatrix_pre_0[bh, :, :, :, :] = greycomatrix(im_ocm0_pre_scaled[bh,:,:], [5], [0], 256, normed=True)
            greycomatrix_pre_1[bh, :, :, :, :] = greycomatrix(im_ocm1_pre_scaled[bh,:,:], [5], [0], 256, normed=True)
            greycomatrix_pre_2[bh, :, :, :, :] = greycomatrix(im_ocm2_pre_scaled[bh,:,:], [5], [0], 256, normed=True)
            greycomatrix_post_0[bh, :, :, :, :] = greycomatrix(im_ocm0_post_scaled[bh,:,:], [5], [0], 256, normed=True)
            greycomatrix_post_1[bh, :, :, :, :] = greycomatrix(im_ocm1_post_scaled[bh,:,:], [5], [0], 256, normed=True)
            greycomatrix_post_2[bh, :, :, :, :] = greycomatrix(im_ocm2_post_scaled[bh,:,:], [5], [0], 256, normed=True)


        ## Plot OCM0 GLCM ##
        im_size = 256
        v_max = 0.002
        fig = plt.figure(figsize=(13, 27))
        # state 1
        for bh in range(0, num_bh):
            ax = fig.add_subplot(5, 2, bh * 2 + 1)
            plt.title('Bh=' + str(bh + 1))
            plt.xlabel('Neighbor pixel value')
            plt.ylabel('Center pixel value')
            divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax)
            a0 = ax.imshow(greycomatrix_pre_0[bh, 0:im_size, 0:im_size, 0, 0], vmin=0, vmax=v_max)
            cax = divider.append_axes('right', '5%', pad='3%')
            cbar = fig.colorbar(a0, cax=cax)
            cbar.set_label('Probability')

        # state 2
        for bh in range(0, num_bh):
            ax = fig.add_subplot(5, 2, bh * 2 + 2)
            a1 = ax.imshow(greycomatrix_post_0[bh, 0:im_size, 0:im_size, 0, 0], vmin=0, vmax=v_max)
            plt.title('Bh=' + str(bh + 6))
            plt.xlabel('Neighbor pixel value')
            plt.ylabel('Center pixel value')
            divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax)
            cax = divider.append_axes('right', '5%', pad='3%')
            cbar = fig.colorbar(a1, cax=cax)
            cbar.set_label('Probability')
        plt.tight_layout
        f_name = 'GLCM'+Sub_run_name+'_OCM0.png'
        plt.savefig(f_name)

        ## Plot OCM1 GLCM ##
        fig = plt.figure(figsize=(13, 27))
        # State 1
        for bh in range(0, num_bh):
            ax = fig.add_subplot(5, 2, bh * 2 + 1)
            plt.title('Bh=' + str(bh + 1))
            plt.xlabel('Neighbor pixel value')
            plt.ylabel('Center pixel value')
            divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax)
            a0 = ax.imshow(greycomatrix_pre_1[bh, 0:im_size, 0:im_size, 0, 0], vmin=0, vmax=v_max)
            cax = divider.append_axes('right', '5%', pad='3%')
            cbar = fig.colorbar(a0, cax=cax)
            cbar.set_label('Probability')
        # State 2
        for bh in range(0, num_bh):
            ax = fig.add_subplot(5, 2, bh * 2 + 2)
            a1 = ax.imshow(greycomatrix_post_1[bh, 0:im_size, 0:im_size, 0, 0], vmin=0, vmax=v_max)
            plt.title('Bh=' + str(bh + 6))
            plt.xlabel('Neighbor pixel value')
            plt.ylabel('Center pixel value')
            divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax)
            cax = divider.append_axes('right', '5%', pad='3%')
            cbar = fig.colorbar(a1, cax=cax)
            cbar.set_label('Probability')
        plt.tight_layout
        f_name = 'GLCM'+Sub_run_name+'_OCM1.png'
        plt.savefig(f_name)

        ## Plot OCM1 GLCM ##
        fig = plt.figure(figsize=(13, 27))
        # State 1
        for bh in range(0, num_bh):
            ax = fig.add_subplot(5, 2, bh * 2 + 1)
            plt.title('Bh=' + str(bh + 1))
            plt.xlabel('Neighbor pixel value')
            plt.ylabel('Center pixel value')
            divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax)
            a0 = ax.imshow(greycomatrix_pre_2[bh, 0:im_size, 0:im_size, 0, 0], vmin=0, vmax=v_max)
            cax = divider.append_axes('right', '5%', pad='3%')
            cbar = fig.colorbar(a0, cax=cax)
            cbar.set_label('Probability')

        # State 2
        for bh in range(0, num_bh):
            ax = fig.add_subplot(5, 2, bh * 2 + 2)
            a1 = ax.imshow(greycomatrix_post_2[bh, 0:im_size, 0:im_size, 0, 0], vmin=0, vmax=v_max)
            plt.title('Bh=' + str(bh + 6))
            plt.xlabel('Neighbor pixel value')
            plt.ylabel('Center pixel value')
            divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax)
            cax = divider.append_axes('right', '5%', pad='3%')
            cbar = fig.colorbar(a1, cax=cax)
            cbar.set_label('Probability')
        plt.tight_layout
        f_name = 'GLCM'+Sub_run_name+'_OCM2.png'
        plt.savefig(f_name)

        ## Texture Analysis ##
        print('Start Texture Analysis')
        for prop_idx in range(0, len(grayco_prop_list)):
            for bh in range(0, 10):
                if bh < 5:
                    out_0[fidx//2, prop_idx, bh, :, :] = greycoprops(greycomatrix_pre_0[bh, :, :, :, :], grayco_prop_list[prop_idx])
                    out_1[fidx//2, prop_idx, bh, :, :] = greycoprops(greycomatrix_pre_1[bh, :, :, :, :], grayco_prop_list[prop_idx])
                    out_2[fidx//2, prop_idx, bh, :, :] = greycoprops(greycomatrix_pre_2[bh, :, :, :, :], grayco_prop_list[prop_idx])
                else:
                    out_0[fidx//2, prop_idx, bh, :, :] = greycoprops(greycomatrix_post_0[bh-5, :, :, :, :], grayco_prop_list[prop_idx])
                    out_1[fidx//2, prop_idx, bh, :, :] = greycoprops(greycomatrix_post_1[bh-5, :, :, :, :], grayco_prop_list[prop_idx])
                    out_2[fidx//2, prop_idx, bh, :, :] = greycoprops(greycomatrix_post_2[bh-5, :, :, :, :], grayco_prop_list[prop_idx])


## Plot metrics vs bh
bh_idx = np.linspace(1, 10, 10)
# OCM0
fig = plt.figure(figsize=(16, 10))
for prop_idx in range(0, len(grayco_prop_list)):
    for f_idx in range(0, len(rep_list)//2):
        ax = fig.add_subplot(2, 3, prop_idx + 1)
        plt.title('OCM0, ' + grayco_prop_list[prop_idx])
        a0 = ax.plot(bh_idx, out_0[f_idx, prop_idx, :, 0, 0], label=sr_list[f_idx*2])
        plt.xlim(1, 10)
        #plt.ylim(0, np.max(out_0[f_idx, prop_idx, :, 0, 0]) * 1.05)
        plt.xlabel('Breath-holds')
        plt.ylabel(grayco_prop_list[prop_idx])
        plt.legend(loc='best')
f_name = 'Metrics_plot_OCM0.png'
plt.gca().set_ylim(bottom=0)
plt.savefig(f_name)

# OCM1
fig = plt.figure(figsize=(16, 10))
for prop_idx in range(0, len(grayco_prop_list)):
    for f_idx in range(0, len(rep_list)//2):
        ax = fig.add_subplot(2, 3, prop_idx + 1)
        plt.title('OCM1, ' + grayco_prop_list[prop_idx])
        a0 = ax.plot(bh_idx, out_1[f_idx, prop_idx, :, 0, 0], label=sr_list[f_idx*2])
        plt.xlim(1, 10)
        #plt.ylim(0, np.max(out_1[f_idx, prop_idx, :, 0, 0]) * 1.05)
        plt.xlabel('Breath-holds')
        plt.ylabel(grayco_prop_list[prop_idx])
        plt.legend(loc='best')
plt.gca().set_ylim(bottom=0)
f_name = 'Metrics_plot_OCM1.png'
plt.savefig(f_name)

# OCM2
fig = plt.figure(figsize=(16, 10))
for prop_idx in range(0, len(grayco_prop_list)):
    for f_idx in range(0, len(rep_list)//2):
        ax = fig.add_subplot(2, 3, prop_idx + 1)
        plt.title('OCM2, ' + grayco_prop_list[prop_idx])
        a0 = ax.plot(bh_idx, out_2[f_idx, prop_idx, :, 0, 0], label=sr_list[f_idx*2])
        plt.xlim(1, 10)
        #plt.ylim(0, np.max(out_2[f_idx, prop_idx, :, 0, 0]) * 1.05)
        plt.xlabel('Breath-holds')
        plt.ylabel(grayco_prop_list[prop_idx])
        plt.legend(loc='best')
plt.gca().set_ylim(bottom=0)
f_name = 'Metrics_plot_OCM2.png'
plt.savefig(f_name)