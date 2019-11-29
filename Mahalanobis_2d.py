#Based on https://gist.github.com/tatamiya/f549aaee716fc1429f588c2240277e51

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2
import pandas as pd
#%matplotlib inline

from sklearn.covariance import EllipticEnvelope, EmpiricalCovariance, MinCovDet
from sklearn.metrics import classification_report
#from sklearn.model_selection import train_test_split
#from sklearn.datasets import make_blobs
from trace_outlier_check import outlier_remove

plt.close('all')
out_list = []

plt.rcParams['font.family'] ='sans-serif'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams["font.size"] = 11


out_list.append("C:\\Users\\Kwon\\Documents\\Panc_OCM\\Subject_01_20180928\\run1.npy") #Before water
out_list.append("C:\\Users\\Kwon\\Documents\\Panc_OCM\\Subject_01_20180928\\run2.npy") #After water
sr_list = ['s1r1','s1r1']
rep_list = [500, 500]

num_train = 3
num_test = 10 - num_train
num_ocm = 3
num_bh = 5 # number of bh in each state

# plot color setting
colors = ['#1f77b4', '#ff7f0e']

# Get max of 90% confidence interval
_, chi2_interval_max = chi2.interval(alpha=0.9, df=2)
chi2_interval_max

fidx = 0
Sub_run_name = sr_list[fidx]
in_filename = out_list[fidx]
ocm = np.load(in_filename)

#crop data
ocm = ocm[300:650, :]
s, t = np.shape(ocm)

# divide the data into each OCM and store absolute value
b = np.linspace(0, t-1, t)
b0 = np.mod(b,4) == 0
ocm0 = ocm[:, b0]
b1 = np.mod(b,4) == 1
ocm1 = ocm[:, b1]
b2 = np.mod(b,4) == 2
ocm2 = ocm[:, b2]
s, c0 = np.shape(ocm0)
print('ocm0:', ocm0.shape)

# first few traces are strange. Remove them
ocm0 = ocm0[:, c0 - num_bh*rep_list[fidx]:]
ocm1 = ocm1[:, c0 - num_bh*rep_list[fidx]:]
ocm2 = ocm2[:, c0 - num_bh*rep_list[fidx]:]

print('ocm0 new:', ocm0.shape)
s, c0_new = np.shape(ocm0)
t_sub = int(c0_new / num_bh)

# Manually remove outlier. (OCM0 contains about 0.5% of outlier)
ocm0_new = outlier_remove(Sub_run_name, c0_new, ocm0)
s, c0_new_removed = np.shape(ocm0_new)
t_sub_removed = int(c0_new_removed / num_bh)
dif = ocm0.shape[1]-ocm0_new.shape[1]

print('ocm0 new_removed:', ocm0_new.shape)
print(str(dif), 'was removed. It is', '{:.2f}'.format(dif*100/ocm0.shape[1]), '%')

ocm0_pre = ocm0_new.T
ocm1_pre = ocm1.T
ocm2_pre = ocm2.T

fidx += 1
Sub_run_name = sr_list[fidx]
in_filename = out_list[fidx]
ocm = np.load(in_filename)

#crop data
ocm = ocm[300:650, :]
s, t = np.shape(ocm)

# divide the data into each OCM and store absolute value
b = np.linspace(0, t-1, t)
b0 = np.mod(b,4) == 0
ocm0 = ocm[:, b0]
b1 = np.mod(b,4) == 1
ocm1 = ocm[:, b1]
b2 = np.mod(b,4) == 2
ocm2 = ocm[:, b2]
s, c0 = np.shape(ocm0)
print('ocm0:', ocm0.shape)

# first few traces are strange. Remove them
ocm0 = ocm0[:, c0 - num_bh*rep_list[fidx]:]
ocm1 = ocm1[:, c0 - num_bh*rep_list[fidx]:]
ocm2 = ocm2[:, c0 - num_bh*rep_list[fidx]:]

print('ocm0 new:', ocm0.shape)
s, c0_new = np.shape(ocm0)
t_sub = int(c0_new / num_bh)

# Manually remove outlier. (OCM0 contains about 0.5% of outlier)
ocm0_new = outlier_remove(Sub_run_name, c0_new, ocm0)
s, c0_new_removed = np.shape(ocm0_new)
t_sub_removed = int(c0_new_removed / num_bh)
dif = ocm0.shape[1]-ocm0_new.shape[1]

print('ocm0 new_removed:', ocm0_new.shape)
print(str(dif), 'was removed. It is', '{:.2f}'.format(dif*100/ocm0.shape[1]), '%')

ocm0_post = ocm0_new.T
ocm1_post = ocm1.T
ocm2_post = ocm2.T

# Set the length to the OCM0
ocm1_pre = ocm1_pre[:ocm0_pre.shape[0], :]
ocm2_pre = ocm2_pre[:ocm0_pre.shape[0], :]
ocm1_post = ocm1_post[:ocm0_pre.shape[0], :]
ocm2_post = ocm2_post[:ocm0_pre.shape[0], :]

depth = 10
n_samples = ocm0_pre.shape[0] + ocm0_post.shape[0]
n_inliers = ocm0_pre.shape[0]
n_outliers = ocm0_post.shape[0]

X_ocm0 = np.concatenate([ocm0_pre[:, depth], ocm0_post[:, depth]], axis=0)
X_ocm1 = np.concatenate([ocm1_pre[:, depth], ocm1_post[:, depth]], axis=0)
X_ocm2 = np.concatenate([ocm2_pre[:, depth], ocm2_post[:, depth]], axis=0)

# Create Input
X_01 = np.concatenate([X_ocm0[:, np.newaxis], X_ocm1[:, np.newaxis]], axis=1)
X_02 = np.concatenate([X_ocm0[:, np.newaxis], X_ocm2[:, np.newaxis]], axis=1)
X_12 = np.concatenate([X_ocm1[:, np.newaxis], X_ocm2[:, np.newaxis]], axis=1)

# Create answer
y = np.ones(X_ocm0.shape[0])
y[n_outliers:] = -1

X_01_in = X_01[:n_inliers, :]
X_01_out = X_01[n_inliers:, :]
X_02_in = X_02[:n_inliers, :]
X_02_out = X_02[n_inliers:, :]
X_12_in = X_12[:n_inliers, :]
X_12_out = X_12[n_inliers:, :]

# 2D plot
plt.scatter(X_01[:-n_outliers, 0], X_01[:-n_outliers, 1], c=colors[0])
plt.scatter(X_01[-n_outliers:, 0], X_01[-n_outliers:, 1], c=colors[0], marker='x')

cov_emp = EmpiricalCovariance().fit(X_01_in)

cov_emp.covariance_

xx, yy = np.meshgrid(np.linspace(-7, 7, 150), np.linspace(-7, 7, 150))

Z = cov_emp.mahalanobis(np.c_[xx.ravel(), yy.ravel()]) > chi2_interval_max # maker sure the degree of freedom for Chi2 is correct
Z = Z.reshape(xx.shape)
plt.contour(xx, yy, Z, levels=[0], linewidths=200, colors='black')

outlier_pred = cov_emp.mahalanobis(X_01) > chi2_interval_max
outlier_true = y == -1

plt.scatter(X_01[outlier_pred&outlier_true, 0], X_01[outlier_pred&outlier_true, 1], c=colors[1], marker='x', label='TP')
plt.scatter(X_01[~outlier_pred&outlier_true, 0], X_01[~outlier_pred&outlier_true, 1], c=colors[0], marker='x', label='FN')
plt.scatter(X_01[outlier_pred&~outlier_true, 0], X_01[outlier_pred&~outlier_true, 1], c=colors[1], marker='o', label='FP')
plt.scatter(X_01[~outlier_pred&~outlier_true, 0], X_01[~outlier_pred&~outlier_true, 1], c=colors[0], marker='o', label='TN')

plt.legend()

pd.DataFrame(classification_report(outlier_true, outlier_pred, output_dict=True)).T


