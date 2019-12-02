# This function get Chi threshold and returns the number of outliers
import numpy as np
from scipy.stats import chi2

def count_outlier(chi_idx, Chi_area, fidx, t_sub_removed, t_sub, s_new, out0_test, out1_test, out2_test, A0, A1, A2):
    _, chi2_interval_max = chi2.interval(alpha=1 - Chi_area, df=1)

    ### Count the number of traces above threshold ###
    if fidx % 2 is 0:
        bh_start = 0
        bh_end = 5
        # OCM0
        for bh in range(bh_start, bh_end):
            for p in range(0, t_sub_removed):
                flag0 = 0
                for depth in range(0, s_new):
                    # if not detected yet
                    if flag0 < 1:  # OCM0
                        # check every depth and count if it's larger than the threshold
                        if A0[depth, bh * t_sub_removed + p] > chi2_interval_max:
                            out0_test[chi_idx, int(fidx / 2), bh] = out0_test[chi_idx, int(fidx / 2), bh] + 1
                            flag0 = 1

        ##OCM1 and OCM2
        for bh in range(bh_start, bh_end):
            for p in range(0, t_sub):
                flag1 = 0
                flag2 = 0
                for depth in range(0, s_new):
                    # if not detected yet
                    if flag1 < 1:  # OCM1
                        if A1[depth, bh * t_sub + p] > chi2_interval_max:
                            out1_test[chi_idx, int(fidx / 2), bh] = out1_test[chi_idx, int(fidx / 2), bh] + 1
                            flag1 = 1
                    if flag2 < 1:  # OCM2
                        if A2[depth, bh * t_sub + p] > chi2_interval_max:
                            out2_test[chi_idx, int(fidx / 2), bh] = out2_test[chi_idx, int(fidx / 2), bh] + 1
                            flag2 = 1
    else:  # State 2
        bh_start = 5
        bh_end = 10
        # OCM0
        for bh in range(bh_start, bh_end):
            for p in range(0, t_sub_removed):
                flag0 = 0
                for depth in range(0, s_new):
                    # if not detected yet
                    if flag0 < 1:  # OCM0
                        # check every depth and count if it's larger than the threshold
                        if A0[depth, (bh - 5) * t_sub_removed + p] > chi2_interval_max:
                            out0_test[chi_idx, int(fidx / 2), bh] = out0_test[chi_idx, int(fidx / 2), bh] + 1
                            flag0 = 1

        ##OCM1 and OCM2
        for bh in range(bh_start, bh_end):
            for p in range(0, t_sub):
                flag1 = 0
                flag2 = 0
                for depth in range(0, s_new):
                    # if not detected yet
                    if flag1 < 1:  # OCM1
                        if A1[depth, (bh - 5) * t_sub + p] > chi2_interval_max:
                            out1_test[chi_idx, int(fidx / 2), bh] = out1_test[chi_idx, int(fidx / 2), bh] + 1
                            flag1 = 1
                    if flag2 < 1:  # OCM2
                        if A2[depth, (bh - 5) * t_sub + p] > chi2_interval_max:
                            out2_test[chi_idx, int(fidx / 2), bh] = out2_test[chi_idx, int(fidx / 2), bh] + 1
                            flag2 = 1

    return out0_test, out1_test, out2_test
