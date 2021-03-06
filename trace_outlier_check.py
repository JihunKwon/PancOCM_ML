# This function get subject and run name and conduct outlier remove
import numpy as np

def outlier_remove(Sub_run_name, c0, ocm0):  # input subject and run name
    count = 0
    ocm0_new = np.zeros(np.shape(ocm0))
    if Sub_run_name is 's1r1':
        for p in range(0, c0):
            if ocm0[-1, p] > -200:
                ocm0_new[:, count] = ocm0[:, p]
                count = count + 1

    elif Sub_run_name is 's1r2':
        for p in range(0, c0):
            if ocm0[0, p] < 3000 and ocm0[-1, p] < 3000:
                    if ocm0[54, p] < 3000:
                        ocm0_new[:, count] = ocm0[:, p]
                        count = count + 1

    elif Sub_run_name is 's2r1':
        for p in range(0, c0):
            if ocm0[0, p] < 6000 and ocm0[170, p] < 1500 and ocm0[200, p] < 4000 and ocm0[225, p] < 2800 and ocm0[-1, p] < 6000:
            #if ocm0[0, p] < 4000 and ocm0[-1, p] < 4500:
                ocm0_new[:, count] = ocm0[:, p]
                count = count + 1

    elif Sub_run_name is 's2r2':
        for p in range(0, c0):
            if ocm0[-1, p] < 1500:  # are we sure we need this?
            #if ocm0[-1, p] < 15000:  # it turned out that no cleaning works better for this case
                ocm0_new[:, count] = ocm0[:, p]
                count = count + 1

    elif Sub_run_name is 's3r1':
        for p in range(0, c0):
            if ocm0[-1, p] < 3000 and ocm0[40, p] < 4500:
                ocm0_new[:, count] = ocm0[:, p]
                count = count + 1

    elif Sub_run_name is 's3r2':
        ocm0_new = ocm0
        count = ocm0_new.shape[1]

    ocm0_new = ocm0_new[:, 0:count]
    return ocm0_new
