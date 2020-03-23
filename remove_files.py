import os

dir = 'C:\\Users\\Kwon\\PycharmProjects\\PancOCM_ML\\'


sr_list = ['s1r1', 's1r2', 's2r1', 's2r2', 's3r1','s3r2']
ang_list = ['0']
ocm_list = ['0','1','2']

for fidx in range(0, len(sr_list)):
    for ang_idx in range(0, len(ang_list)):
        for ocm_idx in range(0, len(ocm_list)):
            file_name = 'GLCM'+sr_list[fidx]+'_dist5_angle'+str(ang_list[ang_idx])+'.png'
            print('file name:', file_name)
            full_name = dir+file_name
            if os.path.exists(full_name):
                os.remove(full_name)
                print('file', file_name, 'was found!')