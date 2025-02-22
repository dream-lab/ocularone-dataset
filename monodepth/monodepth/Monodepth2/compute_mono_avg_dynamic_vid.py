import cv2
import statistics
import pandas as pd
from pypfm import PFMLoader
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

# dist = ['2', '2_5', '3', '3_5', '4']      
dist = ['2_5', '4']
file_names = []
yolo_files = []

# mode_sr_c5_a = 1988.469
# mode_sr_c5_b = -75.546

# def find_dist_from_depth(pred):
#     return (mode_sr_c5_a * pred) + mode_sr_c5_b

for val in range(len(dist)):
    # file_names.append('run_3')
    # yolo_files.append('run_3_vip')
    file_names.append('tanushree_CDS_morn_bend_' + dist[val] + 'm')
    yolo_files.append('tanushree_CDS_morn_bend_' + dist[val] + 'm')

# methods = ['mode', 'disc', 'circum', 'center', 'center_50', 'ten_percent', 'twenty_five_percent', 'fifty_percent', 'median', 'mean', 'all_equal']
# methods = ['circum', 'ten_percent']
# methods = ['10th_percentile']
methods = ['ten_percent']

for method in methods:
    print(method)
    for m in tqdm(range(len(file_names))):
        path = '/media/dream-onano7/Backup_Plus/Prabhav/Everything_IROS/Datasets/Monodepth2_output/Monodepth2_depth/feb_14_24/tanushree_CDS_morn_bend/' + file_names[m] + '/'
        yolo_path = '/media/dream-onano7/Backup_Plus/Prabhav/Everything_IROS/orientation/yolo_files/feb_14_24/tanushree_CDS_morn_bend/' + yolo_files[m] + '.csv'
        x = pd.read_csv(yolo_path)
        df = pd.DataFrame()
        df_arr = []
        dist_arr = []
        vid_num_arr = []

        for i in os.listdir(path):
            print(i)
            if i.split('.')[1] == 'npy':
                # loader = PFMLoader(color=False, compress=False)
                # new_pfm = loader.load_pfm(path + i)
                new_pfm = np.load(path + i).T
                ind = i.split('.')[0]
                j = 0
                flag = 0
                for v in x['vid_name']:
                    if int(v) == int(ind):
                        # print("exists")
                        flag = 1
                        break
                    j = j + 1

                if(flag == 0):
                    continue

                x_scale = 1024 / 1280
                y_scale = 320 / 720
                center_x = round(x['cx'][j] * x_scale)
                center_y = round(x['cy'][j] * y_scale)
    
                xl = round(x['tlx'][j] * x_scale)
                yl = round(x['tly'][j] * y_scale)
                xr = round(x['brx'][j] * x_scale)
                yr = round(x['bry'][j] * y_scale)
                width = round(x['w'][j] * x_scale)
                height = round(x['h'][j] * y_scale)

                only_object = new_pfm[xl:xr, yl:yr]
            
                top_right_x = center_x + round(0.25 * only_object.shape[0])
                top_right_y = center_y + round(0.25 * only_object.shape[1])

                top_left_x = center_x - round(0.25 * only_object.shape[0])
                top_left_y = center_y + round(0.25 * only_object.shape[1])

                bottom_right_x = center_x + round(0.25 * only_object.shape[0])
                bottom_right_y = center_y - round(0.25 * only_object.shape[1])

                bottom_left_x = center_x - round(0.25 * only_object.shape[0])
                bottom_left_y = center_y - round(0.25 * only_object.shape[1])

                radi = round(0.25*(min(width, height))/2)
                radi = min(20, radi)
                leftp = center_x - radi
                rightp= center_x + radi
                upp = center_y + radi
                downp = center_y - radi

                p = new_pfm[downp:upp, leftp:rightp]

                if(method == 'circum' or method == 'disc'):
                    count = 0
                    li_coords = []
                    for i in range(leftp, rightp + 1):
                        for j in range(downp, upp+1):
                            count += 1
                            if(method == 'circum'):
                                if((i - center_x)**2 + (j - center_y)**2 == radi**2):
                                    li_coords.append((i,j))
                            elif(method == 'disc'):
                                if((i - center_x)**2 + (j - center_y)**2 <= radi**2):
                                    li_coords.append((i,j))
                    li_sum = 0
                    for i in li_coords:
                        li_sum += new_pfm[i[0]][i[1]]
                    weighted_avg = li_sum/len(li_coords) 
            
                elif(method == 'mode' or method == 'median' or method == 'mean' or method == 'ten_percent' or method == 'twenty_five_percent' or method == 'fifty_percent' or method == '10th_percentile'):
                    x_num_arr = []
                    for val in only_object:
                        for x_num in val:
                            x_num_arr.append(x_num)
                    
                    # print(len(x_num_arr))
                    sorted_x_num_arr = sorted(x_num_arr)
                    
                    if(method == 'mode'):
                        if(len(sorted_x_num_arr) > 0):
                            weighted_avg = statistics.mode(sorted_x_num_arr)
                        else:
                            continue

                    if(method == 'ten_percent'):
                        ten_percent = round(0.1 * len(x_num_arr))
                        weighted_avg = np.average(sorted_x_num_arr[:ten_percent])

                    elif(method == 'twenty_five_percent'):
                        twenty_five_percent = round(0.25 * len(x_num_arr))
                        weighted_avg = np.average(sorted_x_num_arr[:twenty_five_percent])
            
                    elif(method == 'fifty_percent'):
                        fifty_percent = round(0.5 * len(x_num_arr))
                        weighted_avg = np.average(sorted_x_num_arr[:fifty_percent])
                    
                    elif(method == '10th_percentile'):
                        tenth_percentile = np.percentile(sorted_x_num_arr, 10)
                        weighted_avg = tenth_percentile

                    elif(method == 'median'):
                        weighted_avg = statistics.median(x_num_arr)
                
                    elif(method == 'mean'):
                        weighted_avg = statistics.mean(x_num_arr)
                
                elif(method == 'center'):
                    weighted_avg = new_pfm[center_x][center_y] 
                
                elif(method == 'center_50'):
                    weighted_avg = 0.5 * new_pfm[center_x][center_y] + 0.125 * (new_pfm[top_right_x][top_right_y] + new_pfm[top_left_x][top_left_y] + new_pfm[bottom_left_x][bottom_left_y] + new_pfm[bottom_right_x][bottom_right_y]) 
                
                elif(method == 'all_equal'):
                    weighted_avg = 0.2 * new_pfm[center_x][center_y] + 0.2 * (new_pfm[top_right_x][top_right_y] + new_pfm[top_left_x][top_left_y] + new_pfm[bottom_left_x][bottom_left_y] + new_pfm[bottom_right_x][bottom_right_y])
                
                # df_arr[int(ind)] = weighted_avg
                df_arr.append(weighted_avg)
                vid_num_arr.append(int(ind))
                # dist_arr.append(find_dist_from_depth(weighted_avg))
        df['Predicted'] = df_arr
        df['vid_num'] = vid_num_arr
        # df['distance'] = dist_arr
        # directory = '/media/dream-onano7/Backup_Plus/Prabhav/Everything_IROS/orientation/averaging/Monodepth2_method/feb_14_24_averaging/tanushree_CDS_morn_bend/' +  method + '/'
        # if not os.path.exists(directory):
        #    os.makedirs(directory)
        #df.to_csv('/media/dream-onano7/Backup_Plus/Prabhav/Everything_IROS/orientation/averaging/Monodepth2_method/feb_14_24_averaging/tanushree_CDS_morn_bend/' + method + '/' + yolo_files[m] + '_monodepth_' + method + '_wt_avg.csv')
        return depth_array
