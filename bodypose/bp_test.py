# import csv
import numpy as np
import classifier_test as ct_bp


# res=  [{'left_ear': (75, 122), 'right_ear': (81, 121), 'left_shoulder': (75, 134), 'right_shoulder': (83, 130), 'left_elbow': (77, 151), 'left_hip': (90, 157), 'right_hip': (94, 154), 'left_knee': (112, 144), 'right_knee': (113, 145), 'right_ankle': (130, 154), 'neck': (79, 131)}]       # fall_0_resized.jpg

pose_clasify_model = ct_bp.load_pose_classify()

def get_pose(res):
    conversion_table = {
    'nose':0,
    'left_eye':2,
    'right_eye':5,
    'left_ear':7,
    'right_ear':8,
    'left_shoulder':11,
    'right_shoulder':12,
    'left_elbow':13,
    'right_elbow':14,
    'left_wrist':15,
    'right_wrist':16,
    'left_hip':23,
    'right_hip':24,
    'left_knee':25,
    'right_knee':26,
    'left_ankle':27,
    'right_ankle':28,
    }

    # csv_path = '/home/prabhav/Desktop'
# csv_path = '.'
    csv_files = ['tello.csv','upright.csv', 'start_stop.csv', 'land_go_back.csv', 'kneel.csv', 'fall.csv']
    classes = ['timepass', 'upright', 'start_stop', 'land_go_back', 'kneel', 'fall']
    labels = [0, 1, 2, 3, 4, 5]

    k = 2       # class selector
    # curr_csv = f'{csv_path}/{csv_files[k]}'
    class_name = classes[k]
    label = labels[k]
    # print(curr_csv, class_name, label)

    landmarks = ['img_name', 'class', 'label']
    dummy_array = []        # keep dummy_array as empty for use only with bodypose
    for val in range(1, 34):
        landmarks += ['x{}'.format(val), 'y{}'.format(val), 'z{}'.format(val)]
        dummy_array.append([0,0,0])
    # with open(curr_csv, mode='w', newline='') as f:
        # csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        # csv_writer.writerow(landmarks)

# print(landmarks)
# print(dummy_array)

    for i in res:
        # print(i)
        for key in i:
            if key in conversion_table:
                # print("Yay!!", conversion_table[i])
                coords = list(i[key])
                for j in range(2):
                    coords[j] /= 224       # basically the scale factor for normalization
                coords.append(0)
                # print(coords)
                
                dummy_array[conversion_table[key]] = coords
                if key == "left_eye":
                    dummy_array[conversion_table[key]-1] = coords
                    dummy_array[conversion_table[key]+1] = coords
                if key == "right_eye":
                    dummy_array[conversion_table[key]-1] = coords
                    dummy_array[conversion_table[key]+1] = coords
                if key == "neck":
                    dummy_array[9] = coords
                    dummy_array[10] = coords
                if key == "left_wrist":
                    dummy_array[conversion_table[key]+2] = coords
                    dummy_array[conversion_table[key]+4] = coords
                    dummy_array[conversion_table[key]+6] = coords
                if key == "right_wrist":
                    dummy_array[conversion_table[key]+2] = coords
                    dummy_array[conversion_table[key]+4] = coords
                    dummy_array[conversion_table[key]+6] = coords
                if key == "left_ankle":
                    dummy_array[conversion_table[key]+2] = coords
                    dummy_array[conversion_table[key]+4] = coords
                if key == "right_ankle":
                    dummy_array[conversion_table[key]+2] = coords
                    dummy_array[conversion_table[key]+4] = coords

            else:
                # print("No!!")
                pass

    # print('\n',dummy_array)
        pose_row = list(np.asarray(dummy_array).flatten())

    # print(pose_row)

    # pose_row.insert(0, 'test_img.png')      # 0th index contains image name
    # pose_row.insert(1, class_name)      # 0th index contains image name
    # pose_row.insert(2, label)      # 0th index contains image name

    # pose_clasify_model = ct_bp.load_pose_classify()
    inference_time, pose_class = ct_bp.pose_detect(pose_clasify_model,pose_row)           # To classify on bodypose output, with bodypose output given as the array above
    print("Output of body pose = ", pose_class)
    if(pose_class == 'fall'):
        print("The VIP has fallen")

    return (inference_time, pose_class)

    # with open(curr_csv, mode='a', newline='') as f:
        # csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        # print('logged pose landmarks')
        # csv_writer.writerow(pose_row)
