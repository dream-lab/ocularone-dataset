import time
import numpy as np
import pandas as pd
import cv2
import pickle

# image_path = '/home/prabhav/bodypose_to_Mediapipe/image/fall_0_resized.jpg'

# 4.jpg is good for mediapipe.

# image = cv2.imread(image_path)
# csv_path = '/home/prabhav/Desktop'

def pose_detect_mp(image):
        mp_drawing = mp.solutions.drawing_utils
        mp_pose = mp.solutions.pose
        with mp_pose.Pose(static_image_mode=True, model_complexity=2, enable_segmentation=True, min_detection_confidence=0.5) as pose:
                results = pose.process(image)
                # print(results.pose_landmarks)
                mp_drawing.draw_landmarks(image,results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                                                        mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                                                                        mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=1))
                pose_coords = results.pose_landmarks.landmark
                print([[landmark.x, landmark.y, landmark.z] for landmark in pose_coords])               # To test bodypose after mediapipe output.
                pose_row = list(np.array([[landmark.x, landmark.y, landmark.z] for landmark in pose_coords]).flatten())
                print('\n',pose_row,'\n')
                print(len(pose_row)//3)
                cv2.imshow("Output pose", image)
                return pose_row


def load_pose_classify():
        #load pose classification model
        with open('/home/sumanraj/bodypose/body_language.pkl', 'rb') as f:
                pose_classify_model = pickle.load(f)
        return pose_classify_model

def pose_detect(pose_classify_model, pose_row):
        #inference
        t1 = time.time()
        inf_time = time.time()-t1

    #pose classify
        try:
                # Extract Pose landmarks
                # pose_coords = results.pose_landmarks.landmark
                # pose_row = list(np.array([[landmark.x, landmark.y, landmark.z] for landmark in pose_coords]).flatten())

                # Add code to read the csv output. For now, it is the test input
                # pose_row = [
                #       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5446428571428571, 0.35267857142857145, 0.0, 0.6026785714285714, 0.3482142857142857, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5223214285714286, 0.46875, 0.0, 0.6339285714285714, 0.45982142857142855, 0.0, 0.5089285714285714, 0.6607142857142857, 0.0, 0.65625, 0.6517857142857143, 0.0, 0.5044642857142857, 0.8125, 0.0, 0.6741071428571429, 0.8035714285714286, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5491071428571429, 0.8125, 0.0, 0.6205357142857143, 0.7991071428571429, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                #       ]
                X = pd.DataFrame([pose_row])
                body_language_class = pose_classify_model.predict(X)[0]
                # cv2.putText(image, body_language_class, (20,50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (234, 221, 202), 3, cv2.LINE_AA)
        except Exception as e:
                body_language_class = 'None'
                print('Exception::', e)
                pass
        # return image, inf_time, body_language_class
        return inf_time, body_language_class

# pose_clasify_model = load_pose_classify()
# image, inference_time, pose_class = pose_detect(image,pose_clasify_model,pose_detect_mp(image))               # To classify on mediapipe output directly

# cv2.imshow('Output',image)

# pose_row = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.33482142857142855, 0.5446428571428571, 0.0, 0.36160714285714285, 0.5401785714285714, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.33482142857142855, 0.5982142857142857, 0.0, 0.3705357142857143, 0.5803571428571429, 0.0, 0.34375, 0.6741071428571429, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4017857142857143, 0.7008928571428571, 0.0, 0.41964285714285715, 0.6875, 0.0, 0.5, 0.6428571428571429, 0.0, 0.5044642857142857, 0.6473214285714286, 0.0, 0.0, 0.0, 0.0, 0.5803571428571429, 0.6875, 0.0, 0.0, 0.0, 0.0, 0.5803571428571429, 0.6875, 0.0, 0.0, 0.0, 0.0, 0.5803571428571429, 0.6875, 0.0]
# image, inference_time, pose_class = pose_detect(image,pose_clasify_model,pose_row)            # To classify on bodypose output, with bodypose output given as the array above
# inference_time, pose_class = pose_detect(image,pose_clasify_model,pose_row)           # To classify on bodypose output, with bodypose output given as the array above
# print(pose_class)

cv2.waitKey()
