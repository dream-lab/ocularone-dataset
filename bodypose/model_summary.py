from pose_estimation.trt_pose.tasks.human_pose.bodypose import BodyPoseModel
from torchinfo import summary

model = BodyPoseModel()
print(summary(model.model,(1,3,224,224)))