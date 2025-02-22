from monodepth.load_model import LoadMonodepthModel
from monodepth.inference import MonoLTInferencing
from torchinfo import summary

models_list = LoadMonodepthModel().load_model()
mono = MonoLTInferencing()
for model in models_list:
    print("+"*50)
    print("+"*50)
    print("+"*50)
    print(summary(model))
    print("+"*50)
    print("+"*50)
    print("+"*50)