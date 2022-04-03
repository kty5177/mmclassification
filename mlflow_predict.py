import mlflow
import cv2
import torch

mlflow.set_tracking_uri("http://localhost:5003")
mlflow.set_experiment('test-mmcls2')

logged_model = 'runs:/fc73f916025949dd808f3e233e1568df/models'
loaded_model = mlflow.pytorch.load_model(logged_model)

img = cv2.imread('demo/dog.jpg')
tensor_img = torch.Tensor(img)
pred = loaded_model(tensor_img)
print(pred)


