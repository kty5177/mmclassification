import mlflow
import cv2
import torch
from mmcls.datasets import build_dataloader, build_dataset
import mmcv
from mmcls.apis import multi_gpu_test, single_gpu_test


mlflow.set_tracking_uri("http://localhost:5003")
mlflow.set_experiment('test-mmcls2')

logged_model = 'runs:/fc73f916025949dd808f3e233e1568df/models'
loaded_model = mlflow.pytorch.load_model(logged_model)
print(loaded_model)
print(type(loaded_model))

img = cv2.imread('demo/dog.jpg')
# tensor_img = torch.Tensor(img)
# loaded_model.predict(pd.DataFrame(img.reshape(374,1500)))

#
#
# print(loaded_model)
# print(loaded_model.type)
#
# loaded_model.eval()
configpath = 'configs/resnet/resnet18_8xb16_ingender.py'
cfg = mmcv.Config.fromfile(configpath)
dataset = build_dataset(cfg.data.test, default_args=dict(test_mode=True))
# the extra round_up data will be removed during gpu/cpu collect
data_loader = build_dataloader(
    dataset,
    samples_per_gpu=cfg.data.samples_per_gpu,
    workers_per_gpu=cfg.data.workers_per_gpu,
    dist=False,
    shuffle=False,
    round_up=True)

results = []
dataset = data_loader.dataset
for i, data in enumerate(data_loader):
    print(data)
    print(type(data))
    with torch.no_grad():
        result = loaded_model(return_loss=False, **data)
        print(result)

# img = cv2.imread('demo/dog.jpg')
# tensor_img = torch.Tensor(img)
# pred = loaded_model(tensor_img)2

# print(pred)


