import shutil
import cv2
import os
import numpy as np

def mkdir(path):
    if isinstance(path, str) :
        os.makedirs(path, exist_ok=True)
    elif isinstance(path, list) :
        for path_elem in path :
            os.makedirs(path_elem, exist_ok=True)

def action2cls():
    """
    Root
    /home/ty/DB/mmaction2/data/classification_data
    train
        - normal
            -foldername
                -img_00001.jpg
                -img_00002.jpg
        - cancer
    test
        - normal
        - cancer

    To

    train
        - normal
            foldername.jpg
        - cancer
            foldername.jpg
    test
        - normal
            foldername.jpg
        - cancer
            foldername.jpg


    Returns:

    """
    RootDir = '/home/ty/DB/mmaction2/data/classification_data'
    MoveDir = '/home/ty/Project/mmclassification/data/basbaiImgNet'
    TrainDir = os.path.join(MoveDir, 'train')
    TestDir = os.path.join(MoveDir, 'test')
    mkdir(MoveDir)
    TType = ['train', 'test']
    mkdir([TrainDir, TestDir])
    ClsType = ['normal', 'cancer']

    for tt_type in TType :
        for cls_type in ClsType :
            PrevDir = os.path.join(RootDir, tt_type, cls_type)
            SaveDir = os.path.join(MoveDir, tt_type, cls_type)
            mkdir(SaveDir)

            for folder_name in os.listdir(PrevDir) :
                new_img_name = os.path.join(SaveDir, folder_name + '.jpg')
                img1_path = os.path.join(PrevDir, folder_name, 'img_00001.jpg')
                img2_path = os.path.join(PrevDir, folder_name, 'img_00002.jpg')
                img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
                img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
                # print(img1.shape, img2.shape)
                # assert(img1.shape == img2.shape)

                resize_img1 = cv2.resize(img1, (400, 600))
                resize_img2 = cv2.resize(img2, (400, 600))

                resize_shape = np.zeros((600, 400, 3), np.uint8)
                resize_shape[:, :, 0] = resize_img1
                resize_shape[:, :, 1] = resize_img2

                cv2.imwrite(new_img_name, resize_shape)




















if __name__ == "__main__" :
    action2cls()