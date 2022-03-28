import shutil
import os

"""
add_meta : val 폴더로 data/GenderImgNet/meta/val.txt 생성

파일경로 0, 1, 2 (Female, Male, Unknown)
"""

def add_meta():
    root_path = 'data/GenderImgNet/val'
    meta_file = 'data/GenderImgNet/meta/val.txt'
    ClsType = ['Female', 'Male', 'Unknown']

    for idx, Cls in enumerate(ClsType) :
        FileList = os.listdir(os.path.join(root_path, Cls))
        for file in FileList :
            with open(meta_file, 'a') as f :
                f.write('%s %d\n'%(os.path.join(Cls, file), idx))

if __name__ == "__main__" :
    add_meta()
