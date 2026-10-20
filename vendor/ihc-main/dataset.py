import os
import sys
import json
import albumentations as A

from tqdm import tqdm
from skimage import io
import cv2
from transforms import *
from torch.utils.data import Dataset
import traceback

def load_data(img_path, annot_path, num_classes):
    keys = ['image', 'keypoints'] + [f'keypoints{i}' for i in range(1, num_classes)]
    image = io.imread(img_path)
    values = [image]
    image_height, image_width = image.shape[:2]
    try:
        with open(annot_path, encoding='utf-8') as f:
            annotations = json.loads(f.read())
            values += [np.array(annotations[c]).reshape(-1, 2) for c in annotations['classes']]
            for value in values:
                value[:, 0] = np.clip(value[:, 0], 0, image_width - 1)  # 限制x坐标
                value[:, 1] = np.clip(value[:, 1], 0, image_height - 1)
    except:
        print(traceback.format_exc())
        values += [np.array([]).reshape(-1, 2) for _ in range(num_classes)]
    sample = dict(zip(keys, values))
    return sample


def img_loader(img_dir, ptn_dir, num_classes):
    data = []
    files = []
    reader = tqdm(os.listdir(img_dir), file=sys.stdout)
    reader.set_description(f"loading data from {img_dir}")
    for file in reader:
        files.append(file)
        img_path = os.path.join(img_dir, file)
        json_path = os.path.join(ptn_dir, os.path.splitext(file)[0] + '.json')
        sample = load_data(img_path, json_path, num_classes)
        data.append(sample)
    return data, files


class DataFolder(Dataset):
    def __init__(self, dataset, num_classes, is_train, data_transform, pre_load=False):
        self.img_dir = os.path.join(dataset, 'train_image' if is_train else 'test_image')
        self.ptn_dir = os.path.join(dataset, 'train_point' if is_train else 'test_point')
        self.data_transform = data_transform
        self.pre_load = pre_load
        self.num_classes = num_classes
        if self.pre_load:
            self.data, self.files = img_loader(self.img_dir, self.ptn_dir, num_classes)
        else:
            self.data = []
            self.files = os.listdir(self.img_dir)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index: int):
        assert index <= len(self), 'index range error'
        index = index % len(self.files)
        if self.pre_load and len(self.data) > 0:
            sample = self.data[index]
        else:
            img_path = os.path.join(self.img_dir, self.files[index])
            json_path = os.path.join(self.ptn_dir, os.path.splitext(self.files[index])[0] + '.json')
            sample = load_data(img_path, json_path, self.num_classes)
        sample = self.data_transform(sample)
        return sample


def build_dataset(args, is_train=True):
    mean, std = np.load(os.path.join(args.dataset, 'mean_std.npy'))
    additional_targets = {}
    for i in range(1, args.num_classes):
        additional_targets.update({'keypoints%d' % i: 'keypoints'})
    if is_train is True or is_train == 'train':
        augmentor = A.Compose([
            #A.ShiftScaleRotate(scale_limit=0.3, rotate_limit=0, shift_limit=0, border_mode=0, value=0, p=0.5),
            #A.Rotate(limit=[-90,90],p=0.5),
            #A.RandomGridShuffle(grid=(3,3),always_apply=False,p=1),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            #A.RandomCrop(height=1080, width=1080, always_apply=True),
            #A.RandomCrop(height=768, width=768, always_apply=True),
        ], p=1, keypoint_params=A.KeypointParams(format='xy'), additional_targets=additional_targets)
        transform = Preprocessing(mean, std, augmentor)
    else:
        transform = Preprocessing(mean, std)

    data_folder = DataFolder(args.dataset, args.num_classes, is_train, transform, pre_load=args.pre_load)
    return data_folder
