import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


# 预处理过程
preprocess = transforms.Compose([
    # 转换为张量
    transforms.ToTensor(),
    # 归一化
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def parse_label(file_path):
    '''
    解析标注文件
    '''
    with open(file_path, 'r') as fr:
        class_indexes = []
        centers = []
        # 逐行读入
        for l in fr.readlines():
            es = l.split(' ')
            class_indexes.append(ord(es[0]) - 48)
            centers.append([
                (float(es[1]) + float(es[3])) / 2,
                (float(es[2]) + float(es[4])) / 2,
            ])
        class_indexes = np.array(class_indexes, dtype=np.int32)
        centers = np.array(centers, dtype=np.float32)
        return class_indexes, centers

def get_label_text(file_path):
    with open(file_path, 'r') as fr:
        return ''.join([l.split(' ')[0] for l in fr.readlines()])

class CaptchaDataset(Dataset):

    def __init__(self, txt_path, num_classes, training=True):
        self.num_classes = num_classes
        self.training = training
        with open(txt_path, 'r') as fr:
            self.list = [l.strip() for l in fr.readlines() if l.strip()]
        self.augment = preprocess

    def __len__(self):
        return len(self.list)

    def __getitem__(self, index):
        file_path = self.list[index]
        image = Image.open(file_path)
        image = self.augment(image)

        label_path = file_path.replace('images', 'labels').replace('png', 'txt')
        label = self.create_label(*parse_label(label_path))\
            if self.training else get_label_text(label_path)
        return image, label

    def create_label(self, cls_indexes, centers):
        '''生成标签'''
        h, w = 5, 10
        label = np.zeros((h, w, 2 + self.num_classes), dtype=np.float32)
        for ci, center in zip(cls_indexes, centers):
            x, y = (center // 16).astype(np.int32)

            onehot = np.zeros(self.num_classes, dtype=np.float32)
            onehot[ci] = 1.0

            label[y, x, :] = np.concatenate((center, onehot))
        return label
