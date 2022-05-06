from torch.utils.data import Dataset
import os
import pandas as pd
import cv2
from torchvision import transforms
from PIL import Image
from albumentations import Compose, HueSaturationValue, RandomBrightnessContrast, OneOf, IAAAdditiveGaussianNoise, MotionBlur, GaussianBlur, ImageCompression, GaussNoise, Resize, RandomCrop


data_transform = {
    "train": transforms.Compose([#transforms.RandomCrop(240),transforms.Resize(480),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.RandomVerticalFlip(),
                                 transforms.ToTensor(),
                                 #transforms.Normalize([0.39, 0.28, 0.27], [0.21, 0.16, 0.15])
                                 ]),
    "val": transforms.Compose([
        transforms.CenterCrop(240),
        transforms.Resize(380),
        transforms.ToTensor(),
        #transforms.Normalize([0.39, 0.28, 0.27], [0.21, 0.16, 0.15])
    ])
}


class MyDataset(Dataset):
    def __init__(self, label_file, root_dir, phase=None):

        self.labels = pd.read_csv(label_file)
        self.root_dir = root_dir
        self.pahse = phase

    #  return the number of samples in dataset
    def __len__(self):
        return len(self.labels)

    #  return a sample at the given index
    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.labels.iloc[index, 1])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.pahse == 'train':
            aug = Compose([
            RandomCrop(240, 240),
            Resize(380,380,1,1),
            img = aug(image=img)['image']
            tfms = data_transform['train']
        else:
            tfms = data_transform['val']
        img = Image.fromarray(img.astype('uint8')).convert('RGB')
        img = tfms(img).unsqueeze(0)
        label = self.labels.iloc[index, 0]
        img = face.squeeze(0)

        return face, label
