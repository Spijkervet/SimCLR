from types import NoneType
from torch.utils.data import Dataset
from PIL import Image
import os
import torch
from torchvision import transforms
import pandas as pd


class TFIWDataset(Dataset):
    def __init__(self, img_dir = os.getcwd(), transform = None):
        self.img_dir = img_dir
        self.transform = transform

        self.img_names = os.listdir(img_dir)

        file_names = []
        labels = []
        for i in self.img_names:
            #print(i[:-3])
            if(i[-3:]=='jpg'):
                file_names.extend([i]) #to remove unwanted files names from the img_names like .DS_Store etc.
                labels.extend([int(i[1:5])])
        self.labels = labels
        self.img_names = file_names

        img_names_csv = pd.DataFrame(data= [file_names, self.labels]);
        #img_names_csv['Labels'] = self.labels
        img_names_csv.T.to_csv("/Users/gaurav/Desktop/data.csv")
        #print(self.img_names[0:5])
        #print(self.labels[0:5])

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.img_dir, self.img_names[idx]))
        #image = torch.tensor(image)
        if type(image)!=NoneType: #Some images were throwing empty tensors, hence did this.
            if self.transform is not None:
                image = self.transform(image)
            try:
                #print(idx, self.labels[idx], self.img_names[idx])
                return image, self.labels[idx]       
            except IndexError:
                print(f"Index is not present for index number {idx}")
            
    def __len__(self):
        return len(self.img_names)

#tfiw = TFIWDataset(img_dir='/Users/gaurav/Desktop/thesis-work/Datasets/T-1/train-faces/all')
#example = tfiw[7]
#print(example)