import torch
import sys, os

import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.utils.data as data
from PIL import Image
import numpy as np
import json
import random
from tqdm import tqdm
import pickle
# from sklearn.externals import joblib
from glob import glob
import pandas as pd


cate = ["Cardiomegaly", "Nodule", "Fibrosis", "Pneumonia", "Hernia", "Atelectasis", "Pneumothorax", "Infiltration", "Mass",
        "Pleural_Thickening", "Edema", "Consolidation", "No Finding", "Emphysema", "Effusion"]

category_map = {cate[i]:i+1 for i in range(15)}

class NIHDataset(data.Dataset):
    # In the following, the pickle files represent the serialized version of dictionary objects.
    # Specifically, each of such dict has image index as keys and a list of three elements as values (img name, img file, img labelvector):
    #           traindata = {
    #                           [0]:["00000003_002", np_img, np_labels],
    #                           [1]:["00000006_003", np_img, np_labels], ...
    #                           ...
    #                           [86523]:["00085471_001", np_img, np_labels], ...
    #                       }

    # ## Versione funzionante ma richiede caricamento in CPU RAM in toto, che potrebbe saturare la memoria
    # def __init__(self, data_path,input_transform=None,
    #              used_category=-1,train=True):
    #     self.data_path = data_path
    #     if train == True:
    #         print(f"NIHDataset - train: opening pickle file...")
    #         ####
    #         #'trainvaldata_dgx.pickle' questo e la versione 'testdata' sono quelli originali, con tutto il dataset
    #         #invece le versioni _debug hanno solo qualche paziente
    #         ###
    #         with open('trainvaldata_dgx.pickle', 'rb') as h:
    #         # with open('trainvaldata_new_debug.pickle', 'rb') as h: #TODO debug tmp
    #             self.data = pickle.load(h)
    #         # self.data = pickle.load(open('./traindata.pickle','rb'))
    #     else:
    #         print(f"NIHDataset - test: opening pickle file...")
    #         with open('testdata_dgx.pickle', 'rb') as h:
    #         # with open('testdata_new_debug.pickle', 'rb') as h: #TODO debug tmp
    #             self.data = pickle.load(h)
    #         # self.data = pickle.load(open('./testdata.pickle','rb'))
    #     print(f"NIHDataset - shuffling data...")
    #     random.shuffle(self.data)
    #     self.category_map = category_map
    #     self.input_transform = input_transform
    #     self.used_category = used_category


    # def __getitem__(self, index):
    #     img = Image.fromarray(self.data[index][1]).convert("RGB")
    #     label = np.array(self.data[index][2]).astype(np.float64)
    #     if self.input_transform:
    #         img = self.input_transform(img)
    #     return img, label

    # def getCategoryList(self, item):
    #     categories = set()
    #     for t in item:
    #         categories.add(t['category_id'])
    #     return list(categories)

    # def getLabelVector(self, categories):
    #     label = np.zeros(15)
    #     # label_num = len(categories)
    #     for c in categories:
    #         index = self.category_map[str(c)] - 1
    #         label[index] = 1.0  # / label_num
    #     return label

    # def __len__(self):
    #     return len(self.data)






    # ##TODO Versione funzionante con lazy loading per non saturare la RAM: non usiamo i file pickle pre-creati, ma carichiamo PNG singoli
    # def __init__(self, data_path,input_transform=None,
    #              used_category=-1,train=True):
    #     self.df = pd.read_csv('./Data_Entry_2017_v2020.csv') #TODO

    #     self.df = self.df.loc[:,["Image Index","Finding Labels"]]
    #     if train == True:
    #         images_path = './images/trainval/'
    #     else:
    #         images_path = './images/test/' ##the original version used the test set as inner validation.. so we should create an inner validation set from the trainval set instead.
    #     self.path_to_images = glob(images_path+"*.png")     
    #     self.category_map = category_map
    #     self.input_transform = input_transform
    #     # print(f"Dataset - INIT - read csv, len(path_to_images)={len(self.path_to_images)}")


    # def __getitem__(self, index):
    #     imgname = self.path_to_images[index]
    #     png_name = os.path.basename(imgname)    
    #     df_local=self.df.loc[self.df["Image Index"] == png_name]
    #     labels = df_local["Finding Labels"].values
    #     labels = labels[0].split("|") #eg, "[Cardiomegaly,"Effusion"]
    #     onehotencoding = [int(elem in labels) for elem in cate] #eg, [1,0,0,0,...,0,1]    
        
    #     img = Image.open(imgname).convert("RGB") #PIL
    #     if self.input_transform:
    #         img = self.input_transform(img)

    #     label = np.array(onehotencoding).astype(np.float64)

    #     return img, label

    # def getCategoryList(self, item):
    #     categories = set()
    #     for t in item:
    #         categories.add(t['category_id'])
    #     return list(categories)

    # def getLabelVector(self, categories):
    #     label = np.zeros(15)
    #     # label_num = len(categories)
    #     for c in categories:
    #         index = self.category_map[str(c)] - 1
    #         label[index] = 1.0  # / label_num
    #     return label

    # def __len__(self):
    #     return len(self.path_to_images) ##TODO



    ##TODO Versione aprile 2024
    
    def __init__(self, data_path,input_transform=None,
                 used_category=-1,train=True):
        
        if train == True:
            self.df = pd.read_csv('./Data_Entry_my_trainval.csv') #TODO
            self.images_path = './dataset_chestxray14/trainval/'
        else:
            self.df = pd.read_csv('./Data_Entry_my_test.csv') #TODO
            self.images_path = './dataset_chestxray14/test/' ##the original version used the test set as inner validation.. so we should create an inner validation set from the trainval set instead.
        
        # self.df = self.df.loc[:,["Image Index","Finding Labels"]]

        # self.path_to_images = glob(images_path+"*.png")     
        self.category_map = category_map
        self.input_transform = input_transform
        # print(f"Dataset - INIT - read csv, len(path_to_images)={len(self.path_to_images)}")


    def __getitem__(self, index):
        imgname = os.path.join(self.images_path, self.df.iloc[index, 0])        
        if "BAD" in imgname:
            raise ValueError   
        
        # labels= self.df.loc[index, 'Finding Labels']
        # labels = labels[0].split("|") #eg, "[Cardiomegaly,"Effusion"]
        # onehotencoding = [int(elem in labels) for elem in cate] #eg, [1,0,0,0,...,0,1]    
        # label = np.array(onehotencoding).astype(np.float64)
        ## TODO
        onehotencoding = [int(self.df.loc[index, elem]) for elem in cate] 
        label = np.array(onehotencoding).astype(np.float64)

        img = Image.open(imgname).convert("RGB") #PIL
        if self.input_transform:
            img = self.input_transform(img)

        

        return img, label

    def getCategoryList(self, item):
        categories = set()
        for t in item:
            categories.add(t['category_id'])
        return list(categories)

    def getLabelVector(self, categories):
        label = np.zeros(15)
        # label_num = len(categories)
        for c in categories:
            index = self.category_map[str(c)] - 1
            label[index] = 1.0  # / label_num
        return label

    def __len__(self):
        return len(self.df) ##TODO
