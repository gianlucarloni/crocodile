import torch
import sys, os

import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.utils.data as data
import PIL
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import numpy as np
import json
import random
from tqdm import tqdm
import pickle
# from sklearn.externals import joblib
from glob import glob
import pandas as pd
####################################################TODO
import logging
logging.basicConfig(level=logging.INFO)
#####################################################TODO

def adjustContrast(img):    
    img_array = np.array(img)        

    if np.isnan(img_array).any():            
        np.nan_to_num(img_array,copy=False)
        # logging.info(f"ATTENTION: img_array contained NaN, replaced with np.nan_to_num")
    x_min = img_array.min()
    x_max = img_array.max()
    # logging.info(f"x_min {x_min}, x_max {x_max}")

    if x_max != x_min:            
        img_array = 255.0 * ((img_array - x_min) / (x_max - x_min))
    # else:
    #     logging.info(f"ATTENTION: img_array contained x_max == x_min, most likely all zeros")           
    
    return Image.fromarray(img_array.astype('uint8')) #, 'L')

# the 14+1 classes registered in the NIH ChestXray-14 dataset
# cate = ["Cardiomegaly", "Nodule", "Fibrosis", "Pneumonia", "Hernia", "Atelectasis", "Pneumothorax", "Infiltration", "Mass",
        # "Pleural_Thickening", "Edema", "Consolidation", "No Finding", "Emphysema", "Effusion"]
# category_map = {cate[i]:i+1 for i in range(15)}

#%% Define the common features across different CXR datasets to be used. For instance the following 9 findings might be relevant:
cate = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 'Lung Opacity', 'No Finding', 'Pneumonia', 'Pneumothorax']
# category_map = {cate[i]:i+1 for i in range(9)}

#%%
class CXRDataset(data.Dataset): 
    #generic builder class for CXR dataset: specifying the data_path name of the source dataset we could create different version of CXR dataset
    
    def __init__(self, data_path,
                 input_transform=None, used_category=-1, train=True, use_tSNE_selfreportedrace=False, adjustContrast=True):
        
        self.data_path=data_path
        self.isTrain=train #True, otherwise False (valid)
        self.use_tSNE_selfreportedrace=use_tSNE_selfreportedrace #TODO
        self.adjustContrast = adjustContrast
        print(f"self.adjustContrast: {self.adjustContrast}")
        
        self.srrace_to_int={}
        if use_tSNE_selfreportedrace:
            self.srrace_to_int = {
                'Asian':0,
                'Black':1,
                'Other':2,
                'White':3
            }
            print(f"self.srrace_to_int: {self.srrace_to_int}")


        if self.data_path.endswith("dataset_chestxray14"):
            dataset_name="dataset_chestxray14"
            if train == True:
                self.df = pd.read_csv(os.path.join(data_path,'Data_Entry_my_trainval.csv')) #TODO                
                self.images_path = os.path.join(data_path,'trainval')
            else:
                self.df = pd.read_csv(os.path.join(data_path,'Data_Entry_my_test.csv')) #TODO
                self.images_path = os.path.join(data_path,'test')            
        elif self.data_path.endswith("dataset_chexpert"):
            dataset_name="dataset_chexpert"
            if train:
                if self.use_tSNE_selfreportedrace:
                    self.df = pd.read_csv(os.path.join(data_path,'train_frontal_demog.csv'))
                else:
                    self.df = pd.read_csv(os.path.join(data_path,'train_frontal.csv'))
                
                self.images_path = os.path.join(data_path,'train')
            else:
                if self.use_tSNE_selfreportedrace:
                    self.df = pd.read_csv(os.path.join(data_path,'valid_frontal_demog.csv'))
                else:
                    self.df = pd.read_csv(os.path.join(data_path,'valid_frontal.csv'))
                self.images_path = os.path.join(data_path,'valid')             
        elif self.data_path.endswith("dataset_padchest"):
            dataset_name="dataset_padchest"
            if train:
                self.df = pd.read_csv(os.path.join(data_path,'./csv_files_new/train80val20_train.csv'))
                
            else:
                self.df = pd.read_csv(os.path.join(data_path,'./csv_files_new/train80val20_val.csv'))

            self.images_path = data_path # here, there are the 52 subfolders, each containing around <3000 PNG images: thanks to the CSV file above, we can access the correct subfolder to locate the desired image filename to load. 
        
        # elif self.data_path.endswith("dataset_mimicCXR_JPG"): # TODO devo ancora finire di scaricare i dicom, da cui selezionare i JPG frontali e prenderne i metadati per stratificare e creare i csv split.
        #     if train:
        #         # self.df = pd.read_csv(os.path.join(data_path,'physionet.org/files/mimic-cxr-jpg/2.0.0/mimic-gcExtended-train.csv'))  
        #         self.df = pd.read_csv(os.path.join(data_path,'physionet.org/files/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-chexpert_gc_extended.csv'))  #TODO
                           
        #     else:
        #         self.df = pd.read_csv(os.path.join(data_path,'physionet.org/files/mimic-cxr-jpg/2.0.0/mimic-gcExtended-val.csv'))
        #     self.images_path = os.path.join(data_path,'physionet.org/files/mimic-cxr-jpg/2.0.0/files')
        #     # self.df.insert(loc=0, column='dataset_name', value = ['dataset_mimicCXR_JPG' for _ in range(len(self.df))])
           
        # print(f"Sto preparando la init del {dataset_name}, di lunghezza {len(self.df)} ")# e colonne: {self.df.columns}\n E di cui voglio selezionare solo le categorie: {cate}, più la 0_IMAGEINDEXORPATH") 
        if self.use_tSNE_selfreportedrace:
            expansion_list = ['0_IMAGEINDEXORPATH','self-reported-race']
        else:
            expansion_list = ['0_IMAGEINDEXORPATH']
        self.df = self.df.loc[:, cate + expansion_list]
        self.df.sort_index(axis=1, inplace=True)
        self.df.insert(loc=0, column='dataset_name', value = [dataset_name for _ in range(len(self.df))])    

        # self.path_to_images = glob(images_path+"*.png")     
        # self.category_map = category_map
        self.input_transform = input_transform
        # print(f"Dataset - INIT - read csv, len(path_to_images)={len(self.path_to_images)}")

    def __getitem__(self, index):
        # Depending on the dataset the data came from, it has a peculiar getitem policy, so we first need to determine the source:
        label_dataset=self.df.at[index, 'dataset_name']

        if label_dataset=="dataset_chestxray14":
            label_dataset=0 #TODO
            # 
            # imgname = os.path.join(self.images_path, self.df.iloc[index, 0]) # in NIH dataset, it is easy to locate a file since the CSV already contains the relative path, so you just need to prepend an absolute dir       
            imgname = os.path.join(self.images_path, self.df.at[index, '0_IMAGEINDEXORPATH']) #
            
            # labels= self.df.loc[index, 'Finding Labels']
            # labels = labels[0].split("|") #eg, "[Cardiomegaly,"Effusion"]
            # onehotencoding = [int(elem in labels) for elem in cate] #eg, [1,0,0,0,...,0,1]    
            # label = np.array(onehotencoding).astype(np.float64)
            
        
        elif label_dataset=="dataset_chexpert":
            label_dataset=1 #TODO
            _imgname = self.df.at[index, '0_IMAGEINDEXORPATH'] #e.g.: CheXpert-v1.0-small/valid/patient64541/study1/view1_frontal.jpg
            if self.isTrain:
                s = _imgname.split(sep="train/")
            else:
                s = _imgname.split(sep="valid/") #e.g.: ['CheXpert-v1.0-small/','patient64541/study1/view1_frontal.jpg']
            imgname = os.path.join(self.images_path, s[-1])

            ##TODO
            if self.use_tSNE_selfreportedrace:
                selfreportedrace = self.df.at[index,'self-reported-race']
                srrace = self.srrace_to_int[selfreportedrace]
            
        
        elif label_dataset=="dataset_padchest":
           
            label_dataset=2 #TODO
            # _ImageDir=self.df.at[index,'ImageDir']
            # _ImageID=self.df.at[index,'ImageID']  
            _ImageDir_and_ID = self.df.at[index,'0_IMAGEINDEXORPATH'] 
            _ImageDir, _ImageID = _ImageDir_and_ID.split(sep="/")    
            imgname = os.path.join(self.images_path, _ImageDir, _ImageID)
            
        
        # elif label_dataset=="dataset_mimicCXR_JPG":
        #     label_dataset=3 #TODO ancora da sistemare mimic
        #     imgname =
        #     label =
        else:
            # logging.info(f"CXR Datasets - getitem(): label_dataset {label_dataset} non trovata fra quelle disponibili")
            esci
        
        # Open the image and transform it
        if "BAD" in imgname: # exclude bad cases, poor quality images, or mislabelled cases
            # logging.info(f"GIANLU: CXR datasets - getitem(): trovato BAD in imgname {imgname} (index {index}), raise value error")
            pass    
        
        
        try: 
            img = PIL.Image.open(imgname) #potentially, it is a 16-bit PNG image (e.g., in PadChest dataset)
            #This means that in order to convert it to an RGB with PIL.Image, we need first to normalize it back within 0-255, otherwise it would saturate the channels and give errors in PIL
            if self.adjustContrast:
                img = adjustContrast(img) #returns a PIL Image, i.e., Image.fromarray(img_array.astype('uint8'))
            
        except PIL.UnidentifiedImageError:
            print(f"GIANLU PIL Error in file {imgname}")
            # logging.info(f"GIANLU PIL Error in file {imgname}")
            img = PIL.Image.fromarray(np.zeros((64,64)).astype('uint8'))#.convert("RGB") #TODO just to continue the debugging, then remove this line
            pass
        except OSError:
            print(f"GIANLU OSError in file {imgname}") #such as OSError: image file is truncated
            # logging.info(f"GIANLU OSError in file {imgname}")
            img = PIL.Image.fromarray(np.zeros((64,64)).astype('uint8'))#.convert("RGB") #TODO just to continue the debugging, then remove this line
            pass





        # img = img.convert("RGB") #8-bit RGB PIL image #TODO è ancora veramente necessaria questa? ora che ho modificato resnet.py con un input channel...
        img=img.convert("L")
        



        if self.input_transform:
            img = self.input_transform(img)        
            if img.min() == img.max(): #most likely both are 255 (completely white image) or 0 (black)
                print()
                # logging.info(f"GIANLU: Attention, the img (array and then tensor) at path {imgname} contained x_max == x_min: {img.min()}")
        
        
        
        # logging.info(f"img.size(): {img.size()}") #TODO May 2024
        
        
        
        #
        # onehotencoding = [int(self.df.loc[index, elem]) for elem in cate] 
        onehotencoding = []
        for elem in cate:
            if elem in self.df.columns:
                onehotencoding.append(int(self.df.loc[index, elem]))
            else:
                # logging.info(f"Column '{elem}' not found in df.columns: {self.df.columns}")
                esci
        label = np.array(onehotencoding).astype(np.float64)

        if self.use_tSNE_selfreportedrace:
            return img, label, label_dataset, srrace
        else:
            return img, label, label_dataset

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

    def __len__(self):
        return len(self.df) ##TODO

    def getHead(self):
        return self.df.head()



######
class CXRDataset_OOD(data.Dataset): 
    #builder class for the CXR dataset to be used as OOD dataset to asses domain generalization
    
    def __init__(self, data_path, input_transform=None, train=True):
        
        self.data_path=data_path
        self.isTrain=train

          
        # elif self.data_path.endswith("dataset_padchest"):
        #     dataset_name="dataset_padchest"
        #     if train:
        #         self.df = pd.read_csv(os.path.join(data_path,'./csv_files_new/train80val20_train.csv'))
                
        #     else:
        #         self.df = pd.read_csv(os.path.join(data_path,'./csv_files_new/train80val20_val.csv'))

        #     self.images_path = data_path # here, there are the 52 subfolders, each containing around <3000 PNG images: thanks to the CSV file above, we can access the correct subfolder to locate the desired image filename to load. 
        
        if self.data_path.endswith("dataset_mimicCXR_JPG"): # TODO devo ancora finire di scaricare i dicom, da cui selezionare i JPG frontali e prenderne i metadati per stratificare e creare i csv split.
            if train:
                df_tmp = pd.read_csv(os.path.join(data_path,'physionet.org/files/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-chexpert_gc_extended_jpg.csv'))  #TODO
                df_tmp.rename(columns={"Pleural Effusion": "Effusion"}, inplace=True)
                self.df = df_tmp
                print(self.df.columns)
            else:
                esci
            
            self.images_path = os.path.join(data_path,'physionet.org/files/mimic-cxr-jpg/2.0.0/files')
            # self.df.insert(loc=0, column='dataset_name', value = ['dataset_mimicCXR_JPG' for _ in range(len(self.df))])
           
        # print(f"Sto preparando la init del {dataset_name}, di lunghezza {len(self.df)} ")# e colonne: {self.df.columns}\n E di cui voglio selezionare solo le categorie: {cate}, più la 0_IMAGEINDEXORPATH") 
        self.df = self.df.loc[:, cate + ['image_path']] #tengo solo le colonne delle findings che voglio
        
        self.input_transform = input_transform

    def __getitem__(self, index):                
        

        # p10\p10000032\s50414267\02aa804e-bde0afdd-112c0b34-7bc16630-4e384014.jpg
        tmp = self.df.at[index,'image_path']
        name_superfolder, name_patientfolder, name_studyfolder, name_imagepath = tmp.split("\\")
        imgname = os.path.join(self.images_path, name_superfolder, name_patientfolder, name_studyfolder, name_imagepath)
      
        
        # Open the image and transform it
        if "BAD" in imgname: # exclude bad cases, poor quality images, or mislabelled cases
            # logging.info(f"GIANLU: CXR datasets - getitem(): trovato BAD in imgname {imgname} (index {index}), raise value error")
            pass    
        
        
        try: 
            img = PIL.Image.open(imgname) #potentially, it is a 16-bit PNG image (e.g., in PadChest dataset)
            #This means that in order to convert it to an RGB with PIL.Image, we need first to normalize it back within 0-255, otherwise it would saturate the channels and give errors in PIL
            img = adjustContrast(img) #returns a PIL Image, i.e., Image.fromarray(img_array.astype('uint8'))
            
        except PIL.UnidentifiedImageError:
            print(f"GIANLU PIL Error in file {imgname}")
            # logging.info(f"GIANLU PIL Error in file {imgname}")
            img = PIL.Image.fromarray(np.zeros((64,64)).astype('uint8'))#.convert("RGB") #TODO just to continue the debugging, then remove this line
            pass
        except OSError:
            print(f"GIANLU OSError in file {imgname}") #such as OSError: image file is truncated
            # logging.info(f"GIANLU OSError in file {imgname}")
            img = PIL.Image.fromarray(np.zeros((64,64)).astype('uint8'))#.convert("RGB") #TODO just to continue the debugging, then remove this line
            pass





        # img = img.convert("RGB") #8-bit RGB PIL image #TODO è ancora veramente necessaria questa? ora che ho modificato resnet.py con un input channel...
        img=img.convert("L")
        



        if self.input_transform:
            img = self.input_transform(img)        
            if img.min() == img.max(): #most likely both are 255 (completely white image) or 0 (black)
                print()
                # logging.info(f"GIANLU: Attention, the img (array and then tensor) at path {imgname} contained x_max == x_min: {img.min()}")
        
        
        
        # logging.info(f"img.size(): {img.size()}") #TODO May 2024
        
        
        
        #
        # onehotencoding = [int(self.df.loc[index, elem]) for elem in cate] 
        onehotencoding = []
        for elem in cate:
            if elem in self.df.columns:
                onehotencoding.append(int(self.df.loc[index, elem]))
            else:
                # logging.info(f"Column '{elem}' not found in df.columns: {self.df.columns}")
                esci
        label = np.array(onehotencoding).astype(np.float64)

        return img, label

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

    def __len__(self):
        return len(self.df) ##TODO

    def getHead(self):
        return self.df.head()
