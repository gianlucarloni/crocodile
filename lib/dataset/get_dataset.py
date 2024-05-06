import torchvision.transforms as transforms
from dataset.nihdataset import NIHDataset
from dataset.cxr_datasets import CXRDataset
# from utils.cutout import CutoutPIL_
from randaugment import RandAugment
import argparse

def parser_args():
    parser = argparse.ArgumentParser(description='Training')
    args = parser.parse_args()
    return args

def get_args():
    args = parser_args()
    return args

##aprile 2024 TODO
import PIL
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import numpy as np
import sys, os
from torch.utils.data import ConcatDataset
# ####################################################TODO
import logging
logging.basicConfig(level=logging.INFO)
# #####################################################TODO

# class AdjustContrast(object):
#     def __call__(self, img):
#         img_array = np.array(img)        

#         if np.isnan(img_array).any():            
#             np.nan_to_num(img_array,copy=False)
#             logging.info(f"ATTENTION: img_array contained NaN, replaced with np.nan_to_num")
#         x_min = img_array.min()
#         x_max = img_array.max()
#         # logging.info(f"x_min {x_min}, x_max {x_max}")

#         if x_max != x_min:            
#             img_array = 255.0 * ((img_array - x_min) / (x_max - x_min))
#         # else:
#         #     logging.info(f"ATTENTION: img_array contained x_max == x_min, most likely all zeros")           
        
#         return Image.fromarray(img_array.astype('uint8')) #, 'L')


def get_datasets(args):
    # if args.orid_norm:
    #     normalize = transforms.Normalize(mean=[0, 0, 0],
    #                                      std=[1, 1, 1])
    # else:
    #     print("GET_DATASET: applying no normalization (e.g., N(0;1)) to the data")
    #     normalize = transforms.Normalize(mean=[0, 0, 0],
    #                                      std=[1, 1, 1])

    # train_data_transform_list = [transforms.Resize((args.img_size, args.img_size)), #ORIGINALE DI LORO
    #                                         RandAugment(),
    #                                            transforms.ToTensor(),
    #                                            normalize]


    # if args.cutout:
    #     print("Using Cutout!!!")
    #     train_data_transform_list.insert(1, CutoutPIL_(n_holes=args.n_holes, length=args.length))
    #     train_data_transform = transforms.Compose(train_data_transform_list)

    #     test_data_transform = transforms.Compose([
    #                                         transforms.Resize((args.img_size, args.img_size)),
    #                                         transforms.ToTensor(),
    #                                         normalize])






    if not args.useCrocodile:
        if args.dataname == 'nih':
            dataset_dir = args.dataset_dir
            # nih_transform = transforms.Compose([
            #     transforms.Resize((args.img_size, args.img_size)),
            #     transforms.ToTensor(),
            #     # transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
            # ])
            # TODO aprile 2024 replaced the above nih_transform with the new one that adjusts the contrast and normalizes:
            nih_transform = transforms.Compose([
                transforms.Resize((args.img_size, args.img_size)),
                # AdjustContrast(),  # Custom contrast adjustment, april 2024    
                transforms.ToTensor(), #0...1 range
                # transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) #-1...1 range to speed up and stabilize the training process 
                transforms.Normalize((0.5, ), (0.5, )) #TODO one channel grayscale images instead
                
            ]) 
            

            train_dataset = NIHDataset(
                data_path = dataset_dir,
                input_transform = nih_transform,
                train=True
            )
            val_dataset = NIHDataset(
                data_path=dataset_dir,
                input_transform = nih_transform,
                train=False
            )
        else:
            raise NotImplementedError("Unknown dataname %s" % args.dataname)

    
    else: # Crocodile
    
        dataset_dir = args.dataset_dir #here, it is the parent folder containing the multiple dataset folders

        transform_ResizeAdjustNormalize = transforms.Compose([
            transforms.Resize((args.img_size, args.img_size)),
            # AdjustContrast(),  # Custom contrast adjustment, april 2024    
            transforms.ToTensor(), #0...1 range
            # transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) #-1...1 range to speed up and stabilize the training process 
            transforms.Normalize((0.5, ), (0.5, )) #TODO one channel grayscale images instead
        ]) 
        
        ## We will create several different datasets and then combine them to constitute the training set of images.
        # Actually, we could utilize a huge CSV file which contains all those training instances (each in its respective path)
        # And thus utilize a second big CSV file consisting of the validation instances from the same datasets.

        ## However, the different datasets might need different handling and preprocessing steps, so we keep modularity and load them separately

        # At inference/test/deployment time, we will utilize an external dataset (never before seen dataset) to assess domain generalization
        
        # Dataset NIH CXR 14
        train_dataset_nihCXR14 = CXRDataset(
            data_path = os.path.join(dataset_dir,'dataset_chestxray14'),
            input_transform = transform_ResizeAdjustNormalize,
            train=True
        )
        print(f"Creato train_dataset_nihCXR14")
        val_dataset_nihCXR14 = CXRDataset(
            data_path= os.path.join(dataset_dir,'dataset_chestxray14'),
            input_transform = transform_ResizeAdjustNormalize,
            train=False
        )
        print(f"Creato val_dataset_nihCXR14")

        # Dataset PadChest
        train_dataset_padchest = CXRDataset(
            data_path= os.path.join(dataset_dir,'dataset_padchest'),
            input_transform = transform_ResizeAdjustNormalize,
            train=True
        )
        print(f"Creato train_dataset_padchest")
        val_dataset_padchest = CXRDataset(
            data_path= os.path.join(dataset_dir,'dataset_padchest'),
            input_transform = transform_ResizeAdjustNormalize,
            train=False
        )
        print(f"Creato val_dataset_padchest")

        # Dataset CheXpert
        train_dataset_chexpert = CXRDataset(
            data_path= os.path.join(dataset_dir,'dataset_chexpert'),
            input_transform = transform_ResizeAdjustNormalize,
            train=True
        )
        print(f"Creato train_dataset_chexpert")
        val_dataset_chexpert = CXRDataset(
            data_path= os.path.join(dataset_dir,'dataset_chexpert'),
            input_transform = transform_ResizeAdjustNormalize,
            train=False
        )
        print(f"Creato val_dataset_chexpert")


        ## Dataset MIMIC-CXR-2.0
        # train_dataset_mimic = CXRDataset(
        #     data_path= os.path.join(dataset_dir,'dataset_mimicCXR_JPG'),
        #     input_transform = transform_ResizeAdjustNormalize,
        #     train=True
        # )
        # val_dataset_mimic = CXRDataset(
        #     data_path= os.path.join(dataset_dir,'dataset_mimicCXR_JPG'),
        #     input_transform = transform_ResizeAdjustNormalize,
        #     train=False
        # )

        # Dataset VinDr-CXR (Vietnam, physionet.org)
        # we might use it as the external dataset for deployment test
    
    # else:
    #     raise NotImplementedError("Unknown dataname %s" % args.dataname)



    # print(val_dataset_nihCXR14.getHead())
    # print(val_dataset_padchest.getHead())
    # print(val_dataset_chexpert.getHead())
    # Combine datasets
    train_dataset = ConcatDataset([train_dataset_nihCXR14, train_dataset_padchest, train_dataset_chexpert])#, train_dataset_mimic])
    val_dataset = ConcatDataset([val_dataset_nihCXR14, val_dataset_padchest, val_dataset_chexpert])#, val_dataset_mimic])

    print("len(train_dataset):", len(train_dataset)) 
    print("len(val_dataset):", len(val_dataset))

    # print(val_dataset.getHead())
    
    return train_dataset, val_dataset
