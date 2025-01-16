import torch.utils.data as data
from utils import tools
import os.path as osp
import os
import cv2
import numpy as np
import torch

class COCO_128Lables(object):
    def __init__(self, ind_to_class=None):
        self.ind_to_class = ind_to_class
        self.num_classes = len(ind_to_class)
    def get_target(self, path_target, height_img, width_img):
        results = []

        with open(path_target, 'r') as file:
            lines = file.readlines()

        for line in lines:
            # Check: label_ind center_x center_y width height
            parts = line.strip().split()
            if len(parts) != 5:
                raise ValueError(f"Format -label_ind center_x center_y width height-: {line.strip()}")

            label_ind = int(parts[0])
            center_x = float(parts[1])
            center_y = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])

            # Convert [xmin, ymin, xmax, ymax, label_ind]
            xmin = float(center_x - width / 2) / width_img 
            ymin = float(center_y - height / 2) / height_img
            xmax = float(center_x + width / 2) / width_img 
            ymax = float(center_y + height / 2) / height_img
        
            results.append([xmin, ymin, xmax, ymax, label_ind])

        return results # [[xmin, ymin, xmax, ymax, label_ind], ... ]

    def decoder_target(self, index):
        return self.ind_to_class[index]

class COCO_128Detection(data.Dataset):
    def __init__(self, path_yaml,
                 transform=None, name="Dataset coco128-format"):

        self.name = name
        self.path_yaml = path_yaml
        info_yaml = tools.load_yaml_to_dict(path_yaml)

        self.transform = transform
        self.target_transform = COCO_128Lables(info_yaml['names'].items())
        self.num_classes = self.target_transform.num_classes    
    
        imgs_path = osp.join(info_yaml['path'], info_yaml['train'])
        list_name_imgs = [osp.splitext(item)[0] for item in os.listdir(imgs_path)]
        self.dict_imgs_path = {osp.splitext(item)[0]: osp.join(imgs_path, item) for item in os.listdir(imgs_path)}
        
        labels_path = osp.join(osp.join(info_yaml['path'], 'labels'), osp.basename(osp.normpath(imgs_path)))
        list_name_labels = [osp.splitext(item)[0] for item in os.listdir(labels_path)]
        self.dict_labels_path = {osp.splitext(item)[0]: osp.join(labels_path, item) for item in os.listdir(labels_path)}
        
        self.ids = list(set(list_name_imgs) & set(list_name_labels))

    def __getitem__(self, index):
        im, gt, h, w = self.pull_item(index)
        return im, gt

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        '''
        Return: 
            torch.from_numpy(img), target, height, width
        '''
        name_id = self.ids[index]
        img_path = self.dict_imgs_path[name_id]
        target_path = self.dict_labels_path[name_id]

        img = cv2.imread(img_path)
        height, width, channels = img.shape

        target = self.target_transform.get_target(target_path, height, width)
        
        if self.transform is not None:
            target = np.array(target)
            img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
            img = img[:, :, (2, 1, 0)] # to rgb
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        return torch.from_numpy(img).permute(2, 0, 1), target, height, width 
        
    def pull_image(self, index):
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            PIL img
        '''
        name_id = self.ids[index]
        img_path = self.dict_imgs_path[name_id]
        return cv2.imread(img_path, cv2.IMREAD_COLOR)

    def pull_anno(self, index):
        '''Returns the original annotation of image at index

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        '''

        name_id = self.ids[index]
        img_path = self.dict_imgs_path[name_id]
        target_path = self.dict_labels_path[name_id]
        img = cv2.imread(img_path)
        height, width, channels = img.shape
        target = self.target_transform.get_target(target_path, height, width)

        return name_id, target

    def pull_tensor(self, index):
        '''Returns the original image at an index in tensor form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            tensorized version of img, squeezed
        '''
        return torch.Tensor(self.pull_image(index)).unsqueeze_(0)