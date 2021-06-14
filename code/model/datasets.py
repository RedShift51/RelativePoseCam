import os
import numpy as np
import pickle as pkl
import cv2
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

import torch
from torch.utils.data import DataLoader
import torch.utils.data as data
import torchvision.transforms as transforms

class datasetGetter(data.Dataset):
    def __init__(self, base_path, model_dirs, mode="train"):
        self.norm_ = transforms.Compose([transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], \
                                std=[0.229, 0.224, 0.225])])
        self.unsq_norm_ = lambda x_: self.norm_(x_).float().cuda()
        self.model_dirs_ = model_dirs
        self.imgs_dirs_ = [os.path.join(base_path, k, "images") for k in self.model_dirs_]
        self.mode = mode
        np.random.seed(1)

        markups = [os.path.join(base_path, k, "model_txt", "images.txt") for k in self.model_dirs_]
        self.storage_ = []
        self.belong_ = {}
        self.belong_reverse_ = {}
        self.fnames_ = []
        self.count_elems_ = 0
        for f0, f_curr in enumerate(markups):
            lines = None
            with open(f_curr, "r") as f:
                lines = [s.split(" ") for s in f.readlines() if s.find(".png")!=-1]
                lines = sorted(lines, key=lambda x: int(x[-1][:x[-1].find(".")]))
            lines = {k[-1].strip(): np.array([float(l) for l in k[-9: -2]]) for k in lines}

            self.belong_reverse_[f0 + 1] = []
            for k in list(lines.keys()):
                self.belong_reverse_[f0 + 1].append(self.count_elems_)
                self.belong_[self.count_elems_] = f0 + 1
                self.storage_.append(np.concatenate([
                        Rotation.from_quat(lines[k][:4]).as_euler("xyz"), lines[k][4:]], 0))
                self.fnames_.append(os.path.join(base_path, str(f0+1), "images", k))
                self.count_elems_ += 1

            np.random.shuffle(self.belong_reverse_[f0 + 1])

        # splitting on train / validation: 85 / 15
        self.train_ids_, self.valid_ids_ = [], []
        for f0 in range(len(markups)):
            basic_idx = int(len(self.belong_reverse_[f0 + 1]) * 0.85)
            self.train_ids_ += self.belong_reverse_[f0 + 1][:basic_idx]
            self.valid_ids_ += self.belong_reverse_[f0 + 1][basic_idx:]


    def change_mode(self, new_mode="train"):
        self.mode = new_mode
        return True

    def __len__(self):
        if self.mode == "train":
            return 2 * (len(self.train_ids_) - 1)
        else:
            return 2 * (len(self.valid_ids_) - 1)
        #return 2 * (self.count_elems_ - 1)

    def __getitem__(self, idx):
        actual_idx = None
        reverse = None
        if self.mode == "train":
            actual_idx = np.random.randint(2 * (len(self.train_ids_) - 1))
            reverse = 0 if actual_idx < len(self.train_ids_) else 1
            actual_idx = actual_idx % len(self.train_ids_)
            if actual_idx == self.count_elems_-1:
                actual_idx -= 1
        else:
            actual_idx = np.random.randint(2 * (len(self.valid_ids_) - 1))
            reverse = 0 if actual_idx < len(self.valid_ids_) else 1
            actual_idx = actual_idx % len(self.valid_ids_)
            if actual_idx == self.count_elems_-1:
                actual_idx -= 1

        #actual_idx = np.random.randint()
        #actual_idx = idx % self.count_elems_
        second_idx = np.abs(actual_idx-1) if \
                self.belong_[np.abs(actual_idx-1)] == self.belong_[actual_idx] else \
                actual_idx + 1
        idx1 = min(actual_idx, second_idx)
        idx2 = max(actual_idx, second_idx)

        image1 = cv2.cvtColor(cv2.imread(self.fnames_[idx1]), cv2.COLOR_BGR2RGB).astype(float)
        image2 = cv2.cvtColor(cv2.imread(self.fnames_[idx2]), cv2.COLOR_BGR2RGB).astype(float)
        diff = self.storage_[idx2] - self.storage_[idx1]

        #reverse = 0 if actual_idx < self.count_elems_ else 1

        if reverse == 0:
            return self.unsq_norm_(image1), self.unsq_norm_(image2), \
                        torch.tensor(diff).float().cuda()
        else:
            return self.unsq_norm_(image2), self.unsq_norm_(image1), \
                    -torch.tensor(diff).float().cuda()

