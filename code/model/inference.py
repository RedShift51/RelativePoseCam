import os
import numpy as np
import matplotlib.pyplot as plt
import argparse

import cv2

import torch
from network import *
import torchvision.transforms as transforms

def parse_args():
    parser_ = argparse.ArgumentParser()
    parser_.add_argument("--path_weights", default="weights", type=str)
    parser_.add_argument("--head_epoch", default=4, type=int)
    parser_.add_argument("--feature_extr", default="not_load", type=str)
    parser_.add_argument("--img1", type=str)
    parser_.add_argument("--img2", type=str)
    args_ = parser_.parse_args()

    return vars(args_)


def main():
    args = parse_args()

    featExtr = featureExtractor()
    if args["feature_extr"] == "load":
        if len([k for k in os.listdir(args["path_weights"]) if 
                    k.find(str(args["head_epoch"])+"_featExtr")!=-1]) != 0:
            featExtr.load_state_dict(torch.load(
                os.path.join(args["path_weights"], str(args["head_epoch"]) + "_featExtr.pth")))
    featExtr = featExtr.cuda() 

    head = dataMatcher()
    head.load_state_dict(torch.load(
            os.path.join(args["path_weights"], str(args["head_epoch"]) + "_head.pth")))
    head = head.cuda()

    norm_ = transforms.Compose([transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], \
                                std=[0.229, 0.224, 0.225])])
    img1 = cv2.cvtColor(cv2.imread(args["img1"]), cv2.COLOR_BGR2RGB).astype(float)/255.
    img2 = cv2.cvtColor(cv2.imread(args["img2"]), cv2.COLOR_BGR2RGB).astype(float)/255.
    img1 = norm_(img1).unsqueeze(0).float().cuda()
    img2 = norm_(img2).unsqueeze(0).float().cuda()

    embeds = (featExtr(img1).detach(), featExtr(img2).detach())
    embeds = torch.cat((embeds[0], embeds[1]), axis=1)
    ans = head(embeds)
    print(ans)

if __name__=="__main__":
    main()
