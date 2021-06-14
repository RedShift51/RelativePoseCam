import os
import numpy as np
import matplotlib.pyplot as plt
import argparse

import cv2

import torch
from datasets import *
from network import *

def parse_args():
    parser_ = argparse.ArgumentParser()
    parser_.add_argument("--base_path", default="/storage", type=str)
    parser_.add_argument("--mode", default="train", type=str)
    parser_.add_argument("--epochs_begin", default=5, type=int)
    parser_.add_argument("--epochs_uniform", default=10, type=int)
    parser_.add_argument("--loss_threshold", default=3000, type=float)
    args_ = parser_.parse_args()
    return vars(args_)


#def combined_loss(output, target):
#    loss_ = nn.L1Loss((output[0]-target[0])**2) + torch.mean((output[1]-target[1])**2)
#    #torch.mean((output[0]-target[0])**2) + torch.mean((output[1]-target[1])**2)
#    return loss_


def main():
    args = parse_args()

    dataset_train = datasetGetter(model_dirs=[str(k) for k in range(1, 8)], \
                            base_path=args["base_path"], mode=args["mode"])
    dataloader_train = DataLoader(dataset_train, batch_size=1)
    dataset_eval = datasetGetter(model_dirs=[str(k) for k in range(1, 8)], \
                            base_path=args["base_path"], mode="eval")
    dataloader_eval = DataLoader(dataset_eval, batch_size=1)

    featExtr = featureExtractor().cuda()
    head = dataMatcher().cuda()

    featExtr_optimizer = torch.optim.Adam(featExtr.parameters(), lr=0.001)
    head_optimizer = torch.optim.Adam(head.parameters(), lr=0.001)
    combined_loss = nn.L1Loss()

    for epo in range(args["epochs_begin"]):
        # train stage
        for i0, (img1, img2, vec) in enumerate(dataloader_train):
            featExtr.zero_grad()
            embeds = (featExtr(img1).detach(), featExtr(img2).detach())

            embeds = torch.cat((embeds[0], embeds[1]), axis=1)

            ans = head(embeds)
            loss = combined_loss(ans[0], vec[:,:3]) + combined_loss(ans[1], vec[:,3:])
            # angle
            loss_angle = combined_loss(ans[0], vec[:,:3])
            # shift
            loss_shift = combined_loss(ans[1], vec[:,3:])
            loss = loss_angle + loss_shift
 
            loss_check = loss.clone()
            if loss_check.detach().cpu().numpy() < args["loss_threshold"]:
                loss.backward()
                head_optimizer.step()
            print(epo, i0, loss_angle.detach().cpu().numpy(), loss_shift.detach().cpu().numpy())

        # eval stage
        data

    for epo in range(args["epochs_uniform"]):
        for i0, (img1, img2, vec) in enumerate(dataloader_train):
            featExtr.zero_grad()
            embeds = (featExtr(img1), featExtr(img2))

            embeds = torch.cat((embeds[0], embeds[1]), axis=1)

            ans = head(embeds)
            # angle
            loss_angle = combined_loss(ans[0], vec[:,:3])
            # shift
            loss_shift = combined_loss(ans[1], vec[:,3:])
            loss = loss_angle + loss_shift
            loss_check = loss.clone()
            if loss_check.detach().cpu().numpy() < args["loss_threshold"]:
                loss.backward()
                head_optimizer.step()
            print(epo+args["epochs_begin"], i0, loss_angle.detach().cpu().numpy(), 
                        loss_shift.detach().cpu().numpy())


if __name__ == "__main__":
    main()
