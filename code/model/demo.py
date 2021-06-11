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
    parser_.add_argument("--epochs_begin", default=10, type=int)
    parser_.add_argument("--epochs_uniform", default=10, type=int)
    parser_.add_argument("--loss_threshold", default=3000, type=float)
    args_ = parser_.parse_args()
    return vars(args_)


def combined_loss(output, target):
    loss_ = torch.mean((output[0]-target[0])**2) + torch.mean((output[1]-target[1])**2)
    return loss_


def main():
    args = parse_args()

    dataset = datasetGetter(model_dirs=[str(k) for k in range(1, 8)], \
                            base_path=args["base_path"], mode=args["mode"])
    dataloader = DataLoader(dataset, batch_size=1)
    featExtr = featureExtractor().cuda()
    head = dataMatcher().cuda()

    featExtr_optimizer = torch.optim.Adam(featExtr.parameters(), lr=0.001)
    head_optimizer = torch.optim.Adam(head.parameters(), lr=0.001)

    for epo in range(args["epochs_begin"]):
        for i0, (img1, img2, vec) in enumerate(dataloader):
            featExtr.zero_grad()
            embeds = (featExtr(img1).detach(), featExtr(img2).detach())

            embeds = torch.cat((embeds[0], embeds[1]), axis=1)

            ans = head(embeds)
            loss = combined_loss(ans, [vec[:,:3], vec[:,3:]])
            loss_check = loss.clone()
            if loss_check.detach().cpu().numpy() < args["loss_threshold"]:
                loss.backward()
                head_optimizer.step()
            #print(ans)
            #print(vec)
            #print(combined_loss(ans, [vec[:,:3], vec[:,3:]]))
            """
            if loss.detach().cpu().numpy() > 2000:
                imgo = np.transpose(img1.cpu().numpy()[0], [1,2,0])
                imgt = np.transpose(img2.cpu().numpy()[0], [1,2,0])
                imgo /= np.max(imgo)
                imgt /= np.max(imgt)
                cv2.imwrite("imgo.png", imgo*255)
                cv2.imwrite("imgt.png", imgt*255)
                break
            """
            print(epo, i0, loss.detach().cpu().numpy())
            print(ans)
            print(vec)
        break

if __name__ == "__main__":
    main()
