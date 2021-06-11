import os
import numpy as np
import cv2

import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video")
    args = parser.parse_args()
    return vars(args)

def slice_vid(path_video_):
    base_path = path_video_[:path_video_.rfind("/")]
    new_folder = path_video_[path_video_.rfind("/")+1: path_video_.rfind(".")]

    if not os.path.exists(os.path.join(base_path, new_folder)):
        os.mkdir(os.path.join(base_path, new_folder))
    if not os.path.exists(os.path.join(base_path, new_folder, "images")):
        os.mkdir(os.path.join(base_path, new_folder, "images"))

    vidcap = cv2.VideoCapture(path_video_)
    success, image = vidcap.read()
    count = 0

    while success is True:
        if count % 3 == 0:
            cv2.imwrite(os.path.join(base_path, new_folder, "images", str(count)+".png"), image)
        success, image = vidcap.read()
        count += 1
        #print(count)
        #if count == 2000:
        #    break

    print("Frames", count)


def main():
    path_video = parse_args()["video"]

    slice_vid(path_video)


if __name__ == "__main__":
    main()
