import torch
import torchvision
from tqdm import tqdm
import os
import sys
import cv2
import numpy as np
from pathlib import Path
from argparse import ArgumentParser

# Add directories to the Python path so that we can import custom modules
sys.path.append("backend/models")
sys.path.append("backend/tools")

from track_net import TrackNet
from utils import extract_numbers, write_json, read_json
import logging
import traceback

# Setup paths similar to how YOLOv5 does
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # This is the root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # make it relative

def ball_detect(video_path, result_path):
    imgsz = [288, 512]  # expected input image size for the model
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    # extract original name and starting frame from the filename
    orivi_name, start_frame = extract_numbers(video_name)

    # load court information (not directly used in ball detection here but may be useful for context)
    cd_save_dir = os.path.join(f"{result_path}/courts", f"court_kp")
    cd_json_path = os.path.join(cd_save_dir, f"{orivi_name}.json")
    court = read_json(cd_json_path)['court_info']

    # directory where detection results will be saved
    d_save_dir = os.path.join(result_path, f"loca_info/{orivi_name}")
    f_source = str(video_path)
    if not os.path.exists(d_save_dir):
        os.makedirs(d_save_dir)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # initialize and load the pre-trained TrackNet model
    model = TrackNet().to(device)
    model.load_state_dict(torch.load("backend/models/weights/ball_track.pt"))
    model.eval()

    # open the video file using OpenCV
    vid_cap = cv2.VideoCapture(f_source)
    video_end = False
    video_len = int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = vid_cap.get(cv2.CAP_PROP_FPS)
    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    count = 0
    with tqdm(total=video_len) as pbar:
        # loop over the video until all frames are processed
        while vid_cap.isOpened():
            imgs = []
            # read 3 frames at a time
            for _ in range(3):
                ret, img = vid_cap.read()
                if not ret:
                    video_end = True
                    break
                imgs.append(img)
            # if there are no more frames, exit the loop
            if video_end:
                break

            imgs_torch = []
            # preprocess each of the 3 frames
            for img in imgs:
                # convert from BGR to RGB color space
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # convert image to tensor and resize it to expected input dimensions
                img_torch = torchvision.transforms.ToTensor()(img).to(device)
                img_torch = torchvision.transforms.functional.resize(img_torch, imgsz, antialias=True)
                imgs_torch.append(img_torch)

            # concatenate the 3 frames along the channel dimension
            # each frame has 3 channels, so 3 frames give us a 9-channel input
            imgs_torch = torch.cat(imgs_torch, dim=0).unsqueeze(0)

            # run inference using the model
            preds = model(imgs_torch)
            preds = preds[0].detach().cpu().numpy()

            # threshold predictions to create binary heatmaps
            y_preds = preds > 0.6
            y_preds = y_preds.astype('float32')
            y_preds = y_preds * 255
            y_preds = y_preds.astype('uint8')

            # process predictions for each frame
            for i in range(3):
                if np.amax(y_preds[i]) <= 0:
                    # if there is no detected ball, mark it as not visible
                    ball_dict = {
                        f"{count + start_frame}": {
                            "visible": 0,
                            "x": 0,
                            "y": 0,
                        }
                    }
                    write_json(ball_dict, video_name, f"{d_save_dir}")
                else:
                    # resize the prediction heatmap to match original video dimensions
                    pred_img = cv2.resize(y_preds[i], (w, h), interpolation=cv2.INTER_AREA)
                    # find contours in the prediction heatmap
                    (cnts, _) = cv2.findContours(pred_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    rects = [cv2.boundingRect(ctr) for ctr in cnts]
                    max_area_idx = 0
                    max_area = rects[max_area_idx][2] * rects[max_area_idx][3]
                    # select the contour with the largest area (assumed to be the ball)
                    for ii in range(len(rects)):
                        area = rects[ii][2] * rects[ii][3]
                        if area > max_area:
                            max_area_idx = ii
                            max_area = area
                    target = rects[max_area_idx]
                    (cx_pred, cy_pred) = (int((target[0] + target[2] / 2)),
                                          int((target[1] + target[3] / 2)))
                    ball_dict = {
                        f"{count + start_frame}": {
                            "visible": 1,
                            "x": cx_pred,
                            "y": cy_pred,
                        }
                    }
                    write_json(ball_dict, video_name, f"{d_save_dir}")

                count += 1
                pbar.update(1)

        # if there are any remaining frames, they are marked as not visible
        while count < video_len:
            ball_dict = {
                f"{count + start_frame}": {
                    "visible": 0,
                    "x": 0,
                    "y": 0,
                }
            }
            write_json(ball_dict, video_name, f"{d_save_dir}")
            count += 1
            pbar.update(1)

    vid_cap.release()
