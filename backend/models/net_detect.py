import torch
import torchvision
import numpy as np
import copy
import cv2
from PIL import Image
from torchvision.transforms import transforms
from torchvision.transforms import functional as F
import os

import sys
sys.path.append("backend/tools")
from utils import read_json # ignore the squiggle here


class NetDetect(object):
    """
    Contains methods for detecting the net using Keypoint RCNNs
    """
    def __init__(self):
        # set device on which to load the RCNN
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.normal_net_info = None          # expected info abt the net
        self.got_info = False                # NOTE: see if used

        self.mse = None   # Mean Standard Error
        # set up the RCNN
        self.setup_RCNN()
    
    def setup_RCNN(self):
        # load pre-trained kpRCNN model
        self.__net_kprCNN = torch.load("backend/models/weights/new_kpRCNN.pth", map_location=torch.device(self.device))
        # set device and set to evaluate mode
        self.__net_kprCNN.to(self.device).eval()
    
# reset/deletion methods:
    def reset(self):
        """ Reset the state """
        self.got_info = False
        self.normal_net_info = None
        # Reset self.mse?
    
    def del_RCNN(self):
        """ Delete the RCNN model """
        del self.__net_kpRCNN

# OpenCV video processing
    def pre_process(self, video_path, reference_path=None):
        """
        Pre-process video from video_path.
        Identifies net using reference_path, if exists; otherwise, loops through video frames to identify keypoints
        Return type: int
            if the reference_path is provided, returns the frame number from the reference json on success
            otherwise, returns frame number of the last successfully processed frame on success
        """
        video = cv2.VideoCapture(video_path)        # open video file
        fps = video.get(cv2.CAP_PROP_FPS)           # get video fps
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))     # get number of frames in video

        # CASE 1: If reference exists
        if reference_path is not None:
            reference_data = read_json(reference_path)      # reference_data is a dictionary

            # get expected net info from the reference file
            self.normal_net_info = reference_data.get('net_info')
            # error checking
            if self.normal_net_info is None:
                video.release()
                print("Error: no 'net_info' property in reference\nTotal frames: " + total_frames)
                sys.exit(1)
            self.__multi_points = self.__partition(self.normal_net_info).tolist()

            # get the frame number (to be returned) from the reference file
            frame_number = reference_data.get('frame')
            if frame_number is None:
                # see if first_rally_frame exists instead
                frame_number = reference_data.get('first_rally_frame')
                if frame_number is None:
                    video.release()
                    print("Error: frame number not found in reference data")
                    sys.exit(1)
    
            print(f"video is preprocessed based on {reference_path} for net; frame number is {frame_number}")
            video.release()
            return frame_number
        
        # CASE 2: reference doesn't exist
        consecutive_count = 0          # number of consecutive frames in which net detected
        net_info_list = []
        # number of frames to skip per iteration
        # also doubles as # of consecutive detections needed for validation
        skip_frames = max(int(fps) // 5, 5)     
        
        # loop through video frames and store valid detected kp in net_info_list
        while True:
            current_frame = int(video.get(cv2.CAP_PROP_POS_FRAMES))     # set current frame
            ret, frame = video.read()       # ret = T/F, frame = read frame
            
            print(f"video is pre-processing for net; current frame is {current_frame}")
            
            # nothing read in video.read()
            if not ret:
                video.release()
                # return frame number, adjusting for the skipped frames NOTE:???
                return max(0, current_frame - 2 * skip_frames)

            # don't need more frames
            # (a certain number of frames need to have consistent keypoint detections
            # before we can conclude that the results are valid)
            if consecutive_count >= skip_frames:
                # check for invalid keypoints:
                for net_info in net_info_list:
                    if not self.__check_net(net_info):
                        # reset variables
                        self.normal_net_info = None     # can comment out?
                        net_info_list = []
                        consecutive_count = 0
                        print("Net detection is incorrect!")
                        break
                # set detected keypoints to be the expected ones
                self.normal_net_info = net_info_list[skip_frames // 2]  # NOTE: ?
                return max(0, current_frame - 2 * skip_frames)
            
            # regular loop iteration
            net_info, have_net = self.get_net_info(frame)
            if have_net:
                consecutive_count += 1
                net_info_list.append(net_info)      # add to keypoints list
            else:   # net not detected
                # pre-emptive error checking for video.set()?
                if current_frame + skip_frames >= total_frames:
                    print("Failed to pre-process! Please to check the video or program!")
                    exit(1)
                # update current frame (skip number of frames = skip_frames)
                video.set(cv2.CAP_PROP_POS_FRAMES, current_frame + skip_frames)
                # reset consecutive variables
                consecutive_count = 0
                net_info_list = []

# Torch stuff
    def get_net_info(self, img):
        """
        Get net info from frame (img) 
        Returns list of detected/corrected keypoints, and a flag that indicates if detection was successful
        """
        self.__correct_points = None
        frame = img.copy()
        self.mse = None
        frame_h, frame_w, _ = frame.shape       # get frame height and width
        # convert frame to tensor
        frame = F.to_tensor(frame)
        frame = frame.unsqueeze(0)
        frame = frame.to(self.device)

        output = self.__net_kpRCNN(frame)       # run kpRCNN
        
        scores = output[0]['scores'].detach().cpu().numpy()     # get confidence scores
        # get indices of detections w/ high confidence score (> 0.7)
        high_scores_idxs = np.where(scores > 0.7)[0].tolist()
        # apply non-maximum suppression to remove redundant detections NOTE: idk lol
        post_nms_idxs = torchvision.ops.nms(
            output[0]['boxes'][high_scores_idxs],
            output[0]['scores'][high_scores_idxs], 0.3).cpu().numpy()

        # if no high-confidence keypoints were detected:
        if len(output[0]['keypoints'][high_scores_idxs][post_nms_idxs]) == 0:
            self.got_info = False
            return None, self.got_info

        # store detected keypoints
        keypoints = []
        for kps in output[0]['keypoints'][high_scores_idxs][
                post_nms_idxs].detach().cpu().numpy():
            keypoints.append([list(map(int, kp[:2])) for kp in kps])

        self.__true_net_points = copy.deepcopy(keypoints[0])

        # correct detected points
        self.__correct_points = self.__correction()

        # validate detected keypoints against normal_net_info (NOTE: is this always false?)
        if self.normal_net_info is not None:
            # compare detected keypoints w/ reference
            self.got_info = self.__check_net(self.__true_net_points)
            if not self.got_info:
                return None, self.got_info
    
        # if no normal_net_info, partition corrected keypoints to get more accurate court lines
        if self.normal_net_info is None:
            self.__multi_points = self.__partition(
                self.__correct_points).tolist()
        else:
            self.__multi_points = self.__partition(
                self.normal_net_info).tolist()

        self.got_info = True

        return self.__correct_points.tolist(), self.got_info
    
    def __check_net(self, net_info):
        """ compare net_info to normal_net_info (reference) """
        vec1 = np.array(self.normal_net_info)
        vec2 = np.array(net_info)
        self.mse = np.square(vec1 - vec2).mean()     # calculate mse
        if self.mse > 100:
            return False
        return True

# private Numpy calculation stuff
    def __correction(self):
        """ correct net keypoints to be consistent by taking averages """
        net_kp = np.array(self.__true_net_points)

        # get avg vertical position along left and right sides of court (NOTE: check which is which)
        up_y = int((np.round(net_kp[0][1] + net_kp[3][1])) / 2)
        down_y = int((np.round(net_kp[1][1] + net_kp[2][1]) / 2))
        # get avg horizontal position along top and bottom sides of court (NOTE: check which is which)
        up_x = int(np.round((net_kp[0][0] + net_kp[1][0]) / 2))
        down_x = int(np.round((net_kp[3][0] + net_kp[2][0]) / 2))

        # set corrected points
        net_kp[0][1] = up_y
        net_kp[3][1] = up_y

        net_kp[1][1] = down_y
        net_kp[2][1] = down_y

        net_kp[0][0] = up_x
        net_kp[1][0] = up_x

        net_kp[3][0] = down_x
        net_kp[2][0] = down_x

        return net_kp

    def __partition(self, net_crkp):
        """ parition input net keypoints by adding midpoints along top and bottom edges """
        net_kp = np.array(net_crkp)

        # top left and bottom left corners of net, probably
        p0 = net_kp[0]
        p1 = net_kp[3]

        # top right and bottom right corners, probably
        p4 = net_kp[1]
        p5 = net_kp[2]

        # get midpoints along top/bot edge of net
        p2 = np.array([p0[0], np.round((p4[1] + p0[1]) * (0.5))], dtype=int)
        p3 = np.array([p1[0], np.round((p5[1] + p1[1]) * (0.5))], dtype=int)

        kp = np.array([p0, p1, p2, p3, p4, p5], dtype=int)

        return kp

# Draw net using CV2
    def draw_net(self, image, mode="auto"):
        """ returns the net lines to draw (returns original input image on failure) """

        if self.normal_net_info is None and mode == "auto":
            # print("There is not net in the image! So you can't draw it.")
            return image
        elif mode == "frame_select":
            if self.__correct_points is None:
                return image
            # get keypoints used to draw the net
            self.__multi_points = self.__partition(
                self.__correct_points).tolist()

        image_copy = image.copy()
        c_edges = [[0, 1], [2, 3], [0, 4], [1, 5]]  # each pair represents an edge to draw; 3 vertical lines, 2 horz

        net_color_edge = (53, 195, 242)
        net_color_kps = (5, 135, 242)

        # draw the net:
        # draw edges
        for e in c_edges:
            cv2.line(image_copy, (int(self.__multi_points[e[0]][0]),
                                  int(self.__multi_points[e[0]][1])),
                     (int(self.__multi_points[e[1]][0]),
                      int(self.__multi_points[e[1]][1])),
                     net_color_edge,
                     2,
                     lineType=cv2.LINE_AA)
        # draw keypoints
        for kps in [self.__multi_points]:
            for kp in kps:
                cv2.circle(image_copy, tuple(kp), 1, net_color_kps, 5)

        return image_copy    
    
