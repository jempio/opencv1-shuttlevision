import cv2
import os

class VideoClip(object):
    def __init__(self, video_name, fps, total_frames, frame_width, frame_height, save_path="./") -> None:
        self.video_name = video_name
        self.fps = fps
        self.total_frames = total_frames
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.save_path = save_path

        self.skip_frames = fps // 2  #int() ?
        self.start = 0
        self.end = 1
        self.frame_list = []
        self.no_court_cnt = 0   #???

    def add_frame(self, have_court, frame, frame_count):
        """
        Returns True if a video clip is made; False otherwise
        """
        # reached end of current clip
        if frame_count == self.total_frames - 1:    # why - 1?
            # if less than 0.5 sec worth of frames, reset
            if len(self.frame_list) < int(self.fps * 0.5):
                self.frame_list.clear()
                self.start = -1     # ??
                self.end = 0
                return False
            # else:
            self.end = frame_count
            self.frame_list.append(frame)
            self.__make_video()
            self.__setup()
            return True

        # add frame to frame_list
        if have_court:
            if self.start == -1:
                self.start = frame_count
            self.frame_list.append(frame)
            return False
        
        # no court detected in frame
        else:
            # if > 3 secs of valid frames
            if len(self.frame_list) > self.fps * 3:
                # if no court has been detected for a while, just make clip
                if self.no_court_cnt >= self.skip_frames:
                    self.end = frame_count
                    self.__make_video()
                    self.__setup()
                    return True
                # add court-less frame
                else:
                    self.frame_list.append(frame)
                    self.no_court_cnt += 1
                    return False


    def __setup(self):
        self.frame_list = []
        self.no_court_cnt = 0
    
    def __make_video(self):
        video_name = f"{self.video_name}_{self.start}-{self.end - 1}.mp4"
        full_path = os.path.join(self.save_path, video_name)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_video_format = (self.frame_width, self.frame_height)

        video_writer = cv2.VideoWriter(full_path, fourcc, self.fps,
                                       output_video_format)

        for frame in self.frame_list:
            video_writer.write(frame)

        video_writer.release()
    