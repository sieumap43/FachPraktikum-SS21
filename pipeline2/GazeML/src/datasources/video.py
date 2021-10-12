"""Video (file) data source for gaze estimation."""
import os
import time

import cv2 as cv
from numpy.core.numerictypes import ScalarType

from .frames import FramesSource


class Video(FramesSource):
    """Video frame grabbing and preprocessing."""

    def __init__(self, video_path, scale=1, **kwargs):
        """Create queues and threads to read and preprocess data."""
        self._short_name = 'Video'

        assert os.path.isfile(video_path)
        self._video_path = video_path
        self._capture = cv.VideoCapture(video_path)     
        self._capture.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
        self._capture.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
        self._scale = scale
        # Call parent class constructor
        super().__init__(staging=False, **kwargs)

    def frame_generator(self):
        """Read frame from webcam."""
        last_frame = None
        while True:
            ret, frame = self._capture.read()
            if ret:
                yield frame
                last_frame = frame
            else:
                yield last_frame
                break

    def frame_read_job(self):
        """Read frame from video (without skipping)."""
        generate_frame = self.frame_generator()
        while True:
            before_frame_read = time.time()
            try:
                bgr = next(generate_frame)
                if self._scale != 1:
                    bgr = cv.resize(bgr, (0, 0), fx=self._scale, fy=self._scale, interpolation = cv.INTER_CUBIC)
            except StopIteration:
                break
            if bgr is not None:
                after_frame_read = time.time()
                with self._read_mutex:
                    self._frame_read_queue.put((before_frame_read, bgr, after_frame_read))

        print('Video "%s" closed.' % self._video_path)
        self._open = False
