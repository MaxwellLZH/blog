import os
import cv2
import time
import numpy as np

from model import whoisthis
from crop_face import crop_face, get_faces
from text_message import send_message
from change_screen import show_image


class Video(object):
    """class for capturing images of yourself, and for later getting the real-time images"""

    def __init__(self, name='Me', mode='r', height=243, width=320):
        self.name = name
        self.directory = "./Images/{}/".format(name)
        try:
            os.makedirs(self.directory)
        except FileExistsError:
            pass
        self.mode = mode
        self.cap = cv2.VideoCapture(0)
        self.cap_height = height
        self.cap_width  = width

    def open_cap(self):
        if self.cap.isOpened() is False:
            self.cap.open()

        self.cap.set(3, self.cap_height)
        self.cap.set(4, self.cap_width)

    def save_image(self, n=500):
        """Start recording and save as pictures after the first few seconds"""
        if self.mode == 'r':
            self.empty_folder()

        self.open_cap()

        n_pics = self.n_pics()
        count = 0
        cap = self.cap
        while count < n:
            ret, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            cv2.imwrite(self.directory +"me_{}.jpg".format(n_pics+count), gray)
            time.sleep(0.1)
            count += 1

        cap.release()
        cv2.destroyAllWindows()
        print("Total number of pictures of {}: {}.".format(self.name, self.n_pics()))

    def cctv(self, interval=0.5, gray_scale=True):
        """turn on cctv, return frame by interval"""
        self.open_cap()
        cap = self.cap

        while True:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                raise StopIteration

            time.sleep(interval)
            ret, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            yield gray if gray_scale else frame

    def monitor(self, interval=0.5, time_span=10, threshold=1e-10, gray_scale=True):
        """ During the past `time_span` seconds if there're less than
            `threshold` * `time_span` pictures that's me, do something
        """
        lookback = np.ones([int(time_span / interval)])

        for image in self.cctv(interval, gray_scale):
            faces = get_faces(image)
            if faces is None:
                print('No face detected in camera.')
                continue

            # get the face area
            x, y, w, h = faces[0]
            face = crop_face(image, x, w, y, h)
            # resizing image
            face = cv2.resize(face, (64, 64))

            lookback = np.roll(lookback, shift=-1)
            identity = whoisthis(face)
            print(identity)
            lookback[-1] = identity
            if np.mean(lookback) < threshold:
                show_image()
                send_message(r'[Alert] Someone is look at your screen!')
                raise ValueError('This is not me!')

    def n_pics(self):
        return len(os.listdir(self.directory))

    def empty_folder(self):
        """ Empty the directory first when the mode is 'r' """
        for file in os.listdir(self.directory):
            os.remove(os.path.join(self.directory, file))


video = Video(name='notMe')
video.monitor(threshold=1)