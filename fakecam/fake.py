#!/usr/bin/env python3

import asyncio
import itertools
import signal
import sys
import traceback
from argparse import ArgumentParser
from functools import partial
from typing import Any, Dict

import aiohttp
import cv2
import numpy as np
import pyfakewebcam
import requests
import requests_unixsocket
import os
import fnmatch
import time
import threading


def findFile(pattern, path):
    for root, _, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                print("match: ", os.path.join(root, name))
                return os.path.join(root, name)
    return None

class RealCam:
    def __init__(self, src, frame_width, frame_height, frame_rate):
        self.cam = cv2.VideoCapture(src, cv2.CAP_V4L2)
        self.stopped = False
        self.frame = None
        self.lock = threading.Lock()
        self._set_prop(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
        self._set_prop(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
        self._set_prop(cv2.CAP_PROP_FPS, frame_rate)

    def _set_prop(self, prop, value):
        if self.cam.set(prop, value):
            if value == self.cam.get(prop):
                return True

        print("Cannot set camera property {} to {}, used value: {}".format(prop, value, self.cam.get(prop)))
        return False

    def get_frame_width(self):
        return int(self.cam.get(cv2.CAP_PROP_FRAME_WIDTH))

    def get_frame_height(self):
        return int(self.cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def get_frame_rate(self):
        return int(self.cam.get(cv2.CAP_PROP_FPS))

    def start(self):
        self.thread = threading.Thread(target=self.update)
        self.thread.start()
        return self

    def update(self):
        while not self.stopped:
            grabbed, frame = self.cam.read()
            if grabbed:
                with self.lock:
                    self.frame = frame.copy()

    def read(self):
        with self.lock:
            if self.frame is None:
               return None
            return self.frame.copy()

    def stop(self):
        self.stopped = True
        self.thread.join()

class FakeCam:
    def __init__(
        self,
        fps: int,
        width: int,
        height: int,
        use_foreground: bool,
        hologram: bool,
        tiling: bool,
        bodypix_url: str,
        socket: str,
        fg_pattern: str,
        bg_pattern: str,
        image_folder: str,
        fg_mask_pattern: str,
        webcam_path: str,
        v4l2loopback_path: str,
    ) -> None:
        self.use_foreground = use_foreground
        self.hologram = hologram
        self.tiling = tiling
        self.real_cam = RealCam(webcam_path, width, height, fps)
        # In case the real webcam does not support the requested mode.
        self.width = self.real_cam.get_frame_width()
        self.height = self.real_cam.get_frame_height()
        self.fake_cam = pyfakewebcam.FakeWebcam(v4l2loopback_path, self.width, self.height)
        self.foreground_mask = None
        self.bg_pattern = bg_pattern
        self.fg_pattern = fg_pattern
        self.image_folder = image_folder
        self.fg_mask_pattern = fg_mask_pattern
        self.inverted_foreground_mask = None
        self.session = requests.Session()
        self.chunks = 2
        self.results = [None] * (self.chunks * self.chunks)
        self.oldmask = [None] * self.chunks * self.chunks
        self.mask_smoothen_frames = 3
        for i in range(0, self.chunks * self.chunks):
            self.oldmask[i] =  np.zeros(((int)(self.height / self.chunks), (int)(self.width / self.chunks)))

        if bodypix_url.startswith('/'):
            print("Looks like you want to use a unix socket")
            # self.session = requests_unixsocket.Session()
            self.bodypix_url = "http+unix:/" + bodypix_url
            self.socket = bodypix_url
            requests_unixsocket.monkeypatch()
        else:
            self.bodypix_url = bodypix_url
            self.socket = ""
            # self.session = requests.Session()
        self.images: Dict[str, Any] = {}
        self.image_lock = threading.Lock()


    

    async def _get_mask(self, frame, l_green, u_green):
        def denoise(mask, val):
            thresh = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            cnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cnts = cnts[0] if len(cnts) == 2 else cnts[1]
            applymorphology = False
            for c in cnts:
                area = cv2.contourArea(c)
                if area < 1000:
                    applymorphology = True
                    cv2.drawContours(thresh, [c], -1, (val,val,val), -1)
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                       (20,20))
            return cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel,
                                                      iterations=1) if applymorphology else mask

        def sliding_average(old, new, n):
            return old * (n-1)/n + new/n

        def calculate_mask(frame, l_green, u_green, results, index):
            hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
            mask = cv2.inRange(hsv, l_green, u_green)

            # remove small noise in mask
            mask = denoise(mask, 255)
            mask = denoise(mask, 0)
            mask = cv2.blur(mask, (5, 5))

            sa =  sliding_average(self.oldmask[index], mask, self.mask_smoothen_frames)
            self.oldmask[index] = sa
            results[index] = sa.astype(float)/255

        def split_image(img):
            results = [None] * (self.chunks * self.chunks)
            if self.chunks > 1:
                sizeX = img.shape[1]
                sizeY = img.shape[0]
                nRows = self.chunks 
                mCols = self.chunks
                for i in range(0,nRows):
                    for j in range(0, mCols):
                        index = j + (self.chunks * i)
                        results[index] = img[i*int(sizeY/nRows):i*int(sizeY/nRows)
                                            + int(sizeY/nRows)
                                ,j*int(sizeX/mCols):j*int(sizeX/mCols) + int(sizeX/mCols)]
            else:
                results[0] = img
            return results

        def merge_image(imgs):
            if self.chunks > 1:
                image_shape = (imgs[0].shape[1], imgs[0].shape[0])
                montage_shape = (self.chunks, self.chunks)
                montage_image = np.zeros((image_shape[1] * montage_shape[1], image_shape[0] * montage_shape[0]))
                cursor_pos = [0, 0]
                for img in imgs:
                    montage_image[cursor_pos[1]:cursor_pos[1] + image_shape[1], cursor_pos[0]:cursor_pos[0] + image_shape[0]] = img
                    cursor_pos[0] += image_shape[0]  # increment cursor x position
                    if cursor_pos[0] >= montage_shape[0] * image_shape[0]:
                        cursor_pos[1] += image_shape[1]  # increment cursor y position
                        cursor_pos[0] = 0
                return montage_image
            else:
                return imgs[0]




        def process_image(img):
            # Dimensions of the image
            threads = [None] * (self.chunks * self.chunks)

            index = 0
            if self.chunks > 1:
                for i in split_image(img):
                    threads[index] = threading.Thread(target=calculate_mask,
                                                        args=(i,
                                                                l_green,
                                                                u_green,
                                                                self.results,
                                                                index))
                    threads[index].start()
                    index+=1

                for t in threads:
                    t.join()
            else:
                calculate_mask(img, l_green, u_green, self.results, 0)


            return merge_image(self.results) 
        
        new = process_image(frame)
        # sa =  sliding_average(mask, new, self.mask_smoothen_frames)
#        cv2.imshow('old',mask)
#        cv2.imshow('new',new)
#        cv2.imshow('frame',frame)
#        cv2.imshow('sa', sa)
#        cv2.waitKey(5) & 0xFF
        return new

    def load_images(self):
        print("acquiring lock")
        self.image_lock.acquire()
        print("lock acquired")
        try:
            self.background_image=findFile(self.bg_pattern, self.image_folder)
            self.foreground_image=findFile(self.fg_pattern, self.image_folder)
            self.foreground_mask_image=findFile(self.fg_mask_pattern, self.image_folder)
            self.images: Dict[str, Any] = {}
            print("background image to be loaded: ", self.background_image)
            print("foreground image to be loaded: ", self.foreground_image)
            background = cv2.imread(self.background_image)
            if background is not None:
                if not self.tiling:
                    background = cv2.resize(background, (self.width, self.height))
                else:
                    sizey, sizex = background.shape[0], background.shape[1]
                    if sizex > self.width and sizey > self.height:
                        background = cv2.resize(background, (self.width, self.height))
                    else:
                        repx = (self.width - 1) // sizex + 1
                        repy = (self.height - 1) // sizey + 1
                        background = np.tile(background,(repy, repx, 1))
                        background = background[0:self.height, 0:self.width]
                background = itertools.repeat(background)
            else:
                background_video = cv2.VideoCapture(self.background_image)
                self.bg_video_fps = background_video.get(cv2.CAP_PROP_FPS)
                # Initiate current fps to background video fps
                self.current_fps = self.bg_video_fps
                def read_frame():
                        ret, frame = background_video.read()
                        if not ret:
                            background_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            ret, frame = background_video.read()
                            assert ret, 'cannot read frame %r' % self.background_image
                        frame = cv2.resize(frame, (self.width, self.height))
                        return frame
                def next_frame():
                    while True:
                        self.bg_video_adv_rate = round(self.bg_video_fps/self.current_fps)
                        for i in range(self.bg_video_adv_rate):
                            frame = read_frame();
                        yield frame
                background = next_frame()

            self.images["background"] = background

            if self.use_foreground and self.foreground_image is not None:
                foreground = cv2.imread(self.foreground_image)
                self.images["foreground"] = cv2.resize(foreground,
                                                        (self.width, self.height))
                foreground_mask = cv2.imread(self.foreground_mask_image)
                foreground_mask = cv2.normalize(
                    foreground_mask, None, alpha=0, beta=1,
                    norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                foreground_mask = cv2.resize(foreground_mask,
                                                (self.width, self.height))
                self.images["foreground_mask"] = cv2.cvtColor(
                    foreground_mask, cv2.COLOR_BGR2GRAY)
                self.images["inverted_foreground_mask"] = 1 - self.images["foreground_mask"]
        finally:
            self.image_lock.release()
            print("lock released")

    async def mask_frame(self, frame):
        # fetch the mask with retries (the app needs to warmup and we're lazy)
        # e v e n t u a l l y c o n s i s t e n t
        l_green = np.array([90, 20, 80])
        u_green = np.array([103, 255, 250])
        new_mask = None
        while new_mask is None:
            try:
                new_mask = await self._get_mask(frame, l_green, u_green)
            except Exception as e:
                print(f"Mask request failed, retrying: {e}")
                traceback.print_exc()
        
        # composite the foreground and background
        self.image_lock.acquire()
        try:
            background = next(self.images["background"])
            for c in range(frame.shape[2]):
                frame[:, :, c] = frame[:, :, c] * (1- new_mask) + background[:, :, c] * (new_mask)

            if self.use_foreground and self.foreground_image is not None:
                for c in range(frame.shape[2]):
                    frame[:, :, c] = (
                        frame[:, :, c] * self.images["inverted_foreground_mask"]
                        + self.images["foreground"][:, :, c] * self.images["foreground_mask"]
                        )
        finally:
            self.image_lock.release()
        return frame

    def put_frame(self, frame):
        self.fake_cam.schedule_frame(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    def stop(self):
        self.real_cam.stop()

    async def run(self):
        self.load_images()
        self.real_cam.start()
        t0 = time.monotonic()
        print_fps_period = 1
        frame_count = 0
        while True:
            frame = self.real_cam.read()
            if frame is None:
                await asyncio.sleep(0.1)
                continue
            await self.mask_frame(frame)
            self.put_frame(frame)
            frame_count += 1
            curtime = time.monotonic()
            td = curtime - t0
            if td > print_fps_period:
                self.current_fps = frame_count / td
                print("FPS: {:6.2f}".format(self.current_fps), end="\r")
                frame_count = 0
                t0 = curtime

def parse_args():
    parser = ArgumentParser(description="Faking your webcam background under \
                            GNU/Linux. Please make sure your bodypix network \
                            is running. For more information, please refer to: \
                            https://github.com/fangfufu/Linux-Fake-Background-Webcam")
    parser.add_argument("-W", "--width", default=1280, type=int,
                        help="Set real webcam width")
    parser.add_argument("-H", "--height", default=720, type=int,
                        help="Set real webcam height")
    parser.add_argument("-F", "--fps", default=30, type=int,
                        help="Set real webcam FPS")
    parser.add_argument("-B", "--bodypix-url", default="http://127.0.0.1:9000",
                        help="Tensorflow BodyPix URL")
    parser.add_argument("-w", "--webcam-path", default="/dev/video0",
                        help="Set real webcam path")
    parser.add_argument("-v", "--v4l2loopback-path", default="/dev/video2",
                        help="V4l2loopback device path")
    parser.add_argument("-i", "--image-folder", default=".",
                        help="Folder which contains foreground and background images")
    parser.add_argument("-b", "--background-image", default="background.*",
                        help="Background image path, animated background is \
                        supported.")
    parser.add_argument("--tile-background", action="store_true",
                        help="Tile the background image")
    parser.add_argument("--no-foreground", action="store_true",
                        help="Disable foreground image")
    parser.add_argument("-f", "--foreground-image", default="foreground.*",
                        help="Foreground image path")
    parser.add_argument("-m", "--foreground-mask-image",
                        default="foreground-mask.*",
                        help="Foreground mask image path")
    parser.add_argument("--hologram", action="store_true",
                        help="Add a hologram effect")
    return parser.parse_args()


def sigint_handler(loop, cam, signal, frame):
    if not cam.image_lock.locked():
        print("Reloading background / foreground images")
        cam.load_images()
    else:
        print("Unable to acquire lock. Please try again to reload scenery")


def sigquit_handler(loop, cam, signal, frame):
    print("Killing fake cam process")
    cam.stop()
    sys.exit(0)


def main():
    args = parse_args()
    cam = FakeCam(
        fps=args.fps,
        width=args.width,
        height=args.height,
        use_foreground=not args.no_foreground,
        hologram=args.hologram,
        tiling=args.tile_background,
        bodypix_url=args.bodypix_url,
        socket="",
        image_folder=args.image_folder,
        fg_pattern=args.foreground_image,
        bg_pattern=args.background_image,
        fg_mask_pattern=args.foreground_mask_image,
        webcam_path=args.webcam_path,
        v4l2loopback_path=args.v4l2loopback_path)
    loop = asyncio.get_event_loop()
    signal.signal(signal.SIGINT, partial(sigint_handler, loop, cam))
    signal.signal(signal.SIGQUIT, partial(sigquit_handler, loop, cam))
    print("Running...")
    print("Please CTRL-C to reload the background / foreground images")
    print("Please CTRL-\ to exit")
    # frames forever
    loop.run_until_complete(cam.run())


if __name__ == "__main__":
    main()
