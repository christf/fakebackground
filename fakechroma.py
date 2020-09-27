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
        scale_factor: float,
        use_foreground: bool,
        hologram: bool,
        tiling: bool,
        bodypix_url: str,
        socket: str,
        background_image: str,
        foreground_image: str,
        foreground_mask_image: str,
        webcam_path: str,
        v4l2loopback_path: str,
    ) -> None:
        self.use_foreground = use_foreground
        self.hologram = hologram
        self.tiling = tiling
        self.background_image = background_image
        self.foreground_image = foreground_image
        self.foreground_mask_image = foreground_mask_image
        self.scale_factor = scale_factor
        self.real_cam = RealCam(webcam_path, width, height, fps)
        # In case the real webcam does not support the requested mode.
        self.width = self.real_cam.get_frame_width()
        self.height = self.real_cam.get_frame_height()
        self.fake_cam = pyfakewebcam.FakeWebcam(v4l2loopback_path, self.width, self.height)
        self.foreground_mask = None
        self.inverted_foreground_mask = None
        self.session = requests.Session()
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
        self.image_lock = asyncio.Lock()


    async def _get_mask(self, frame):
#        _, data = cv2.imencode(".png", frame)
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        # define range of green color in HSV
        l_green = np.array([57, 0, 3])
        u_green = np.array([105, 255, 253])
#        l_green = np.array([110, 0, 0])
#        u_green = np.array([139, 255, 255])
        # Threshold the HSV image to extract green color
        mask = cv2.inRange(hsv, l_green, u_green)
#        mask = cv2.bitwise_not(mask)
        mask = mask.reshape((frame.shape[0], frame.shape[1]))

        # Filter using contour area and remove small noise
        thresh = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        cnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        for c in cnts:
            area = cv2.contourArea(c)
            if area < 5000:
                cv2.drawContours(thresh, [c], -1, (0,0,0), -1)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
#        cv2.imshow('close',close)
#        cv2.waitKey(5) & 0xFF
        mask = cv2.blur(close.astype(float), (10, 10))
        return mask.astype(float)/255

    def shift_image(self, img, dx, dy):
        img = np.roll(img, dy, axis=0)
        img = np.roll(img, dx, axis=1)
        if dy > 0:
            img[:dy, :] = 0
        elif dy < 0:
            img[dy:, :] = 0
        if dx > 0:
            img[:, :dx] = 0
        elif dx < 0:
            img[:, dx:] = 0
        return img

    async def load_images(self):
        async with self.image_lock:
            self.images: Dict[str, Any] = {}

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

    def hologram_effect(self, img):
        # add a blue tint
        holo = cv2.applyColorMap(img, cv2.COLORMAP_WINTER)
        # add a halftone effect
        bandLength, bandGap = 2, 3
        for y in range(holo.shape[0]):
            if y % (bandLength+bandGap) < bandLength:
                holo[y,:,:] = holo[y,:,:] * np.random.uniform(0.1, 0.3)
        # add some ghosting
        holo_blur = cv2.addWeighted(holo, 0.2, self.shift_image(holo.copy(), 5, 5), 0.8, 0)
        holo_blur = cv2.addWeighted(holo_blur, 0.4, self.shift_image(holo.copy(), -5, -5), 0.6, 0)
        # combine with the original color, oversaturated
        out = cv2.addWeighted(img, 0.5, holo_blur, 0.6, 0)
        return out


    async def mask_frame(self, frame):
        # fetch the mask with retries (the app needs to warmup and we're lazy)
        # e v e n t u a l l y c o n s i s t e n t
        mask = None
        while mask is None:
            try:
                mask = await self._get_mask(frame)
            except Exception as e:
                print(f"Mask request failed, retrying: {e}")
                traceback.print_exc()

        if self.hologram:
            frame = self.hologram_effect(frame)

        # composite the foreground and background
        async with self.image_lock:
            background = next(self.images["background"])
            for c in range(frame.shape[2]):
                frame[:, :, c] = frame[:, :, c] * mask + background[:, :, c] * (1 - mask)

            if self.use_foreground and self.foreground_image is not None:
                for c in range(frame.shape[2]):
                    frame[:, :, c] = (
                        frame[:, :, c] * self.images["inverted_foreground_mask"]
                        + self.images["foreground"][:, :, c] * self.images["foreground_mask"]
                        )

        return frame

    def put_frame(self, frame):
        self.fake_cam.schedule_frame(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    def stop(self):
        self.real_cam.stop()

    async def run(self):
        await self.load_images()
        self.real_cam.start()
        if self.socket != "":
            conn = aiohttp.UnixConnector(path=self.socket)
        else:
            conn = None
        async with aiohttp.ClientSession(connector=conn) as session:
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
                td = time.monotonic() - t0
                if td > print_fps_period:
                    self.current_fps = frame_count / td
                    print("FPS: {:6.2f}".format(self.current_fps), end="\r")
                    frame_count = 0
                    t0 = time.monotonic()

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
    parser.add_argument("-S", "--scale-factor", default=0.5, type=float,
                        help="Scale factor of the image sent to BodyPix network")
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
    print("Reloading background / foreground images")
    asyncio.ensure_future(cam.load_images())


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
        scale_factor=args.scale_factor,
        use_foreground=not args.no_foreground,
        hologram=args.hologram,
        tiling=args.tile_background,
        bodypix_url=args.bodypix_url,
        socket="",
        background_image=findFile(args.background_image, args.image_folder),
        foreground_image=findFile(args.foreground_image, args.image_folder),
        foreground_mask_image=findFile(args.foreground_mask_image, args.image_folder),
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