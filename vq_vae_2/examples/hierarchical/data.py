"""
Data loading pipeline.

All data should be stored as images in a single directory.
"""

import os
import random

from PIL import Image
import numpy as np
import torch


class SwipeCropper(object):
    
    def __init__(self, image, wr, hr):
        self.image = image
        self.wr = wr
        self.hr = hr
        
    def __iter__(self):
        return self
        
    def tiles(self):
        wr = self.wr
        hr = self.hr
        w, h, c = self.image.shape
        if (h < hr) or (w < wr):
            print("Image too small.")
            return
        hd = (hr - (h % hr)) / ( h // hr )
        wd = (wr - (w % wr)) / ( w // wr )
        for hn in range(h//hr + 1):
            for wn in range(w//wr + 1):
                h0 = int((hn * hr) - (hn * hd))
                w0 = int((wn * wr) - (wn * wd))
                h1 = int(h0 + hr)
                w1 = int(w0 + wr)
                yield(self.image[w0:w1, h0:h1, :])

    
def load_tiled_images(dir_path, batch_size=8, width=128, height=128):
    images = load_single_images_uncropped(dir_path)
    batch = []
    while True:
        try:
            image = next(images)
            cropper = SwipeCropper(np.array(image), width, height)
            tiles = cropper.tiles()
            while True:
                try:
                    tile = next(tiles)
                    batch.append(tile)
                    if len(batch) == batch_size:
                        batch = np.array(batch)
                        batch = torch.from_numpy(batch).permute(0, 3, 1, 2).contiguous()
                        batch = batch.float() / 255
                        yield batch
                        batch = []
                except StopIteration:
                    break
        except StopIteration:
            continue

                        
def load_images(dir_path, batch_size=16):
    images = load_single_images(dir_path)
    while True:
        batch = np.array([next(images) for _ in range(batch_size)])
        batch = torch.from_numpy(batch).permute(0, 3, 1, 2).contiguous()
        batch = batch.float() / 255
        yield batch


def load_single_images(dir_path):
    while True:
        with os.scandir(dir_path) as listing:
            for entry in listing:
                if not (entry.name.endswith('.png') or entry.name.endswith('.jpg')):
                    continue
                try:
                    img = Image.open(entry.path)
                except OSError:
                    # Ignore corrupt images.
                    continue
                width, height = img.size
                scale = IMAGE_SIZE / min(width, height)
                img = img.resize((round(scale * width), round(scale * height)))
                img = img.convert('RGB')
                tensor = np.array(img)
                row = random.randrange(tensor.shape[0] - IMAGE_SIZE + 1)
                col = random.randrange(tensor.shape[1] - IMAGE_SIZE + 1)
                yield tensor[row:row + IMAGE_SIZE, col:col + IMAGE_SIZE]

def load_single_images_uncropped(dir_path):
    while True:
        with os.scandir(dir_path) as listing:
            for entry in listing:
                if not (entry.name.endswith('.png') or entry.name.endswith('.jpg')):
                    continue
                try:
                    img = Image.open(entry.path)
                except OSError:
                    # Ignore corrupt images.
                    continue
                img = img.convert('RGB')
                tensor = np.array(img)
                yield tensor

