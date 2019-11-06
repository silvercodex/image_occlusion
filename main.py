import os
from image_occlusion import *


def load_images(dir = ""):
    files = os.listdir(dir)
    images = []
    for f in files:
        images.append(cv2.imread(dir + "/" + f))
    return images

def get_edges(images):
    return [cv2.Canny(image,100,200) for image in images]


if __name__ == '__main__':
    images = load_images()
    edges = get_edges(images)
    