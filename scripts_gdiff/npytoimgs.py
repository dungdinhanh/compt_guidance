import numpy as np
from PIL import Image
import argparse
import os


def read_file_to_numpy(file):
    npzfile = np.load(file)
    n = len(npzfile)
    if n == 1:
        return list(npzfile['arr_0']), None
    else:
        all_images = list(npzfile['arr_0'])
        all_labels = list(npzfile['arr_1'])
        return all_images, all_labels
    pass


def load_numpy_labels_to_image(array1, array2, folder_path):
    if array2 is None:
        array2 = np.ones(len(array1))
    for i in range(len(array1)):
        im = Image.fromarray(array1[i], "RGB")
        img_name = "image"
        img_name += f"{array2[i]}"
        im_path = os.path.join(folder_path, f"class_{array2[i]}")
        os.makedirs(im_path, exist_ok=True)
        im.save(os.path.join(im_path, f"{img_name}_{i}.png"))
    pass


def load_numpy_labels_to_grayimage(array1, array2, folder_path):
    for i in range(len(array1)):
        img = np.squeeze(array1[i], axis=-1)
        im = Image.fromarray(img, "L") # for RGB use different
        im.save(os.path.join(folder_path, f"%d_image%d.png"%(int(array2[i]), i)))


parser = argparse.ArgumentParser("numpy to images")
parser.add_argument("--numpy", help="numpy files", default="../outputhfai/runs/")
parser.add_argument("--rgb", action="store_true", help="process rbg")



if __name__ == '__main__':
    args = parser.parse_args()
    file_numpy = args.numpy
    folder_images = os.path.join(os.path.dirname(file_numpy), "images2")
    os.makedirs(folder_images, exist_ok=True)
    images_array, labels = read_file_to_numpy(file_numpy)
    if not args.rgb:
        load_numpy_labels_to_grayimage(images_array, labels, folder_images)
    else:
        load_numpy_labels_to_image(images_array, labels, folder_images)