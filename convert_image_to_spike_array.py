import numpy as np
import cv2
import os
import argparse

import pylab

from poisson_tools import image_to_poisson_trains, image_to_ttfs_trains
from util_functions import *


def img_to_spike_array(img_file_name, encoding, max_freq, on_duration, off_duration,
                       save_as_pickle=True):
    img = cv2.imread(img_file_name, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Image couldn't be read! -> {}".format(img_file_name))
        return

    height, width = img.shape
    image_data = np.array([img.reshape(height * width)])

    if encoding == 'ttfs':
        spikes = image_to_ttfs_trains(image_data, height, width, on_duration, off_duration)
    else:
        spikes = image_to_poisson_trains(image_data, height, width,
                                         max_freq, on_duration, off_duration)

    pylab.figure()
    raster_plot_spike(spikes)
    pylab.title("{} encoding — {}".format(encoding.upper(), os.path.basename(img_file_name)))
    pylab.xlabel("Time (ms)")
    pylab.ylabel("Neuron ID")
    pylab.show()

    if save_as_pickle:
        base = img_file_name[img_file_name.rfind('/') + 1: img_file_name.rfind('.')]
        pickle_file = "spike_array_{}_{}".format(encoding, base)
        pickle_it(spikes, pickle_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Convert an image (or folder of images) to a spike array for SNNs.'
    )
    parser.add_argument('img_path',
                        help='Path to an image file or a folder containing .png images')
    parser.add_argument('--encoding', choices=['poisson', 'ttfs'], default='poisson',
                        help='Spike encoding scheme: poisson (rate coding) or ttfs '
                             '(time-to-first-spike). Default: poisson')
    parser.add_argument('--max_freq', type=int, default=1000,
                        help='Max firing frequency in Hz (Poisson only). Default: 1000')
    parser.add_argument('--on_duration', type=int, default=200,
                        help='Stimulus presentation duration in ms. Default: 200')
    parser.add_argument('--off_duration', type=int, default=100,
                        help='Silent gap between images in ms. Default: 100')
    parser.add_argument('--no_pickle', action='store_true',
                        help='Do not save the spike array as a pickle file')

    args = parser.parse_args()

    print("Encoding:     {}".format(args.encoding))
    print("on_duration:  {} ms".format(args.on_duration))
    print("off_duration: {} ms".format(args.off_duration))
    if args.encoding == 'poisson':
        print("max_freq:     {} Hz".format(args.max_freq))

    save_as_pickle = not args.no_pickle

    if os.path.isdir(args.img_path):
        import glob2
        image_list = glob2.glob(os.path.join(args.img_path, "**/*.png"))
        for img in image_list:
            if os.path.isfile(img):
                img_to_spike_array(img, args.encoding, args.max_freq,
                                   args.on_duration, args.off_duration, save_as_pickle)
    elif os.path.isfile(args.img_path):
        img_to_spike_array(args.img_path, args.encoding, args.max_freq,
                           args.on_duration, args.off_duration, save_as_pickle)
    else:
        print("Error: '{}' is not a valid file or directory.".format(args.img_path))
