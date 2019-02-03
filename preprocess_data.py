import numpy as np
import cv2
import glob
import os
import json
import tensorflow as tf
import matplotlib.pyplot as plt
import time

work_dir = os.path.dirname(os.path.realpath('__file__'))
data_dir = work_dir + '\\datasets\\2019-02-02-11-31-07 - Day - No Traffic\\'


class Dataset(object):
    def __init__(self, dataset_name):
        info_path = os.path.join(data_dir, dataset_name + '.json')
        with open(info_path, 'r') as fp:
            info = json.load(fp)
        self.items = info['Items']


def get_semantic_map(path):
    dataset = Dataset('Windridge')
    file_names = glob.glob(path + "*_layer.png")
    color_to_id = {tuple(map(int, item['color'].split(','))): item['id'] for item in dataset.items}
    color_to_id[(0, 0, 0)] = 13
    color_to_id[(255, 255, 255)] = 14
    for step, file in enumerate(file_names):
        start = time.time()
        if step % 10 == 0:
            print("Step %d/%d" %(step, len(file_names)))
        file_name = file.split(path)[1].split('_')[0]
        rgb_image = cv2.imread(file)
        semantic_map = np.zeros((rgb_image.shape[0], rgb_image.shape[1], rgb_image.shape[2]))
        for i in range(rgb_image.shape[0]):
            for j in range(rgb_image.shape[1]):
                label = color_to_id.get(tuple(rgb_image[i, j, :]))
                semantic_map[i, j, :] = np.array([label, label, label])
        cv2.imwrite(path + file_name + '_semantic.png', semantic_map)
        end = time.time()
        print(end - start)


def get_semantic_map_tensor(path):
    dataset = Dataset('Windridge')
    file_names = glob.glob(path + "*_layer.png")
    palette = []
    for item in dataset.items:
        palette.append(np.array(list(map(int, item['color'].split(',')))))
    palette.append(np.array([0, 0, 0]))
    palette.append(np.array([255, 255, 255]))
    palette = np.array(palette)
    with tf.Session() as sess:
        for step, file in enumerate(file_names):
            start = time.time()
            if step % 10 == 0:
                print("Step %d/%d" %(step, len(file_names)))
            file_name = file.split(path)[1].split('_')[0]
            rgb_image = cv2.imread(file)
            semantic_map = []
            for colour in palette:
                class_map = tf.reduce_all(tf.equal(rgb_image, colour), axis=-1)
                semantic_map.append(class_map)
            semantic_map = tf.stack(semantic_map, axis=-1)
            semantic_map = tf.cast(semantic_map, tf.float32)
            class_indexes = tf.argmax(semantic_map, axis=-1)
            class_indexes = tf.expand_dims(class_indexes, 2)
            class_indexes = tf.tile(class_indexes, [1, 1, 3])
            semantic_map = sess.run(class_indexes)
            cv2.imwrite(path + file_name + '_semantic.png', semantic_map)
            end = time.time()
            print(end - start)


# Faster
get_semantic_map_tensor(data_dir)

# Slower
#get_semantic_map(data_dir)








