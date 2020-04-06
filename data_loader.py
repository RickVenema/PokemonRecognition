import numpy as np
import cv2
import os

import random
from tqdm import tqdm
import pickle


class DataLoader:
    def __init__(self, data_loc):
        self.data_loc = data_loc
        self.X, self.y = self.load_image_folder()

    def load_image_folder_create(self):
        training_data = []
        IMG_SIZE = 150
        data_dir = self.data_loc + "PokemonData/"
        categories = os.listdir(data_dir)
        print(categories)
        for cat in categories:
            class_num = categories.index(cat)
            path = os.path.join(data_dir, cat)
            for img in tqdm(os.listdir(path)):
                try:
                    img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)  # convert to array
                    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
                    training_data.append([new_array, class_num])  # add this to our training_data
                except Exception as e:  # in the interest in keeping the output clean...
                    pass

        print(training_data)
        random.shuffle(training_data)

        X = []
        y = []

        for features, label in training_data:
            X.append(features)
            y.append(label)

        X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

        pickle_out = open("X.pickle", "wb")
        pickle.dump(X, pickle_out)
        pickle_out.close()

        pickle_out = open("y.pickle", "wb")
        pickle.dump(y, pickle_out)
        pickle_out.close()
        return X, y

    def load_image_folder(self):
        try:
            pickle_in = open("X.pickle", "rb")
            X = pickle.load(pickle_in)

            pickle_in = open("y.pickle", "rb")
            y = pickle.load(pickle_in)
        except Exception as e:
            X, y = self.load_image_folder_create()
        return X, y