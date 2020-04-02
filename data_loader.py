import pandas as pd
import tensorflow as tf


class DataLoader:
    def __init__(self, data_loc):
        self.data_loc = data_loc
        self.csv_data = self.load_class_csv()
        self.train_data, self.validation = self.load_image_folder()

    def load_class_csv(self):
        csv_loc = self.data_loc + "pokemon.csv"
        csv_data = pd.read_csv(csv_loc, delimiter=",", index_col=0)
        return csv_data

    def load_image_folder(self):
        image_loc = self.data_loc + "/images/images/"
        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1. / 255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)
        test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)

        train_generator = train_datagen.flow_from_directory(
            image_loc,
            target_size=(120, 120),
            batch_size=32,
            class_mode='binary')
        validation_generator = test_datagen.flow_from_directory(
            image_loc,
            target_size=(120, 120),
            batch_size=32,
            class_mode='binary')

        return train_generator, validation_generator
