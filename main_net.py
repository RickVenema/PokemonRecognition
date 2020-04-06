import tensorflow as tf
import os
from data_loader import DataLoader


class Network:
    def __init__(self, train_data):
        self.data = train_data
        self.model = self.create_model()

    def __str__(self):
        return f"""
        \r Network with followin model:
        {self.model.summary()}
        """

    def create_model(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Conv2D(256, (3, 3), input_shape=self.data.X.shape[1:], activation='relu'))
        # model.add(tf.keras.layers.Activation('relu'))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

        model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
        # model.add(tf.keras.layers.Activation('relu'))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

        model.add(tf.keras.layers.Conv2D(64, (3, 3), activation="relu"))
        # model.add(tf.keras.layers.Activation('relu'))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

        model.add(tf.keras.layers.Conv2D(32, (3, 3), activation="relu"))
        # model.add(tf.keras.layers.Activation('relu'))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

        model.add(tf.keras.layers.Flatten())  # this converts our 3D feature maps to 1D feature vectors
        model.add(tf.keras.layers.Dropout(0.2))

        model.add(tf.keras.layers.Dense(1024, activation="relu"))
        model.add(tf.keras.layers.Dense(512, activation="relu"))
        # model.add(tf.keras.layers.Dense(256, activation="relu"))

        model.add(tf.keras.layers.Dense(150, activation="softmax"))

        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        print(model)
        return model

    def train(self):
        self.model.fit(self.data.X, self.data.y, batch_size=32, epochs=30, validation_split=0.3)

