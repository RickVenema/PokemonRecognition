import tensorflow as tf
import os
from data_loader import DataLoader


class Network:
    def __init__(self, train_data):
        self.model = self.create_model()
        self.data = train_data

    def __str__(self):
        return f"""
        \r Network with followin model:
        {self.model.summary()}
        """

    @staticmethod
    def create_model():
        model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(120, 120)),
            tf.keras.layers.Dense(4096),
            tf.keras.layers.Dense(4096),
            tf.keras.layers.Dense(4096),
            tf.keras.layers.Dense(18)
        ])
        model.compile(optimizer='adam',
                      loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                      metrics=['accuracy'])
        return model

    def train(self):
        self.model.fit_generator(
            self.data.train_data,
            steps_per_epoch=2000,
            epochs=50,
            validation_data=self.data.validation,
            validation_steps=800)

