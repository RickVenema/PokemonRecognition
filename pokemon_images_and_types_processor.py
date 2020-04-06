import pandas as pd
import os
import matplotlib.pyplot as plt
import cv2


def process():
    path = "pokemon-images-and-types/"
    csv_loc = path + "pokemon.csv"
    csv_data = pd.read_csv(csv_loc, delimiter=",", index_col=0)
    for Cat in csv_data["Type1"].unique():
        tmp_data = csv_data[csv_data["Type1"] == Cat]
        file_names = tmp_data.index.values
        for file in file_names:
            # print(path+"images/images/"+file)
            try:
                img = cv2.imread(path + "images/images/" + file + ".png")
                cv2.imwrite(path + "images/processed_images/" + Cat + "/" + file + ".png", img)
            except cv2.error:
                img = cv2.imread(path + "images/images/" + file + ".jpg")
                cv2.imwrite(path + "images/processed_images/" + Cat + "/" + file + ".jpg", img)


if __name__ == '__main__':
    process()
