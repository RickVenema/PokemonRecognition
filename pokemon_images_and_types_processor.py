import pandas as pd
import os


def process():
    path = "pokemon-images-and-types/"
    csv_loc = path + "pokemon.csv"
    csv_data = pd.read_csv(csv_loc, delimiter=",", index_col=0)
    for root, dirs, files in os.walk(path + "images/images/"):
        print(files)


if __name__ == '__main__':
    process()
