import main_net
import data_loader


def main():
    data = data_loader.DataLoader("pokemon-images-and-types/")
    nn = main_net.Network(data)
    nn.train()
    # print(nn)


if __name__ == '__main__':
    main()
