from matplotlib import pyplot as plt

lf = lambda x: (1 - x / epochs) * (1.0 - lrf) + lrf

if __name__ == '__main__':
    lrf = 0.01
    epochs = 100
    for i in range(100):
        plt.plot(i, lf(i), 'ro')
    plt.xlabel("epochs")
    plt.ylabel("lr")
    plt.savefig('linear.png')
