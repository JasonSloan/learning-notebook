import math
from matplotlib import pyplot as plt

def one_cycle(y1=0.0, y2=1.0, steps=100):
    # lambda function for sinusoidal ramp from y1 to y2 https://arxiv.org/pdf/1812.01187.pdf
    return lambda x: ((1 - math.cos(x * math.pi / steps)) / 2) * (y2 - y1) + y1


if __name__ == '__main__':
    lf = one_cycle(1, 0.01, 32)
    for i in range(100):
        plt.plot(i, lf(i), 'ro')
    plt.xlabel("epochs")
    plt.ylabel("lr")
    plt.savefig('one_cycle.jpg')
