import matplotlib.pyplot as plt
import numpy as np
import cross_validation as cv

for i in range(1,7):
    X, Y, masks, ind, cutoffs = cv.CV(agent='rainbow_agent_{}'.format(i))
    freq_count = np.zeros(20)

    for y in Y: # y is numpy.ndarray
        y = list(y).index(1)
        freq_count[y]+=1

    plt.subplot(3, 2, i)
    freq_count = list(freq_count)
    plt.bar(np.arange(0, 20, 1), freq_count, width = 1)

plt.show()
