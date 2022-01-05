import csv

import matplotlib.pyplot as plt
import numpy as np

# Read CSV
# for attack in ['awgn', 'median', 'blur', 'resize', 'sharpen']:
for attack in ['jpeg']:
    # for im in ['buildings', 'rollercoaster', 'tree']:
    csvFileName = f'{attack}-test.csv'  # f'{im}-alpha.csv'
    csvData = []
    with open(csvFileName, 'r') as csvFile:
        csvReader = csv.reader(csvFile, delimiter=';')
        for csvRow in csvReader:
            csvData.append(csvRow)

    # Get X, Y, Z
    csvData = np.array(csvData)[:]
    csvData = csvData.astype(float)
    X, Y, Z, thr = csvData[:, 0], csvData[:, 1], csvData[:, 2], csvData[:, 3],

    Xr, Yr, Zr = [], [], []
    Xg, Yg, Zg = [], [], []

    for i in range(len(thr)):
        if Z[i] > 35 and thr[i] == 0:
            Xr.append(X[i])
            Yr.append(Y[i])
            Zr.append(Z[i])
        else:
            Xg.append(X[i])
            Yg.append(Y[i])
            Zg.append(Z[i])

    #
    #     plt.plot(X, Y)
    #
    # plt.vlines(0.3, 40, 45.16,
    #            colors='black', linestyles='dashed', linewidth=1)
    # plt.legend(['buildings', 'rollercoaster', 'tree'])
    #
    # plt.xticks(rotation=30)
    # plt.xlabel('Alpha')
    # plt.ylabel('WPSNR')

    # Plot X,Y,Z

    fig = plt.figure(figsize=(10, 10), dpi=400)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(X, Y, Z, color='grey', alpha=0.33)
    ax.scatter(Xg, Yg, Zg, c='grey')
    ax.scatter(Xr, Yr, Zr, c='red')
    ax.set_xlabel('Alpha')
    ax.set_ylabel(f'{attack}-param')
    ax.set_zlabel('WPSNR')
    plt.tight_layout()
    plt.savefig(f'{attack}-alpha.png')
