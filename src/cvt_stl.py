#!/usr/bin/env python3

from stl2voxel import doExport
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def _3d_plot(voxel, sub_sample=10):
    idx = np.arange(voxel.shape[0])
    sparse_voxel = voxel[idx % sub_sample == 0]
    x, y, z = sparse_voxel[:, 0], sparse_voxel[:, 1], sparse_voxel[:, 2]
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(x, y, z)
    # ax.set_aspect('equal')
    # plt.axis("equal")
    plt.grid()
    plt.show()


src_path = '../data/stl/stl210-017_03.STL'
dest_path = '../data/xyz/stl210-017_03.xyz'

doExport(src_path, dest_path, 98)

voxel = np.loadtxt(dest_path, dtype=np.int)

_3d_plot(voxel, 10)
