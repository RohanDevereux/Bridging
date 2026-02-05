import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def soft_contact_maps(traj, idx1, idx2, stride, d0_nm, k_nm, dtype="float16"):
    xyz = traj.xyz[::stride]
    A = xyz[:, idx1, :]
    B = xyz[:, idx2, :]

    diff = A[:, :, None, :] - B[:, None, :, :]
    D = np.sqrt((diff * diff).sum(-1))
    C = sigmoid(k_nm * (d0_nm - D))

    return C.astype(dtype)
