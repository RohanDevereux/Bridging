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


def contact_distance_channels(
    traj,
    idx1,
    idx2,
    stride,
    d0_nm,
    k_nm,
    d_clip_nm=2.0,
    use_log_dist=True,
    dtype="float16",
):
    xyz = traj.xyz[::stride]
    A = xyz[:, idx1, :]
    B = xyz[:, idx2, :]

    diff = A[:, :, None, :] - B[:, None, :, :]
    D = np.sqrt((diff * diff).sum(-1))
    C = sigmoid(k_nm * (d0_nm - D))

    D_clip = np.clip(D, 0.0, d_clip_nm)
    if use_log_dist:
        D_feat = np.log1p(D_clip) / np.log1p(d_clip_nm)
    else:
        D_feat = D_clip / d_clip_nm

    X = np.stack([C, D_feat], axis=1)
    return X.astype(dtype)
