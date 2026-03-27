import h5py

path = "../data/GLA_test.h5"

with h5py.File(path, "r") as f:
    print("\nDATASETS:")
    for k in f.keys():
        print(k, f[k].shape, f[k].dtype)