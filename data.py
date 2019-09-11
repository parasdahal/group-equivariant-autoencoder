import numpy as np
import os, sys
import keras
from disentanglement_lib.data.ground_truth.dsprites import DSprites


def disentanglement_data(dataset):

  data_path = os.environ.get("DISENTANGLEMENT_LIB_DATA", ".")

  if dataset == 'dsprites':
    DSPRITES_PATH = os.path.join(
        data_path, "dsprites", "dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz")
    with file(DSPRITES_PATH, "rb") as data_file:
      data = np.load(data_file, encoding="latin1", allow_pickle=True)
      data = np.array(data["imgs"])
      input_shape = (64, 64, 1)
      ground_truth = DSprites()

  data = data.reshape((data.shape[0],) + input_shape)
  return ground_truth, input_shape, data