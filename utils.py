from groupy.gfunc.z2func_array import Z2FuncArray
from groupy.gfunc.p4func_array import P4FuncArray
from groupy.gfunc.p4mfunc_array import P4MFuncArray
import groupy.garray.C4_array as C4a
import groupy.garray.D4_array as D4a
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt


def check_equivariance(im, output, input_array, output_array, point_group):

  # Transform the image
  f = input_array(im.transpose((0, 3, 1, 2)))
  g = point_group.rand()
  gf = g * f
  im1 = gf.v.transpose((0, 2, 3, 1))

  # Compute
  yx = output([im])[0]
  yrx = output([im1])[0]

  # Transform the computed feature maps
  fmap1_garray = output_array(yrx.transpose((0, 3, 1, 2)))
  r_fmap1_data = (g.inv() * fmap1_garray).v.transpose((0, 2, 3, 1))

  print(np.abs(yx - r_fmap1_data).sum())
  assert np.allclose(yx, r_fmap1_data, rtol=1e-5, atol=1e-3)


def visualize_latent_space(predict_fn, n=15, size=28, atent_dim=10):

  # linearly spaced coordinates on the unit square were transformed
  # through the inverse CDF (ppf) of the Gaussian to produce values
  # of the latent variables z, since the prior of the latent space
  # is Gaussian

  z1 = norm.ppf(np.linspace(0.01, 0.99, n))
  z2 = norm.ppf(np.linspace(0.01, 0.99, n))
  z_grid = np.dstack(np.meshgrid(z1, z2))

  x_pred_grid = predict_fn(z_grid.reshape(n * n, latent_dim))  #.reshape(n, n, digit_size, digit_size)

  plt.figure(figsize=(10, 10))
  plt.imshow(np.block(list(map(list, x_pred_grid))), cmap='gray')
  plt.show()


def plot_many_vs_reconstructed(original, reconstructed, n=10):
  plt.figure(figsize=(12, 4))
  n = [2, n]
  for i in range(n[1]):
    idx = indexes[i]
    plt.subplot(n[0], n[1], i + 1)
    plt.imshow(original[idx, :, :, 0])  #, cmap=plt.get_cmap("Greys"))
    plt.axis('off')
    plt.subplot(n[0], n[1], i + 1 + n[1])
    plt.imshow(reconstructed[idx, :, :, 0])  #, cmap=plt.get_cmap("Greys"))
    plt.axis('off')
