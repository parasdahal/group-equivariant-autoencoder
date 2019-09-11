import numpy as np
import os, sys
import keras
from disentanglement_lib.evaluation.metrics import beta_vae, factor_vae, mig,\
 modularity_explicitness, dci, sap_score, irs, unsupervised_metrics
from disentanglement_lib.evaluation.metrics import utils


def disentanglement_metric(representation_function, ground_truth_data, metric):

  available_metrics = [
      'beta_vae', 'factor_vae', 'mutual_info', 'modularity', 'dci', 'sap',
      'irs', 'unsupervised'
  ]

  if metric not in available_metrics:
    raise ValueError('{} not a valid metric'.format(metric))

  random_state = np.random.RandomState(42)
  batch_size = 64
  num_train = 2000
  num_eval = 1000
  num_test = num_eval
  num_variance_estimate = 2000

  if metric == 'beta_vae':
    score = beta_vae.compute_beta_vae_sklearn(ground_truth_data,
                                              representation_function,
                                              random_state, batch_size,
                                              num_train, num_eval)
  if metric == 'factor_vae':
    score = factor_vae.compute_factor_vae(ground_truth_data,
                                          representation_function, random_state,
                                          batch_size, num_train, num_eval,
                                          num_variance_estimate)
  if metric == 'mutual_info':

    utils.make_discretizer = lambda target, num_bins=20: utils._histogram_discretize(
        target, num_bins=20)

    score = mig.compute_mig(ground_truth_data, representation_function,
                            random_state, num_train, batch_size)
  if metric == 'modularity':
    score = modularity_explicitness.compute_modularity_explicitness(
        ground_truth_data, representation_function, random_state, num_train,
        num_test, batch_size)
  if metric == 'dci':
    score = dci.compute_dci(ground_truth_data, representation_function,
                            random_state, num_train, num_test, batch_size)
  if metric == 'sap':
    score = sap_score.compute_sap(ground_truth_data,
                                  representation_function,
                                  random_state,
                                  num_train,
                                  num_test,
                                  batch_size,
                                  continuous_factors=True)

  if metric == 'irs':
    score = irs.compute_irs(ground_truth_data,
                            representation_function,
                            random_state,
                            num_train,
                            batch_size,
                            diff_quantile=0.99)
  if metric == 'unsupervised':
    score = unsupervised_metrics.unsupervised_metrics(ground_truth_data,
                                                      representation_function,
                                                      random_state, num_train,
                                                      batch_size)

  return score