"""
Helper function for analysis
"""
import os
import sys
import pickle
from collections import Counter
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, f1_score,
                             confusion_matrix)
from imblearn.datasets import make_imbalance

def xstat(df, no_zero=False, debug=False):
  """
  Function to apply to grouped data to summarize as input X:
    Same as above but easier to add stats
  """
  Fails = df['failure'].sum() > 0
  # if set as True,then do NOT take 0 day or day of the failure
  if no_zero:
    df = df[df.daysDelta != 0]
  df = df[df.daysDelta != 0]
  r_dict = {'Fails': Fails, 'count': df.shape[0]}
  if not Fails and debug:
    print(Fails, r_dict)
  for col in df.columns:
    if col[:-1] == 'metric':
      no = col[-1]
      max_v = df[col].max()
      min_v = df[col].min()
      dDelta_max = None
      dDelta_min = None
      if Fails:
        dDelta_max = df[df[col] >= max_v].daysDelta.iloc[0]
        dDelta_min = df[df[col] <= min_v].daysDelta.iloc[0]
      if debug:
        print(col, Fails, dDelta_max, max_v)
      t_dict = {'m{}_max'.format(no): max_v,
                'm{}_mean'.format(no): df[col].mean(),
                'm{}_min'.format(no): min_v,
                'm{}_max_d'.format(no): dDelta_max,
                'm{}_min_d'.format(no): dDelta_min, }
      r_dict = {**r_dict, **t_dict}
  return pd.Series(r_dict)


def resample_split_data(df, y_col='Fails', x_cols=['count',
                                                   'm1_max', 'm1_mean', 'm1_min',
                                                   'm2_max', 'm2_mean', 'm2_min',
                                                   'm3_max', 'm3_mean', 'm3_min',
                                                   'm4_max', 'm4_mean', 'm4_min',
                                                   'm5_max', 'm5_mean', 'm5_min',
                                                   'm6_max', 'm6_mean', 'm6_min',
                                                   'm7_max', 'm7_mean', 'm7_min',
                                                   'm8_max', 'm8_mean', 'm8_min',
                                                   'm9_max', 'm9_mean', 'm9_min',],
                        resample_ratio=1.0, test_size=0.3, debug=False):
  """
  x_cols: Could have used an auto generate or slice of df.columns, but this is clearer
  Resample data based on minor classification multiples, as boosting does NOT quite do the job
  Using only train & test as data availability is limited to have validation set
  And not part of this excercis
  if index: y = df.index.get_level_values(1).values
  """
  y = df[y_col]
  X = df[x_cols]

  # get the min of ("max dominant rows" or "multiple of minor classification")
  dominant_resample_count = min(int(y.sum() * resample_ratio), y.shape[0] - y.sum())
  X_res, y_res = make_imbalance(X, y,
                                sampling_strategy={
                                  True: y.sum(),
                                  False: dominant_resample_count},)
                                #random_state=42)
  if debug:
    print('Distribution before imbalancing: {}'.format(Counter(y)))
    print('Distribution after  imbalancing: {}'.format(Counter(y_res)))
  # return split data (1-test_size) 70% tkraining & test_size 30% test data
  return train_test_split(X_res, y_res, test_size=test_size)


def print_prediction_results(train_y, test_y, predict_y):
  """ Quick print statement to look at stats of the model
    F1 is really important as that reduces downtime accuracy is NOT as
  """
  conf_mtx = confusion_matrix(test_y, predict_y)
  print("Result from confusion_matrix\n{}\n".format(conf_mtx))
  print("Training data had {} data points & {} fails".format(train_y.shape[0],
                                                             train_y.sum()))
  print("Testing  data had {} data points & {} fails".format(test_y.shape[0],
                                                             test_y.sum()))
  print("Out of {} Fails, predicted {} or {:,.1f}% correctly".format(
    conf_mtx[1, :].sum(), conf_mtx[1, 1],
    100 * conf_mtx[1, 1] / conf_mtx[1, :].sum()))

  print("Accuracy: {:,.3f}%".format(100 * accuracy_score(test_y, predict_y)))
  print("*F1_score = {:,.1f}%".format(100 * f1_score(test_y, predict_y, 'binary')))


def pca_2d_imbal(X_orig, X_rs):
  """
  Use principal component to condense the ? features to 2 features
  """
  pca_o = PCA(n_components=2).fit(X_orig)
  pca_o2d = pca_o.transform(X_orig)
  pca_r = PCA(n_components=2).fit(X_rs)
  pca_r2d = pca_r.transform(X_rs)
  return pca_o2d, pca_r2d

def pca2d_graph_imbal(orig_df, x_cols, X_imb_adj, y_imb_adj):
  """
  graph 2d PCA prior to imbalance adjustment & after adjustment
  """
  pca_2d_orig, pca_2d_trasf = pca_2d_imbal(orig_df[x_cols], X_imb_adj)
  pca2d_df_o = pd.DataFrame(pca_2d_orig, columns=['class1', 'class2'])
  pca2d_df_t = pd.DataFrame(pca_2d_trasf, columns=['class1', 'class2'])

  cmap = cm.get_cmap('Spectral') # Colour map (there are many others)
  mpl.rcParams.update(mpl.rcParamsDefault)

  fig, axes = plt.subplots(1,2)
  fig.suptitle("Comparing 2D PCA of Original vs Adj Data")
  axes[0].set(title="Original Data")
  axes[1].set(title="Imbalanced Adj Data")
  axes[0].scatter(x=pca2d_df_o['class1'], y=pca2d_df_o['class2'], c=orig_df['Fails'],
                  s=40, cmap=cmap, edgecolor='None')
  axes[1].scatter(x=pca2d_df_t['class1'], y=pca2d_df_t['class2'], c=y_imb_adj,
                  s=40, cmap=cmap, edgecolor='None')
  plt.legend()
  plt.savefig("/tmp/PCA.png")
  plt.show()

if __name__ == '__main__':
  pkl_file = './parsed_dataframe.pkl'
  if os.path.isfile(pkl_file):
    # Alternative is Load from Pickled set: Saves Time
    with open(pkl_file, 'rb') as fp:
      ret_dict = pickle.load(fp)
  else:
    print("pickle file {} not avail".format(pkl_file))
    sys.exit()

  prev_df = ret_dict['csv_df']
  #: csv parse & load
  grp = ret_dict['grp_dev_fail']
  #: grouped by device & [Failed or Normal set]
  g_stat_df = ret_dict['g_stat_df']
  #: processing based on the grp above -> max, min, mean & # days prior to max & min
  s_df = grp.get_group(('S1F03YZM', True))
  s_df = grp.get_group(('S1F0CTDN', True))
  print(s_df)
