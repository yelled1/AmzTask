"""
Helper function for analysis
"""
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from imblearn.datasets import make_imbalance
#from sklearn.preprocessing import LabelBinarizer
#from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error

def resample_split_data(df, resample_ratio=1.0, test_size=0.3, debug=False):
  """
  Resample data based on minor classification multiples, as boosting does NOT quite do the job
  Using only train & test as data availability is limited to have validation set
  And not part of this excercis
  """
  y = df['Fails']
  X = df[['count', 'm1_max', 'm1_mean', 'm1_min', 'm2_max', 'm2_mean', 'm2_min',
          'm3_max', 'm3_mean', 'm3_min', 'm4_max', 'm4_mean', 'm4_min',
          'm5_max', 'm5_mean', 'm5_min', 'm6_max', 'm6_mean', 'm6_min',
          'm7_max', 'm7_mean', 'm7_min', 'm8_max', 'm8_mean', 'm8_min',
          'm9_max', 'm9_mean', 'm9_min']]
  # get the min of either (max available dominant classification or multiple of minor classification
  dominant_resample_count = min(int(y.sum() * resample_ratio), y.shape[0] - y.sum())
  X_res, y_res = make_imbalance(X, y,
                                sampling_strategy={
                                  True: y.sum(), False: dominant_resample_count},
                                random_state=42)
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
    conf_mtx[1, :].sum(), conf_mtx[1, 1], 100 * conf_mtx[1, 1] / conf_mtx[1, :].sum()))

  print("Accuracy: {:,.3f}%".format(100 * accuracy_score(test_y, predict_y)))
  print("*F1_score = {:,.1f}%".format(100 * f1_score(test_y, predict_y, 'binary')))