import pandas as pd
import numpy as np

def train_validate_test_split(df, train_percent=.6, validate_percent=.2, seed=None):
  """
  this split Function was written to handle seeding of randomized set creation.
  Seed guarantees same result unless a new seed is entered
  """
  np.random.seed(seed)
  perm = np.random.permutation(df.index)
  m = len(df.index)
  train_end = int(train_percent * m)
  validate_end = int(validate_percent * m) + train_end
  train = df.iloc[perm[:train_end]]
  validate = df.iloc[perm[train_end:validate_end]]
  test = df.iloc[perm[validate_end:]]
  return train, validate, test

"""
train_df, validate_df, test_df = train_validate_test_split(pm_df, seed=951)
train_df.failure.value_counts()
validate_df.failure.value_counts()
test_df.failure.value_counts()
"""
