"""
load data from the csv & sort_values
adds days delta calc vs. failure date if available
"""
import time
import pandas as pd
import numpy as np

def days_delta_calc(df):
  """ given data_frame returns days from the failure date  """
  return df.date.progress_apply(lambda x: (x - df.loc[df[df.failure].index[0]].date).days)

def load_parse_save(in_file_name='./predictive_maintenance.csv', save_file='', debug=False):
  """
  read in input csv file & returns pased dateframe & saves
  a save_file, if save_file name is specificed
  """
  if debug:
    print(time.ctime())
  dt_parser = lambda x: pd.datetime.strptime(x, "%m/%d/%y")
  df = pd.read_csv(in_file_name, parse_dates=[0], date_parser=dt_parser)
  #df.date = df.date.progress_apply(lambda x: pd.to_datetime(x, format="%m/%d/%Y"))
  #sort by device & date
  pm_df = df.sort_values(['device', 'date'])
  pm_df = pm_df.reset_index(drop=True)
  df = None
  pm_df.failure = pm_df.failure.apply(lambda x: True if x > 0 else False)
  # adds following new columns
  pm_df['daysDelta'] = np.nan # date from the failure date
  # convert date to pandas datetime for calculation
  # get list of failed devices for special dateDelta processing
  dev_failed = pm_df.device[pm_df.failure].tolist()

  # loop & if device is
  for ky in dev_failed:
    pm_df.loc[pm_df.device == ky, 'daysDelta'] = days_delta_calc(pm_df[pm_df.device == ky])
  if save_file != '':
    pm_df.to_pickle(save_file)
  if debug:
    print(time.ctime())
  return pm_df
