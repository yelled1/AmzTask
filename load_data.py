"""
load data from the csv & sort_values
adds days delta calc vs. failure date if available
"""
import time
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
from hlp_analysis import xstat

def days_delta_calc(df):
  """ given data_frame returns days from the failure date  """
  return df.date.progress_apply(lambda x: (x - df.loc[df[df.failure].index[0]].date).days)

def load_parse_save(in_file_name='./predictive_maintenance.csv', save_file='', debug=False):
  """
  read in input csv file & returns pased dateframe & saves
  a save_file, if save_file name is specificed
  """
  beg = time.time()
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
    beg = time.time()
  pm_df['Fail_set'] = pm_df.daysDelta.apply(lambda x: True if x <= 0.0 else False)
  grp = pm_df.groupby(['device', 'Fail_set'])
  if debug:
    print('Processing csv at {} took {:,.2f} sec'.format(time.ctime(), time.time()-beg))
  g_stat_df = grp.apply(lambda x: xstat(x, False, False))
  # saving dict of processed_csv, group, & stat_df from group
  ret_dict = {'csv_df': pm_df, 'grp_dev_fail': grp, 'g_stat_df': g_stat_df}
  if debug:
    print('Processing g_df at {} took {:,.2f} sec'.format(time.ctime(), time.time()-beg))
    print(g_stat_df.head(15))
  if save_file != '':
    with open(save_file, 'wb') as fw:
      pickle.dump(ret_dict, fw)
  return ret_dict

if __name__ == '__main__':
  pkl_file = './parsed_dataframe.pkl'
  tqdm.pandas(desc='load & parse csv')
  pkl_dict = load_parse_save(save_file='./parsed_dataframe.pkl', debug=True)
