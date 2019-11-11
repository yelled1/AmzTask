The observations likely time related issues where machine(s) w/ some
beyond the range &/or combinations of metric(s) result in future failures 
whichwould mean finding acceptable vs. critical range (as logit?) and identifying
simultaneous or lagged combos that results in failures.

So, instance of failure metric may be meaningful but for this project is to
more valueable, the prior reading of metric(s) that indicate future failure is
important; however, # days_prior to failure may matter (as data can be stale).

Preventive replacement of device is often a lot cheaper than unexpected failure
in industrial settings.
1. The data is sparse in that some machines have a lot of observations (over 200+)
   while a lot of machines only 1 readings. One must assume that only time an
   observation is recorded is when sensor(s) fire due to something unusual.
2. Opm_df.shape: (124494, 12) <- one may think this is imbalanced class problem
   if one just looks at the fact only 106 fails out of that many data points.
   Still F1 or AUC may be interesting stat to look at
3. len(grp_device) = 1169, Failures = 106
4. min, Max of cols are
  -  device S1F01085 Z1F2PBHX
  -  failure 0 1
  -  metric1 0 244140480
  -  metric2 0 64968
  -  metric3 0 24929
  -  metric4 0 1666
  -  metric5 1 98
  -  metric6 8 689161
  -  metric7 0 832
  -  metric8 0 832
  -  metric9 0 18701
  - Interesting to look at these as 0 observation might indicate nulls.
5. No device failed more than 1 time.
6. Probably need to look at pre/post failure rows for device separately.
7. Do we include actual failure or only before?
