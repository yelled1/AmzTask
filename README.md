# OVERVIEW
The main goal is to predict Failure by Preventive Maintenance of Devices.
The Objectives are according to instructions:
1. Predict which device would likely fail
2. Reduce false postives & false negatives

## TO RUN
- Use pyenv
- Run ```$ pipenv install``` to create pipenv to run
- Run ```jupyter lab``` or ```jupyter notebook```
- Open Analytics.ipynb

## Inside notebook
- A RandomForest & (Ada/XG) Boost models with print out of the results at each section
- The models were minimally tweaked using imbalance module to reduce the dominant classification data size to 100~50% of the minor classification as to make the f1_score (rather than to improve accuracy), as "An Ounce of Prevention is Worth a Pound of Cure (or Failure time)"
- This is usually because modern manufacturing or systems require extremely high availability
- The notebook contains quite a bit of documentation but likely need more
- One of the critical Q to answer is when should one perform the prevention. This should be done after model is tuned.
- Hyperparameter tunning or factor selection had not been done due to time constraints. Currently too many variables are dumped into the model & regularization or other fine tuning had not been performed. This takes considerable time & effort & just had not had the time.
- *** Upto now the visualization is not included in the project Hope to add later***

### Below are Some thoughts which were written down prior to starting for my own notes
- 1st Project: I do NOT know Urdu. So, I have no verification of correct translation rather than via Google or Bixby Translation. And other language resources were extremely difficult to find on my last try few years ago. Although it would be fun to try LSTM via Tensorflow, attempting the double unknown would be too extreme.
- 2nd Project: This would likely be some combination of MNIST dataset &/or sign/road identification typically done in either SVN or CNN networks. I have done this before with Keras/Tensorflow combination successfully. So, it would be easiest of 3 but least challenging.
- 3rd Project: This is imbalanced classification problem & my next project will have a portion in this subject matter. So, this was chosen, as I wanted / could use the practice.

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
