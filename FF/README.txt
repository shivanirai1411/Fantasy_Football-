DEEP LEARNING BASELINE

  The dl_baseline/dl_baseline.ipynb notebook has two functions of note, predict and compute score:

  (squad, candidates, captain, vice_captain, priorities) = predict(2019, 1)
  compute_score(squad, candidates, captain, vice_captain, priorities, 2019, 1)

  You might want to use GPU hardware acceleration. It takes about 10 minutes to train the model with default arguments.

  Predict has more options; please see the predict package at./fpl_prediction/dl_baseline/predict for more details, especially the predict.py module.

  The predict folder is a python package containing four modules; predict, optimize, model, and data.

  The predict.py module contains the two main functions of interest to the user, namely predict and compute score.
  The optimize.py module implements the linear programming solver.
  The model.py module implements the deep learning model and training loop.
  The data.py module contains functions for fetching, cleaning and preparing the data for training.

#non_dl_baseline: code for the non-deep learning basline
#To run the code: run non_dl_baseline.ipynb, run each cell from top to bottom


#adv_dl_model: code for the advanced deep learning model
#To run the code: run AdvancedModels.ipynb, run each cell from top to bottom, each model is in thier own training loop

