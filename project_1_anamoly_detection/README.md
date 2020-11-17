All datasets stored in this folder are downloaded from 
**Outlier Detection DataSets (ODDS)**: http://odds.cs.stonybrook.edu/#table1

If you use any data here, please see ODDS' citation policy:

Shebuti Rayana (2016).  ODDS Library [http://odds.cs.stonybrook.edu]. Stony Brook, NY: Stony Brook University, Department of Computer Science.


# Anomoly Detection Using Machhine Learning

__Anamoly :__

- Uncertainity 

- Outlier detection 

- Novelty detection (Something new)

- Unusual pattern 

- Inconsistent data points


Time Series Anamoly detection

Video level amanoly detection

Image level anamoly detection
  
  - Out of Distribution (OOD) Detection target
  
  - Anamoly segmentation target
  
  
__PyOD__ python toolkit for detecting outlying objects in multivariate data (Since May 2018)


## Benchmark 

__Linear model for Outlier detection__

- __PCA: Principal component analysis__: Use sum of weighted projected distance to the eigenvector hyperplane as the outlier scores

- __MCD: Minimum co-variance determinent__: minimum diff between SD & each data points.

- __OCSVM: One class- Support Vector machine__: both for Classification | Regression problems. 
   
   OCSVM Not suggested for Non Linear data 

__Proximity based outlier detection Models__: 

- __LOC:__ Local outlier Factor 

- __CBLOF:__ Cluster based local outlier factor

- __KNN: k Nearest Neighbours:__ based on distance to the kth nearest neighbour as the outlier score

- __HBOS:__ Histogram based Outlier Score

__Probabilistic Model:__

- __ABOD__ Angle Based Outlier detection

__Outlier Ensembles & combination Frameworks__: optimisation

- __Isolation Forest__

- __Feature Bagging__



## Import packages

```python
import os
import sys # to Load our files
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn .model_selection import train_test_split # to split the dataset
from scipy.io import loadmat # to load matlab files
```

## Import PyOD packages & methods

```python 
from pyod.models.pca import PCA       # Principal component analysis
from pyod.models.mcd import MCD       # Minimum co-variance determinent
from pyod.models.ocsvm import OCSVM   # One class- Support Vector machine
from pyod.models.lof import LOF       # Local outlier Factor
from pyod.models.cblof import CBLOF   # Cluster based local outlier factor
from pyod.models.knn import KNN       # k Nearest Neighbours
from pyod.models.hbos import HBOS     # Histogram based Outlier Score
from pyod.models.abod import ABOD     # Angle based outlier detection
from pyod.models.iforest import IForest 
from pyod.models.feature_bagging import FeatureBagging 
```

## import performance Metric Package
```python 
from pyod.utils.utility import standardizer
from pyod.utils.utility import precision_n_scores
from sklearn.metrics import roc_auc_score # to evaluate classification model performane
```

# ROC Data 

![Screenshot](https://github.com/mohammedaz33m/-LetsUpgrade-AI-ML/blob/master/project_1_anamoly_detection/images/ROC.JPG)

# Precision Data 

![Screenshot](https://github.com/mohammedaz33m/-LetsUpgrade-AI-ML/blob/master/project_1_anamoly_detection/images/precision.JPG)


# Time Data 

![Screenshot](https://github.com/mohammedaz33m/-LetsUpgrade-AI-ML/blob/master/project_1_anamoly_detection/images/time.JPG)


# Model Comparision

![Screenshot](https://github.com/mohammedaz33m/-LetsUpgrade-AI-ML/blob/master/project_1_anamoly_detection/images/model_comparision.JPG)
