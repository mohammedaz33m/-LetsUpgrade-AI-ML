# Movie Recommender System

- __Collaborative__ filtering based on majority interest of the users.


- __Content based__ filtering/ individual tracking of a user.

DATA: 
Genere - __.Json type__

Movie Recommendation using EDA

___Score willl be created & based on that the recommendation of top 10 high score will be shown to the user___

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```


## Data set
![Screenshot](https://github.com/mohammedaz33m/-LetsUpgrade-AI-ML/blob/master/Project_5_movie_recommender%20_system/images/data_df.JPG)

### Data info 
![Screenshot](https://github.com/mohammedaz33m/-LetsUpgrade-AI-ML/blob/master/Project_5_movie_recommender%20_system/images/df_info.JPG)

![Screenshot](https://github.com/mohammedaz33m/-LetsUpgrade-AI-ML/blob/master/Project_5_movie_recommender%20_system/images/df_describe.JPG)

# Univariate Analysis

![Screenshot](https://github.com/mohammedaz33m/-LetsUpgrade-AI-ML/blob/master/Project_5_movie_recommender%20_system/images/df_hist1.JPG)

![Screenshot](https://github.com/mohammedaz33m/-LetsUpgrade-AI-ML/blob/master/Project_5_movie_recommender%20_system/images/df_hist2.JPG)

![Screenshot](https://github.com/mohammedaz33m/-LetsUpgrade-AI-ML/blob/master/Project_5_movie_recommender%20_system/images/df_box1.JPG)

Here we can observe outliers in the data set

![Screenshot](https://github.com/mohammedaz33m/-LetsUpgrade-AI-ML/blob/master/Project_5_movie_recommender%20_system/images/df_box2.JPG)

# Building A Recommender System


![Screenshot](https://github.com/mohammedaz33m/-LetsUpgrade-AI-ML/blob/master/Project_5_movie_recommender%20_system/images/list1.JPG)

![Screenshot](https://github.com/mohammedaz33m/-LetsUpgrade-AI-ML/blob/master/Project_5_movie_recommender%20_system/images/plot1.JPG)

## Above list is the list of moies recommended without treating the null values & outliers in the data set

# Treating the missing values & outliers

data set with null values 

![Screenshot](https://github.com/mohammedaz33m/-LetsUpgrade-AI-ML/blob/master/Project_5_movie_recommender%20_system/images/df_null.JPG)

```python
df[['runtime','vote_average','vote_count']] = df[['runtime','vote_average','vote_count']].replace(0,nan)
df.dropna(inplace=True)
df.shape
```

![Screenshot](https://github.com/mohammedaz33m/-LetsUpgrade-AI-ML/blob/master/Project_5_movie_recommender%20_system/images/df_nonull.JPG)


## Plotting the graphs as above for this data without null values

![Screenshot](https://github.com/mohammedaz33m/-LetsUpgrade-AI-ML/blob/master/Project_5_movie_recommender%20_system/images/df_hist3.JPG)

![Screenshot](https://github.com/mohammedaz33m/-LetsUpgrade-AI-ML/blob/master/Project_5_movie_recommender%20_system/images/df_hist4.JPG)

### Box Plots

![Screenshot](https://github.com/mohammedaz33m/-LetsUpgrade-AI-ML/blob/master/Project_5_movie_recommender%20_system/images/df_box3.JPG)

![Screenshot](https://github.com/mohammedaz33m/-LetsUpgrade-AI-ML/blob/master/Project_5_movie_recommender%20_system/images/df_box4.JPG)

### List of movies

![Screenshot](https://github.com/mohammedaz33m/-LetsUpgrade-AI-ML/blob/master/Project_5_movie_recommender%20_system/images/list2.JPG)

![Screenshot](https://github.com/mohammedaz33m/-LetsUpgrade-AI-ML/blob/master/Project_5_movie_recommender%20_system/images/plot2.JPG)


## Thus we can observe that, there is a difference in the list of movies recommended for the data set with & without null values. 


