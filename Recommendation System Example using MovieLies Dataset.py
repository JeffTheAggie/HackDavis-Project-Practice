#!/usr/bin/env python
# coding: utf-8

# In[59]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#Reading the big_movies.csv
df_movies = pd.read_csv("big_movies.csv")
df_movies.head()


# In[60]:


#Reading the big_ratings.csv
df_ratings = pd.read_csv("big_ratings.csv")
df_ratings.head()


# In[61]:


#big_ratings.csv and big_movies.csv have the same movieId coloumn so we should merge both csv files by that column
df = pd.merge(df_movies, df_ratings, on = "movieId")
df.head()


# In[62]:


#Info on the dataset of df
df.describe()


# In[63]:


#Creating a Data Frame on the average rating per movie
ratings = pd.DataFrame(df.groupby("title")["rating"].mean())
ratings.head()


# In[64]:


#Creating a Data Frame that finds the number of ratings per movie
ratings["number_of_ratings"] = df.groupby("title")["rating"].count()
ratings.head()


# In[65]:


#Histogram on the average rating per movie
ratings["rating"].hist(bins = 50)


# In[66]:


#Histogram on the number of ratings per movie
ratings["number_of_ratings"].hist(bins = 60)


# In[67]:


#Finding the Linear Relationship between the average rating and the number of ratings per movie
sns.jointplot(x = "rating", y = "number_of_ratings", data = ratings)


# In[72]:


#Building the Recommendation System
movie_matrix = df.pivot_table(index = "userId", columns = "title", values = "rating")
movie_matrix.head()


# In[69]:


#Sorting the most rated movies 
ratings.sort_values("number_of_ratings", ascending = False).head(10)


# In[77]:


#Recommending Forrest Gump to the user based on the user's watch history
FG_user_rating = movie_matrix["Forrest Gump (1994)"]
FG_user_rating.head()


# In[78]:


#Recommeding Pulp Fiction to the user based on the user's watch history
PF_user_rating = movie_matrix["Pulp Fiction (1994)"]
PF_user_rating.head()


# In[76]:


#Correlating Forrest Gump's ratings compared to other movies
similar_to_Forrest_Gump = movie_matrix.corrwith(FG_user_rating)
similar_to_Forrest_Gump.head()


# In[75]:


#Correlating Pulp Fiction's ratings compared to other movies
similar_to_Pulp_Fiction = movie_matrix.corrwith(PF_user_rating)
similar_to_Pulp_Fiction.head()


# In[81]:


#Movie Matrix had to many NaN values so we have to transform the correlation results into dataframes(Forrest Gump Ratings)

corr_FG = pd.DataFrame(similar_to_Forrest_Gump, columns = ["correlation"])
corr_FG.dropna(inplace = True)
corr_FG.head()


# In[79]:


#Movie Matrix had to many NaN values so we have to transform the correlation results into dataframes(Pulp Fiction Ratings)
corr_PF = pd.DataFrame(similar_to_Pulp_Fiction, columns = ["correlation"])
corr_PF.dropna(inplace = True)
corr_PF.head()


# In[82]:


#Setting Threshold for the number of ratings since of the movies have very few ratings(Forrest Gump Ratings)
corr_FG = corr_FG.join(ratings["number_of_ratings"])
corr_FG.head()


# In[83]:


#Setting Threshold for the number of ratings since of the movies have very few ratings(Pulp Fiction Ratings)
corr_PF = corr_PF.join(ratings["number_of_ratings"])
corr_PF.head()


# In[84]:


#Obtaining the movies that are similar to Forrest Gump
corr_FG[corr_FG["number_of_ratings"] > 100].sort_values(by = "correlation", ascending = False).head(10)


# In[85]:


#Obtaining the movies that are similar to Pulp Fiction
corr_PF[corr_PF["number_of_ratings"] > 100].sort_values(by = "correlation", ascending = False).head(10)

