#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns


# In[4]:


value = np.random.normal(loc=70,scale=3,size=100)
sns.distplot(value)


# In[5]:


value = np.random.normal(loc=64.5,scale=2.5,size=100)
sns.distplot(value)


# In[6]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[9]:


df = pd.read_csv("C:\\Users\\ajith\\Downloads\\weight-height.csv")


# In[11]:


df.head()


# In[12]:


df.describe()


# In[13]:


plt.style.use('ggplot')

# Histogram of the height
df.Height.plot(kind='hist',color='blue',edgecolor='black',figsize=(10,7))
plt.title('Distribution of Height', size=24)
plt.xlabel('Height (inches)', size=18)
plt.ylabel('Frequency', size=18)


# In[14]:


df.Weight.plot(kind='hist',color='blue',edgecolor='black',figsize=(10,7))
plt.title('Distribution of Weight', size=24)
plt.xlabel('Weight (pounds)', size=18)
plt.ylabel('Frequency', size=18);


# In[15]:


# Stats male
statistics_male = df[df['Gender']=='Male'].describe()
statistics_male.rename(columns=lambda x:x+'_male',inplace=True)

# Stats male
statistics_female = df[df['Gender']=='Female'].describe()
statistics_female.rename(columns=lambda x:x+'_female',inplace=True)

# Dataframe that contains statistics for both male and female
statistics = pd.concat([statistics_male,statistics_female], axis=1)
statistics


# In[18]:


# Scatter plots to view the co-relation

df_males = df[df['Gender']=='Male']
df_females = df[df['Gender']=='Female']

# Scatter plots.
ax1= df_males.plot(kind='scatter', x='Height',y='Weight', color='blue',alpha=0.5, figsize=(10,7))
df_females.plot(kind='scatter', x='Height',y='Weight', color='magenta',alpha=0.5, figsize=(10,7),ax=ax1)


# In[51]:


##Corelation Matri for Male and Femaile Height and Weight
f,ax=plt.subplots(figsize = (5,5))
sns.heatmap(df.corr(),annot= True,linewidths=0.5,fmt = ".1f",ax=ax)
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.title('Correlation Map')
plt.savefig('graph.png')
plt.show()


# In[30]:


#"Covariance between Height and Weight of Both Male and Female
np.cov(df.Height,df.Weight)
print("Covariance between Height and Weight of Both Male and Female: ",df.Height.cov(df.Weight))


# In[32]:


#Lets Split the data into two different Data frames based on the Gender
df1,df2 = [x for _,x in df.groupby(df['Gender'] == 'Male')]
#Printing Female
df1


# In[39]:


#co-variance between Female height and Weight
np.cov(df1.Height,df1.Weight)
print("Covariance between Height and Weight of Females: ",df1.Height.cov(df1.Weight))


# In[37]:


#Printing Male
df2


# In[41]:


#co-variance between Female height and Weight
np.cov(df2.Height,df2.Weight)
print("Covariance between Height and Weight of Males: ",df2.Height.cov(df2.Weight))


# In[43]:


#Pearson Co-relation:
p1 = df1.loc[:,["Height","Weight"]].corr(method= "pearson")
p2 = df1.Weight.cov(df1.Height)/(df1.Weight.std()*df1.Height.std())
print('Pearson correlation: ')
print(p1)
print('Pearson correlation of Female Height and Weight: ',p2)


# In[45]:


#Pearson Co-relation:
p1 = df2.loc[:,["Height","Weight"]].corr(method= "pearson")
p2 = df2.Weight.cov(df2.Height)/(df2.Weight.std()*df2.Height.std())
print('Pearson correlation: ')
print(p1)
print('Pearson correlation of Male Height and Weight: ',p2)


# In[46]:


p1 = df.loc[:,["Height","Weight"]].corr(method= "pearson")
p2 = df.Weight.cov(df.Height)/(df.Weight.std()*df.Height.std())
print('Pearson correlation: ')
print(p1)
print('Pearson correlation of Male& Female Height and Weight: ',p2)


# In[49]:


f,ax=plt.subplots(figsize = (5,5))
sns.heatmap(df1.corr(),annot= True,linewidths=0.5,fmt = ".1f",ax=ax)
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.title('Correlation Map')
plt.savefig('graph.png')
plt.show()


# In[50]:


f,ax=plt.subplots(figsize = (5,5))
sns.heatmap(df2.corr(),annot= True,linewidths=0.5,fmt = ".1f",ax=ax)
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.title('Male Height and Weight Correlation Map')
plt.savefig('graph.png')
plt.show()


# In[52]:


# Import packages
import numpy as np
from scipy.stats import multivariate_normal

# Prepare your data
x = np.linspace(-10, 10, 500)
y = np.linspace(-10, 10, 500)
X, Y = np.meshgrid(x,y)

# Get the multivariate normal distribution
mu_x = np.mean(x)
sigma_x = np.std(x)
mu_y = np.mean(y)
sigma_y = np.std(y)
rv = multivariate_normal([mu_x, mu_y], [[sigma_x, 0], [0, sigma_y]])

# Get the probability density
pos = np.empty(X.shape + (2,))
pos[:, :, 0] = X
pos[:, :, 1] = Y
pd = rv.pdf(pos)


# In[55]:


print (pd)


# In[73]:


##ASSUMING THE HEIGHTS OF 100 PEOPLE IS BETWEEN 50 AND 80 Inches. Below are the Random numbers.
##It Still looks like a noraml distribution
import seaborn as sns

import matplotlib.pyplot as plt
a = np.random.uniform(50,80, 1000)
sns.histplot(a)
plt.show()


# In[ ]:




