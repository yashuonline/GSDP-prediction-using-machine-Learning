#!/usr/bin/env python
# coding: utf-8

# In[23]:


# Import the required Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
from functools import reduce
from itertools import cycle, islice
pd.options.display.float_format='{:.4f}'.format
plt.rcParams['figure.figsize'] = [11.5,8]
pd.set_option('display.max_colwidth', 1000)  # Example width
pd.set_option('display.max_colwidth', None)


# In[2]:


# File path. 
df_1a_original = pd.read_csv(r"C:\Users\yaswa\Downloads\GSDP_ML.csv")
df_1a_original.head()


# In[5]:


# shape of data
df_1a_original.shape


# In[6]:


df_1a_original.describe()


# In[8]:


# Data Information
df_1a_original.info()


# In[9]:


## Data Cleansing and Preparation


# In[10]:


# Calculating the Missing Values % contribution in DF
df_1a_original_null=df_1a_original.isna().mean().round(4) * 100
df_1a_original_null


# In[11]:


# Dropping columns where all rows are NaN
df_1a_originalx1 = df_1a_original.dropna(axis = 1, how = 'all')


# In[12]:


# Dropping the data for Duration 2016-17 as it will not be used in Analysis

df_1a_originalx2 = df_1a_originalx1[df_1a_originalx1.Duration != '2016-17']


# In[13]:


# Dropping the UT as it is not needed for Analysis

df_1a_originalx3 = df_1a_originalx2.T
df_1a_originalx4 = df_1a_originalx3.drop(labels = ['Andaman & Nicobar Islands','Chandigarh','Delhi','Puducherry'])
#dfx3


# In[14]:


# Mean of the row (% Growth over previous year) for duration 2013-14, 2014-15 and 2015-16

df_1a_originalx4_mean = df_1a_originalx4.iloc[2:, 6:10].mean(axis=1).fillna(0).round(2).sort_values()
df_1a_originalx4_mean


# In[15]:


## Data Visualization and Insights Extraction


# In[16]:


# Bar Plot for Average growth rates of the various states for duration 2013-14, 2014-15 and 2015-16

plt.rcParams['figure.figsize'] = [11.5,8]
df_1a_originalx4_mean.plot(kind='barh',stacked=True, colormap = 'Set1')
plt.title("Avg.% Growth of States for Duration 2013-14, 2014-15 and 2015-16", fontweight = 'bold')
plt.xlabel("Avg. % Growth", fontweight = 'bold')
plt.ylabel("States", fontweight = 'bold')


# In[17]:


# Average growth rate of my home state against the National average Growth rate

df_1a_originalx4_myhome = df_1a_originalx4_mean[['Andhra Pradesh', 'All_India GDP']]


# In[18]:


df_1a_originalx4_myhome.plot(kind='bar',stacked=True, colormap = 'Dark2')
plt.title("Avg. % Growth of Home State vs National Avg. for Duration 2013-14, 2014-15 and 2015-16", fontweight = 'bold')
plt.ylabel("Average % Growth", fontweight = 'bold')
plt.xlabel("Home State Vs National Average", fontweight = 'bold')


# In[19]:


#Selecting the GSDP for year 2015-16

df_1a_originalx5_total_gdp = df_1a_originalx4.iloc[2:,4:5]


# In[20]:


# Dropping the GSDP of All_India as it will not be included in the plot

df_1a_originalx6_total_gdp = df_1a_originalx5_total_gdp.drop(labels = ['All_India GDP'])


# In[21]:


#Plot for GSDP of all states including States with NaN
df_1a_originalx6_total_gdp[4] = pd.to_numeric(df_1a_originalx6_total_gdp[4], errors='coerce')
df_1a_originalx6_total_gdp.sort_values(by=4, inplace=True)
df_1a_originalx6_total_gdp.sort_values(by=4).plot(kind='bar',stacked=True, colormap = 'Set1')
plt.title("Total GDP of States for duration 2011-24" , fontweight = 'bold')
plt.ylabel("Total GDP (in cr)",fontweight = 'bold')
plt.xlabel("States",fontweight = 'bold')


# In[22]:


# Dropping the States whose GSDP in NaN for year 2015-16

df_1a_originalx7_total_gdp = df_1a_originalx6_total_gdp.dropna().sort_values(by = 4)


# In[20]:


#Plot for GSDP of all states excluding States with NaN

df_1a_originalx7_total_gdp.plot(kind='bar',stacked=True, colormap = 'autumn')
plt.title("Total GDP of States for duration 2015-16" , fontweight = 'bold')
plt.ylabel("Total GDP (in cr)",fontweight = 'bold')
plt.xlabel("States",fontweight = 'bold')


# In[21]:


df_1a_originalx7_total_gdp.shape


# In[22]:


# GSDP of Top 5 States
df_1a_originalx7_total_gdp.tail(5).plot(kind='bar',stacked=True, colormap = 'Dark2')
plt.title("Total GDP of top 5 States for 2015-16", fontweight = 'bold')
plt.ylabel("Total GDP (in cr)",fontweight = 'bold')
plt.xlabel("States",fontweight = 'bold')


# GSDP of Bottom 5 States
df_1a_originalx7_total_gdp.head(5).plot(kind='bar',stacked=True, colormap = 'Set1')
plt.title("Total GDP of bottom 5 States for 2015-16", fontweight = 'bold')
plt.ylabel("Total GDP (in cr)",fontweight = 'bold')
plt.xlabel("States",fontweight = 'bold')


# In[23]:


# Reading all the csv files using glob functionality from a directory for further analysis

import pandas as pd
import glob

path = r'C:\Users\yaswa\Downloads\N'  # Define path properly
files = glob.glob(path + "/*.csv")  # Ensure files are selected

data = pd.DataFrame()

for f in files:
    dfs = pd.read_csv(f, encoding='unicode_escape')
    dfs['State'] = f.replace(path, '').replace('NAD-', '').replace('-GSVA_cur_2016-17.csv', '') \
                    .replace('-GSVA_cur_2015-16.csv', '').replace('-GSVA_cur_2014-15.csv', '').replace('_', ' ')
    data = pd.concat([data, dfs], ignore_index=True)  # Use concat instead of append

data = data.iloc[:, ::-1]  # Reverse columns if needed
data.sort_values(by="State", inplace=True)  # Sort properly if required


# In[24]:


# Selecting the required columns for the Analysis

df_1a_original = data[['State', 'Item', '2014-15']] 
df_1a_original1 = df_1a_original.reset_index(drop = True)


# In[25]:


# Cleansing the columns name

df_1a_original1['Item'] = df_1a_original1['Item'].map(lambda x: x.rstrip('*') if isinstance(x, str) else x)
df_1a_original1 = df_1a_original1.set_index('State')


# In[26]:


# Pivoting the df for enhanced analysis of data

df_1a_original2 = pd.pivot_table(df_1a_original1, values = '2014-15', index=['Item'], columns = 'State').reset_index()
df_1a_original3 = df_1a_original2.set_index('Item',drop=True)
#df3


# In[27]:


# Dropping the UT as it will not be used in further analysis

df_1a_original4 = df_1a_original3.drop(['Andaman Nicobar Islands', 'Chandigarh', 'Delhi', 'Puducherry'], axis=1, errors='ignore')


# In[28]:


df_1a_original5_percapita = df_1a_original4.loc['Per Capita GSDP (Rs.)'].sort_values()


# In[29]:


#Plot for GDP per capita in Rs. for all states

df_1a_original5_percapita.plot(kind='barh',stacked=True, colormap = 'gist_rainbow')
plt.title("GDP per Capita for All States for duration 2014-15", fontweight = 'bold')
plt.xlabel("GDP per Capita (in Rs.)",fontweight = 'bold')
plt.ylabel("States", fontsize = 12, fontweight = 'bold')


# In[30]:


#Plot for GDP per Capita of top 5 States for 2014-15

df_1a_original5_percapita.tail(5).plot(kind='bar',stacked=True, colormap = 'winter')
plt.title("GDP per Capita of top 5 States for 2014-15", fontweight = 'bold')
plt.ylabel("GDP per Capita (in Rs.)", fontweight = 'bold')
plt.xlabel("States", fontsize = 12, fontweight = 'bold')


# In[31]:


#Plot for GDP per Capita of bottom 5 States for 2014-15

df_1a_original5_percapita.head(5).plot(kind='bar',stacked=True, colormap = 'Set1')
plt.title("GDP per Capita of bottom 5 States for 2014-15", fontweight = 'bold')
plt.ylabel("GDP per Capita (in Rs.)", fontweight = 'bold')
plt.xlabel("States", fontweight = 'bold')


# In[32]:


def safe_get(series, index_name):
    return series.loc[index_name] if index_name in series.index else None

Goa_percapita = safe_get(df_1a_original5_percapita, 'Goa') / df_1a_original5_percapita.sum() * 100 if 'Goa' in df_1a_original5_percapita.index else None
Sikkim_percapita = safe_get(df_1a_original5_percapita, 'Sikkim') / df_1a_original5_percapita.sum() * 100 if 'Sikkim' in df_1a_original5_percapita.index else None
Bihar_percapita = safe_get(df_1a_original5_percapita, 'Bihar') / df_1a_original5_percapita.sum() * 100 if 'Bihar' in df_1a_original5_percapita.index else None
UP_percapita = safe_get(df_1a_original5_percapita, 'Uttar Pradesh') / df_1a_original5_percapita.sum() * 100 if 'Uttar Pradesh' in df_1a_original5_percapita.index else None


# In[33]:


# Ratio of the highest per capita GDP to the lowest per capita GDP

h_percapita = df_1a_original5_percapita.iloc[-1]
l_percapita = df_1a_original5_percapita.iloc[0]
percapita_ratio = (h_percapita/l_percapita).round(3)

percapita_ratio


# In[34]:


# Selecting Primary Secondary and Tertiary sector for percentage contribution in total GDP

df_1a_original_gdp_con = df_1a_original4.loc[['Primary', 'Secondary', 'Tertiary','Gross State Domestic Product']]
df_1a_original_gdp_percon = (df_1a_original_gdp_con.div(df_1a_original_gdp_con.loc['Gross State Domestic Product'])*100).round(2)
df_1a_original_gdp_percon =df_1a_original_gdp_percon.T.iloc[:,:3]


# In[35]:


# Plot for % contribution of sectors in total GDP

df_1a_original_gdp_percon.plot(kind='bar',stacked=True, colormap = 'prism')
plt.title("% Contribution of Primary, Secondary, Tertiary sector in total GDP for 2014-15",fontweight = 'bold')
plt.ylabel("% Contribution", fontweight = 'bold')
plt.xlabel("States", fontweight = 'bold')


# In[36]:


# Sorting the df for better visualization

df_1a_original_sort = df_1a_original4.T.sort_values(by = 'Per Capita GSDP (Rs.)', ascending = False)
df_1a_original_sort 


# In[37]:


# Define the quantile values and bins for categorisation

df_1a_original_sort.quantile([0.2,0.5,0.85,1], axis = 0)
bins = [0, 67385, 101332, 153064.85, 271793]
labels = ["C4", "C3", "C2", "C1"]
df_1a_original_sort['Category'] = pd.cut(df_1a_original_sort['Per Capita GSDP (Rs.)'], bins = bins, labels = labels)
df_1a_original_index = df_1a_original_sort.set_index('Category')
df_1a_original_sum = df_1a_original_index.groupby(['Category']).sum()
df_1a_original_rename =  df_1a_original_sum.rename(columns = {"Population ('00)" : "Population (00)"})


# In[38]:


# Selecting the sub sectors which will be used for further analysis

df_1a_original7_sector = df_1a_original_rename[['Agriculture, forestry and fishing','Mining and quarrying','Manufacturing','Electricity, gas, water supply & other utility services',
                 'Construction','Trade, repair, hotels and restaurants','Transport, storage, communication & services related to broadcasting','Financial services',
                'Real estate, ownership of dwelling & professional services','Public administration','Other services','Gross State Domestic Product']]


# In[39]:


# Calculating and rounding the percentage contribution of each subsector in total GSDP

df_1a_original8_per = (df_1a_original7_sector.T.div(df_1a_original7_sector.T.loc['Gross State Domestic Product'])*100)
df_1a_original8_round = df_1a_original8_per.round(2)
df_1a_original9_per = df_1a_original8_round.drop('Gross State Domestic Product')
df_1a_original9_per


# In[40]:


# Plot for % Contribution of subsectors in Total GDP for C1 states for 2014-15

df_1a_original9_per['C1'].sort_values().plot(kind='bar',stacked=True, colormap = 'Accent')
plt.title("% Contribution of subsectors in Total GDP for C1 states for 2014-15", fontweight = 'bold')
plt.xlabel("Sub-sectors", fontweight = 'bold')
plt.ylabel("% Contribution", fontweight = 'bold')


# In[41]:


# Plot for % Contribution of subsectors in Total GDP for C2 states for 2014-15

df_1a_original9_per['C2'].sort_values().plot(kind='bar',stacked=True, colormap = 'Accent')
plt.title("% Contribution of subsectors in Total GDP for C2 states for 2014-15", fontweight = 'bold')
plt.ylabel("% Contribution", fontweight = 'bold')
plt.xlabel("Sub-sectors", fontweight = 'bold')


# In[42]:


# Plot for % Contribution of subsectors in Total GDP for C3 states for 2014-15

df_1a_original9_per['C3'].sort_values().plot(kind='bar',stacked=True, colormap = 'Accent')
plt.title("% Contribution of subsectors in Total GDP for C3 states for 2014-15", fontweight = 'bold')
plt.ylabel("% Contribution", fontweight = 'bold')
plt.xlabel("Sub-sectors", fontweight = 'bold')


# In[44]:


### Plot for top 3/4/5/6 sub-sectors that contribute to approximately 80% of the GSDP of each category.


# In[87]:


# Selecting the columns which will be used for further analysis
df_1a_originala = df_1a_original_rename[['Level of Education - State','Primary - 2014-2015','Upper Primary - 2014-2015','Secondary - 2014-2015']] 


# In[86]:


# Renaming the columns which are incorrect

df_1a_original_rename = df_1a_original_dropout.rename(columns = {'Primary - 2014-2015' : 'Primary - 2013-2014','Primary - 2014-2015.1' : 'Primary - 2014-2015'})


# In[89]:


# Dropping the union territory because it will not be used in further analysis

df_1a_originala1 = df_1a_originala.drop([0, 5, 7, 8, 9, 18, 26, 35, 36])
df_1a_originala2 = df_1a_originala1.reset_index(drop=True)


# In[90]:


# Calculating the Missing Values % contribution in DF

df_1a_originala2.isna().mean().round(2) * 100


# In[91]:


# Selecting the required column for further analysis

df_1a_originala3 = df_1a_original4.T.reset_index()
df_1a_originala4 = df_1a_originala3[['State', 'Per Capita GSDP (Rs.)']]


# In[93]:


# Concatenating the Education dropout df and Per Capita of States df

df_1a_originala5 = pd.concat([df_1a_originala2, df_1a_originala4], axis = 1)
df_1a_originala6 = df_1a_originala5.drop(['State'], axis = 1) 
df_1a_originala7 = df_1a_originala6.set_index('Level of Education - State', drop = True)


# In[95]:


# Scatter Plot for GDP per capita with dropout rates in education

f = plt.figure()    
f, axes = plt.subplots(nrows = 2, ncols = 2, sharex=True, sharey = False)
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=None)

sc = axes[0][0].scatter(df_1a_originala7['Primary - 2014-2015'],df_1a_originala7['Per Capita GSDP (Rs.)'], s=100, c='DarkRed',marker="o")
axes[0][0].set_ylabel('Per Capita GSDP (Rs.)')
axes[0][0].set_xlabel('Primary Education')

sc = axes[0][1].scatter(df_1a_originala7['Upper Primary - 2014-2015'],df_1a_originala7['Per Capita GSDP (Rs.)'], s=100, c='DarkBlue',marker="*")
axes[0][1].set_ylabel('Per Capita GSDP (Rs.)')
axes[0][1].set_xlabel('Upper Primary Education')

sc = axes[1][0].scatter(df_1a_originala7['Secondary - 2014-2015'],df_1a_originala7['Per Capita GSDP (Rs.)'], s=100, c='DarkGreen',marker="s")
axes[1][0].set_ylabel('Per Capita GSDP (Rs.)')
axes[1][0].set_xlabel('Secondary Education')


# In[97]:


df_1a_originala7.plot(kind='scatter',x='Primary - 2014-2015',y='Per Capita GSDP (Rs.)', s=150, c='DarkRed',marker="o")


# In[98]:


df_1a_originala7.plot(kind='scatter',x='Upper Primary - 2014-2015',y='Per Capita GSDP (Rs.)', s=150, c='DarkRed',marker="*")


# In[99]:


df_1a_originala7.plot(kind='scatter',x='Secondary - 2014-2015',y='Per Capita GSDP (Rs.)', s=150, c='DarkRed',marker="s")

