# Databricks notebook source
# code is adapted from this source: https://gist.github.com/karamanbk/962443877d629713e0e410d52443c7d6
# For more information see this Medium post: https://towardsdatascience.com/data-driven-growth-with-python-part-2-customer-segmentation-5c019d150444

# COMMAND ----------

# MAGIC %pip install lifetimes nbconvert altair

# COMMAND ----------

from datetime import datetime, timedelta
import pandas as pd
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from __future__ import division

import altair as alt

# COMMAND ----------

# MAGIC %md 
# MAGIC RFM stands for Recency - Frequency - Monetary Value. 
# MAGIC 
# MAGIC Theoretically we will have segments like below:
# MAGIC 
# MAGIC - **Low Value**: Customers who are less active than others, not very frequent buyer/visitor and generates very low - zero - maybe negative revenue.  
# MAGIC 
# MAGIC - **Mid Value**: In the middle of everything. Often using our platform (but not as much as our High Values), fairly frequent and generates moderate revenue.
# MAGIC 
# MAGIC - **High Value**: The group we don’t want to lose. High Revenue, Frequency and low Inactivity.
# MAGIC 
# MAGIC We can use Recency, Frequency and Monetary Value to identify which customers should fall into the Low Value, Mid Value and High Value categories

# COMMAND ----------

orders_pd = pd.read_csv("/dbfs/FileStore/jeanne_choo@databricks.com/gaming.csv")
orders_pd.head()

# COMMAND ----------

orders_pd.dtypes

# COMMAND ----------

# MAGIC %md
# MAGIC # Data Cleaning
# MAGIC - Convert our `InvoiceDate` column to `datetime` format
# MAGIC - Convert `CustomerID` from `float` to `str`
# MAGIC - Remove `nan` values from `CustomerID`

# COMMAND ----------

orders_pd["InvoiceDate"] = pd.to_datetime(orders_pd["InvoiceDate"])

# COMMAND ----------


orders_pd["CustomerID"] = orders_pd["CustomerID"].astype(str)

# COMMAND ----------

orders_pd = orders_pd[orders_pd["CustomerID"]!="nan"]

# COMMAND ----------

# MAGIC %md 
# MAGIC # RFM Analysis

# COMMAND ----------

#create a generic user dataframe to keep CustomerID and new segmentation scores
tx_user = pd.DataFrame(orders_pd['CustomerID'].unique())
tx_user.columns = ['CustomerID']

# COMMAND ----------

# MAGIC %md
# MAGIC ## Recency
# MAGIC We group the data by `CustomerID` and find the latest date that they last transacted with us

# COMMAND ----------

tx_max_purchase = orders_pd.groupby('CustomerID').InvoiceDate.max().reset_index()
tx_max_purchase.columns = ['CustomerID','MaxPurchaseDate']

# COMMAND ----------

# we take the most recent date in the dataset as the latest date
# latest date - latest purchase date from a customer will tell us how recently a customer engaged with us
tx_max_purchase['Recency'] = (tx_max_purchase['MaxPurchaseDate'].max() - tx_max_purchase['MaxPurchaseDate']).dt.days

# COMMAND ----------

#merge this dataframe to our new user dataframe
tx_user = pd.merge(tx_user, tx_max_purchase[['CustomerID','Recency']], on='CustomerID')

# COMMAND ----------

tx_user = tx_user.sort_values(by="Recency", ascending=False)
tx_user.head()

# COMMAND ----------

# MAGIC %md 
# MAGIC If we plot the recency of transactions as a histogram, we see that the data has a long tail. 
# MAGIC 
# MAGIC This means that most customers have transacted in the last 50 days, while many others have much longer last transacted dates. 

# COMMAND ----------

alt.Chart(tx_user.iloc[0:5000,:]).mark_bar().encode(
  alt.X("Recency", bin=True),
  y="count()"
)

# COMMAND ----------

# MAGIC %md 
# MAGIC To further group our customers by `Recency`, we can use a `Kmeans` model to classify customers into segments. From the elbow plot below, we can see that 4 is probably the optimal number of groups we should cluster customers into. 

# COMMAND ----------

from sklearn.cluster import KMeans

sse={}
tx_recency = tx_user[['Recency']]
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, max_iter=1000).fit(tx_recency)
    tx_recency["clusters"] = kmeans.labels_
    sse[k] = kmeans.inertia_ 
plt.figure()
plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel("Number of cluster")
plt.show()

# COMMAND ----------

kmeans = KMeans(n_clusters=4)
kmeans.fit(tx_user[['Recency']])
tx_user['RecencyCluster'] = kmeans.predict(tx_user[['Recency']])

# COMMAND ----------

# MAGIC %md 
# MAGIC Getting summary statistics from our customers shows us 
# MAGIC - How many customers fall into each cluster 
# MAGIC - What the `recency` value for each cluster is like

# COMMAND ----------

tx_user.groupby('RecencyCluster')['Recency'].describe()

# COMMAND ----------

def order_cluster(cluster_field_name, target_field_name,df,ascending):
    new_cluster_field_name = 'new_' + cluster_field_name
    df_new = df.groupby(cluster_field_name)[target_field_name].mean().reset_index()
    df_new = df_new.sort_values(by=target_field_name,ascending=ascending).reset_index(drop=True)
    df_new['index'] = df_new.index
    df_final = pd.merge(df,df_new[[cluster_field_name,'index']], on=cluster_field_name)
    df_final = df_final.drop([cluster_field_name],axis=1)
    df_final = df_final.rename(columns={"index":cluster_field_name})
    return df_final

# COMMAND ----------

tx_user = order_cluster('RecencyCluster', 'Recency',tx_user,False)

# COMMAND ----------

tx_user.head()

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Frequency

# COMMAND ----------

#get order counts for each user and create a dataframe with it
tx_frequency = orders_pd.groupby('CustomerID').InvoiceDate.count().reset_index()
tx_frequency.columns = ['CustomerID','Frequency']
tx_frequency = tx_frequency.sort_values(by="Frequency", ascending=False)

# COMMAND ----------

alt.Chart(tx_frequency.iloc[0:5000,:]).mark_bar().encode(
  alt.X("Frequency", bin=True),
  y="count()"
)

# COMMAND ----------

#add this data to our main dataframe
tx_user = pd.merge(tx_user, tx_frequency, on='CustomerID')
tx_user.head()

# COMMAND ----------

#k-means
kmeans = KMeans(n_clusters=4)
kmeans.fit(tx_user[['Frequency']])
tx_user['FrequencyCluster'] = kmeans.predict(tx_user[['Frequency']])

#order the frequency cluster
tx_user = order_cluster('FrequencyCluster', 'Frequency',tx_user,True)

#see details of each cluster
tx_user.groupby('FrequencyCluster')['Frequency'].describe()

# COMMAND ----------

orders_pd['Revenue'] = orders_pd['UnitPrice'] * orders_pd['Quantity']
tx_revenue = orders_pd.groupby('CustomerID').Revenue.sum().reset_index()
tx_revenue = tx_revenue.sort_values(by="Revenue", ascending=False)
tx_revenue.head()

# COMMAND ----------

tx_user = pd.merge(tx_user, tx_revenue, on='CustomerID')

# COMMAND ----------

alt.Chart(tx_revenue.iloc[0:5000,:]).mark_bar().encode(
  alt.X("Revenue", bin=True),
  y="count()"
)

# COMMAND ----------

tx_user

# COMMAND ----------

sse={}
tx_revenue = tx_user[['Revenue']]
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, max_iter=1000).fit(tx_revenue)
    tx_revenue["clusters"] = kmeans.labels_
    sse[k] = kmeans.inertia_ 
plt.figure()
plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel("Number of cluster")
plt.show()

# COMMAND ----------

kmeans = KMeans(n_clusters=4)
kmeans.fit(tx_user[['Revenue']])
tx_user['RevenueCluster'] = kmeans.predict(tx_user[['Revenue']])

# COMMAND ----------

tx_user = order_cluster('RevenueCluster', 'Revenue',tx_user,True)

# COMMAND ----------

tx_user.groupby('RevenueCluster')['Revenue'].describe()

# COMMAND ----------

# MAGIC %md 
# MAGIC # Overall Segmentation

# COMMAND ----------

tx_user.head()
tx_user['OverallScore'] = tx_user['RecencyCluster'] + tx_user['FrequencyCluster'] + tx_user['RevenueCluster']
tx_user.groupby('OverallScore')['Recency','Frequency','Revenue'].mean()

# COMMAND ----------

tx_user['Segment'] = 'Low-Value'
tx_user.loc[tx_user['OverallScore']>2,'Segment'] = 'Mid-Value' 
tx_user.loc[tx_user['OverallScore']>4,'Segment'] = 'High-Value' 

# COMMAND ----------

tx_graph = tx_user.query("Revenue < 50000 and Frequency < 2000")

# COMMAND ----------

# MAGIC %md 
# MAGIC Overall, out RFM analysis can successfully separate customers into Low, Mid and High value segments

# COMMAND ----------

alt.Chart(tx_graph).mark_circle().encode(
  alt.X("Frequency"),
  alt.Y("Revenue"), 
  color="Segment"
)

# COMMAND ----------

alt.Chart(tx_graph).mark_circle().encode(
  alt.X("Recency"),
  alt.Y("Revenue"), 
  color="Segment"
)

# COMMAND ----------

alt.Chart(tx_graph).mark_circle().encode(
  alt.X("Recency"),
  alt.Y("Frequency"), 
  color="Segment"
)

# COMMAND ----------


