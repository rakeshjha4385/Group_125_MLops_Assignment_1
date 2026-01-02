# Databricks notebook source
# MAGIC %md
# MAGIC # Content
# MAGIC Time series clustering using K means with Euclidean and DTW distance
# MAGIC
# MAGIC How to decide the number of clusters ?
# MAGIC
# MAGIC How can we calculate performance of clustering ?

# COMMAND ----------

# DBTITLE 1,Install Required Libraries
!pip install --upgrade numpy==1.24.0
!pip install tslearn
!pip install numba==0.56.4

# COMMAND ----------

# DBTITLE 1,Restart Python to get latest version
import numpy as np
dbutils.library.restartPython()
print(np.__version__)


# COMMAND ----------

# MAGIC %md
# MAGIC ## Import Libraries

# COMMAND ----------

# MAGIC %matplotlib inline
# MAGIC from tslearn.clustering import silhouette_score
# MAGIC from sklearn.decomposition import PCA
# MAGIC
# MAGIC import numpy as np
# MAGIC import matplotlib.pyplot as plt
# MAGIC plt.rcParams['figure.figsize'] = [25, 8]
# MAGIC
# MAGIC from tslearn.clustering import TimeSeriesKMeans
# MAGIC from tslearn.datasets import CachedDatasets
# MAGIC from tslearn.preprocessing import TimeSeriesScalerMeanVariance
# MAGIC import pandas as pd

# COMMAND ----------

# MAGIC %md
# MAGIC tslearn is a Python package that provides machine learning tools for the analysis of time series.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Ingestion

# COMMAND ----------

# DBTITLE 1,Create, merge and save POS Data
# select * from dev_rgmx_etl.uk_all_hygiene_dishwash_econ_pos
# -- select * from dev_rgmx_etl.us_all_dishcare_econ_pos
# -- select * from dev_rgmx_etl.uk_all_health_ccfst_econ_pos

# -- select PROMO_SALES, NET_SALES_VOLUME, NET_SALES_WITH_TAXES from dev_rgmx_etl.uk_all_health_surfacecleaners_econ_pos
# -- where PROMO_SALES <> 0
 
# -- # select total_sales, non_promo_sales from dev_rgmx_etl.uk_all_health_surfacecleaners_step_03 limit 10
 
# -- # select rgm_ppg, pos.* from dev_uk_all_health_surfacecleaners_prep.t_pos_transaction pos
# -- # left join
# -- # dev_uk_all_health_surfacecleaners_prep.t_product_master pm
# -- # on pos.ITEM_CODE = pm.item_code

# query = f"select * from dev_rgmx_etl.uk_all_hygiene_dishwash_econ_pos"
# query = f"select * from dev_rgmx_etl.us_all_dishcare_econ_pos"
# query = f"select * from dev_rgmx_etl.uk_all_health_ccfst_econ_pos"


category_list =[
"us_all_aircare_econ_pos",
"us_all_dishcare_econ_pos",
"us_all_health_analgesics_econ_pos",
"us_all_health_swb_econ_pos",
"us_all_hygiene_carpet_econ_pos",
"us_all_hygiene_fabric_econ_pos",
"us_all_immunity_econ_pos",
"us_all_surfacecare_econ_pos",
"us_all_ur_econ_pos"
]

# final_df = pd.DataFrame()
# for c in category_list:
#     query = f"select * from dev_rgmx_etl.{c}"
#     print(query)
#     spark_df = spark.sql(query)
#     temp_df = spark_df.toPandas()
#     temp_df['NON_PROMO_SALES_VOLUME'] = temp_df['NET_SALES_VOLUME'] - temp_df['PROMO_SALES_VOLUME']
#     temp_df['CATEGORY'] = c
    
#     final_df = pd.concat([final_df, temp_df], ignore_index=True)

# # Final combined dataset
# print("Final shape:", final_df.shape)
# display(final_df)

# # final_df.reset_index().to_csv('/dbfs/FileStore/tables/subhash_dixit/baseline_forecast_project/baseline_shape_analysis/us_all_category_pos_all_manuf_data.csv', index=False)


# COMMAND ----------

# DBTITLE 1,Read POS Cleaned Data

# Read the merged pos data
# pos_df_all_manuf = pd.read_csv('/dbfs/FileStore/tables/subhash_dixit/baseline_forecast_project/baseline_shape_analysis/us_all_category_pos_all_manuf_data.csv')
# print(f"pos_df shape: {pos_df_all_manuf.shape}")

# pos_df = pos_df_all_manuf[pos_df_all_manuf['MANUFACTURER_NM'].str.lower().isin(['reckitt', 'reckitt benckiser'])]
# print(f"pos_df shape: {pos_df.shape}")

# Write the merged pos data in dbfs for faster reading
# pos_df_all_manuf.reset_index().to_csv('/dbfs/FileStore/tables/subhash_dixit/baseline_forecast_project/baseline_shape_analysis/us_all_category_pos_all_manuf_data.csv', index=False)

# pos_df.reset_index().to_csv('/dbfs/FileStore/tables/subhash_dixit/baseline_forecast_project/baseline_shape_analysis/us_all_category_pos_reckitt_data.csv', index=False)

pos_df = pd.read_csv('/dbfs/FileStore/tables/subhash_dixit/baseline_forecast_project/baseline_shape_analysis/us_all_category_pos_reckitt_data.csv')
print(f"pos_df shape: {pos_df.shape}")


# COMMAND ----------

# MAGIC %md
# MAGIC # EDA

# COMMAND ----------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------
# 1️⃣ Keep only required columns
# ---------------------------
cols_to_keep = [
    'PERIOD_ID',
    'RETAILER_ID',
    'PPG_NM',
    'NON_PROMO_SALES_VOLUME',
    'NET_SALES_VOLUME',
    'PROMO_SALES_VOLUME',
    'NET_SALES_WITH_TAXES'
]

df_filtered = pos_df[cols_to_keep].copy()
print(f"Filtered dataset shape: {df_filtered.shape}")
print(f"Missing values:\n{df_filtered.isnull().sum()}")

# ---------------------------
# 2️⃣ Descriptive Statistics
# ---------------------------
print("\nStatistical Summary:")
df_filtered.describe()

# COMMAND ----------

# ---------------------------
# 3️⃣ Univariate Analysis
# ---------------------------
# , 'NET_SALES_WITH_TAXES'
numeric_cols = ['NON_PROMO_SALES_VOLUME', 'NET_SALES_VOLUME', 'PROMO_SALES_VOLUME']

import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

plt.figure(figsize=(10,6))

colors = ['blue', 'green', 'red', 'orange', 'purple', 'brown']  # Extend if needed
for i, col in enumerate(numeric_cols):
    sns.kdeplot(df_filtered[col], label=col, color=colors[i % len(colors)], fill=True, alpha=0.3)

plt.title("Distribution of Numerical Columns")
plt.xlabel("Value")
plt.ylabel("Density")
plt.legend()
plt.show()

df_filtered['UID'] = df_filtered['RETAILER_ID'].astype(str) + " | " + df_filtered['PPG_NM'].astype(str)

# Plot counts per Retailer-PPG combination
plt.figure(figsize=(12,8))
sns.countplot(y='UID', data=df_filtered, 
              order=df_filtered['UID'].value_counts().index)
plt.title("Counts per Retailer-PPG Combination")
plt.xlabel("Count")
plt.ylabel("Retailer | PPG")
plt.tight_layout()
plt.show()

# COMMAND ----------

import matplotlib.pyplot as plt
import seaborn as sns

# Pivot: rows = UID, columns = PERIOD_ID
pivot_df = df_filtered.pivot_table(index='UID', columns='PERIOD_ID', values='NON_PROMO_SALES_VOLUME', fill_value=0)

plt.figure(figsize=(20,12))
sns.heatmap(pivot_df, cmap='viridis', cbar_kws={'label':'Non-Promo Sales'})
plt.title("Heatmap of Non-Promo Sales across UIDs over Periods")
plt.xlabel("Period")
plt.ylabel("UID Index")
plt.show()


# COMMAND ----------

# ---------------------------
# 4️⃣ Bivariate Analysis
# ---------------------------

import matplotlib.pyplot as plt
import seaborn as sns

# Sample 20 UIDs (adjust as needed)
sample_uids = df_filtered['UID'].unique()[:20]

plt.figure(figsize=(15,8))
for uid in sample_uids:
    temp = df_filtered[df_filtered['UID']==uid]
    sns.lineplot(x='PERIOD_ID', y='NON_PROMO_SALES_VOLUME', data=temp, label=uid)

plt.title("Non-Promo Sales Trend for Sample UIDs")
plt.xlabel("Period")
plt.ylabel("Non-Promo Sales Volume")
plt.legend(title="UID", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


# COMMAND ----------

# Boxplots of numeric columns per UID
# for col in numeric_cols:
#     plt.figure(figsize=(12,6))
#     sns.boxplot(x='UID', y=col, data=df_filtered)
#     plt.xticks(rotation=90)
#     plt.title(f"{col} distribution across UIDs")
#     plt.show()

# Sales trend per UID
# plt.figure(figsize=(12,6))
# for retailer in df_filtered['UID'].unique()[:5]:  # sample 5 retailers
#     temp = df_filtered[df_filtered['UID']==retailer]
#     sns.lineplot(x='PERIOD_ID', y='NON_PROMO_SALES_VOLUME', data=temp, label=retailer)
# plt.title("Non-Promo Sales Trend for Sample UIDs")
# plt.xlabel("Period")
# plt.ylabel("Non-Promo Sales Volume")
# plt.legend()
# plt.show()


# COMMAND ----------

# ---------------------------
# 5️⃣ Correlation Analysis
# ---------------------------
corr_matrix = df_filtered[numeric_cols].corr()
print("Correlation Matrix:\n")
display(corr_matrix)
plt.figure(figsize=(6,5))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap of Numeric Features")
plt.show()



# COMMAND ----------

# ---------------------------
# Aggregated Analysis at Retailer-PPG Level
# ---------------------------
agg_metrics = ['mean', 'sum', 'median', 'max', 'min', 'std']  # metrics to calculate

retailer_ppg_agg = df_filtered.groupby(['RETAILER_ID', 'PPG_NM'])[numeric_cols].agg(agg_metrics)
print("Aggregated metrics at Retailer-PPG level:\n")
display(retailer_ppg_agg)
# Optional: sort by NON_PROMO_SALES_VOLUME mean to see top performers
retailer_ppg_agg_sorted = retailer_ppg_agg.sort_values(('NON_PROMO_SALES_VOLUME', 'mean'), ascending=False)
print("\nTop Retailer-PPG combinations by avg NON_PROMO_SALES_VOLUME:\n")
display(retailer_ppg_agg_sorted.head(10))
# ---------------------------
# Aggregated Analysis per PPG
# ---------------------------
ppg_agg = df_filtered.groupby('PPG_NM')[numeric_cols].agg(agg_metrics)
ppg_agg_sorted = ppg_agg.sort_values(('NON_PROMO_SALES_VOLUME', 'mean'), ascending=False)
print("\nAggregated metrics per PPG:\n")
display(ppg_agg_sorted)
# ---------------------------
# Aggregated Analysis per Retailer
# ---------------------------
retailer_agg = df_filtered.groupby('RETAILER_ID')[numeric_cols].agg(agg_metrics)
retailer_agg_sorted = retailer_agg.sort_values(('NON_PROMO_SALES_VOLUME', 'mean'), ascending=False)
print("\nAggregated metrics per Retailer:\n")
display(retailer_agg_sorted.head(10))


# COMMAND ----------

import matplotlib.pyplot as plt
import seaborn as sns

# Reset index for plotting
retailer_ppg_agg_reset = retailer_ppg_agg.reset_index()

# Sort by NON_PROMO_SALES_VOLUME mean
retailer_ppg_agg_reset = retailer_ppg_agg_reset.sort_values(('NON_PROMO_SALES_VOLUME', 'mean'), ascending=False)

plt.figure(figsize=(12,6))
sns.lineplot(
    x=range(len(retailer_ppg_agg_reset)),  # Use index instead of names
    y=retailer_ppg_agg_reset[('NON_PROMO_SALES_VOLUME','mean')],
    marker='o'
)

plt.title("Average Non-Promo Sales Volume per Retailer-PPG")
plt.xlabel("Retailer-PPG Index")
plt.ylabel("Average Non-Promo Sales Volume")
plt.tight_layout()
plt.show()


# COMMAND ----------

import matplotlib.pyplot as plt

# Aggregate across all UIDs (sum over period)
ts_overall = df_filtered.groupby('PERIOD_ID')['NON_PROMO_SALES_VOLUME'].sum()

plt.figure(figsize=(12,6))
plt.plot(ts_overall.index, ts_overall.values, marker='o')
plt.title("Overall Non-Promo Sales Volume Trend")
plt.xlabel("Period")
plt.ylabel("Non-Promo Sales Volume")
plt.show()


# COMMAND ----------

from statsmodels.tsa.seasonal import seasonal_decompose

# Decompose overall series
decomposition = seasonal_decompose(ts_overall, model='additive', period=12)  # period can be 12 for monthly

decomposition.plot()
plt.suptitle("Time Series Decomposition of Non-Promo Sales Volume", fontsize=16)
plt.show()



# COMMAND ----------

import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# Get all unique retailers
retailers = df_filtered['RETAILER_ID'].unique()

# Loop through each retailer
for retailer in retailers:
    ts_retailer = df_filtered[df_filtered['RETAILER_ID'] == retailer].sort_values('PERIOD_ID')
    ts_series = ts_retailer.set_index('PERIOD_ID')['NON_PROMO_SALES_VOLUME']

    # Only decompose if enough data points
    if len(ts_series) >= 12:  # period=12 for monthly-like seasonality
        decomposition = seasonal_decompose(ts_series, model='additive', period=12)
        
        plt.figure(figsize=(10,6))
        decomposition.plot()
        plt.suptitle(f"Time Series Decomposition for Retailer: {retailer}", fontsize=14)
        plt.tight_layout()
        plt.show()


# COMMAND ----------

import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# Step 1: Aggregate total sales per UID
uid_total_sales = df_filtered.groupby('UID')['NON_PROMO_SALES_VOLUME'].sum().sort_values(ascending=False)

# Step 2: Select top 20 UIDs
top_20_uids = uid_total_sales.head(20).index

# Step 3: Loop through each UID and decompose
for uid in top_20_uids:
    ts_uid = df_filtered[df_filtered['UID'] == uid].sort_values('PERIOD_ID')
    ts_uid_series = ts_uid.set_index('PERIOD_ID')['NON_PROMO_SALES_VOLUME']

    # Only decompose if enough data points
    if len(ts_uid_series) >= 12:  # period=12 for monthly-like seasonality
        decomposition = seasonal_decompose(ts_uid_series, model='additive', period=12)
        
        plt.figure(figsize=(10,6))
        decomposition.plot()
        plt.suptitle(f"Time Series Decomposition for UID: {uid}", fontsize=14)
        plt.tight_layout()
        plt.show()


# COMMAND ----------

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

plt.figure(figsize=(12,4))
plot_acf(ts_overall, lags=20, ax=plt.gca())
plt.title("Autocorrelation of Non-Promo Sales Volume")
plt.show()

plt.figure(figsize=(12,4))
plot_pacf(ts_overall, lags=20, ax=plt.gca())
plt.title("Partial Autocorrelation of Non-Promo Sales Volume")
plt.show()


# COMMAND ----------

import seaborn as sns

plt.figure(figsize=(12,6))
sns.boxplot(x='PERIOD_ID', y='NON_PROMO_SALES_VOLUME', data=df_filtered)
plt.xticks(rotation=90)
plt.title("Non-Promo Sales Volume Distribution per Period")
plt.show()


# COMMAND ----------

# Example: mean sales per PPG
ppg_ts = df_filtered.groupby(['PERIOD_ID', 'PPG_NM'])['NON_PROMO_SALES_VOLUME'].mean().unstack()

ppg_ts.plot(figsize=(15,6), legend=False, alpha=0.5)
plt.title("Mean Non-Promo Sales Volume per PPG over Time")
plt.xlabel("Period")
plt.ylabel("Non-Promo Sales Volume")
plt.show()


# COMMAND ----------

# df_filtered['log_non_promo'] = np.log1p(df_filtered['NON_PROMO_SALES_VOLUME'])

# # Optional: scale to 0-1
# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# df_filtered['scaled_non_promo'] = scaler.fit_transform(df_filtered[['log_non_promo']])


# COMMAND ----------



# COMMAND ----------

# DBTITLE 1,Data Filtering

# Select only the relevant columns
cols_to_keep = [
    'PERIOD_ID',
    'RETAILER_ID',
    'PPG_NM',
    'NON_PROMO_SALES_VOLUME',
]

print(f"Original dataset shape: {pos_df.shape}")
print(f"Length of unique PPGs in Orginal Dataset: {len(pos_df['PPG_NM'].unique())}")
pos_df_filtered = pos_df[cols_to_keep].copy()
print(f"Filtered (Only Columns) dataset shape: {pos_df_filtered.shape}")
print(f"Length of unique PPGs after column filter: {len(pos_df_filtered['PPG_NM'].unique())}")

# # Step 1: Calculate total NON_PROMO_SALES_VOLUME for each retailer–PPG pair
# sales_summary = (
#     pos_df_filtered.groupby(['RETAILER_ID', 'PPG_NM'], as_index=False)['NON_PROMO_SALES_VOLUME']
#     .sum()
# )

# # Step 2: Rank products within each retailer based on NON_PROMO_SALES_VOLUME
# sales_summary['Rank'] = sales_summary.groupby('RETAILER_ID')['NON_PROMO_SALES_VOLUME'].rank(method='first', ascending=False)

# # Step 3: Select top 20 products per retailer
# top_products_per_retailer = sales_summary[sales_summary['Rank'] <= 20]

# # Step 4: Merge back with original dataset to keep only those top 20 products per retailer
# pos_df_filtered = pos_df_filtered.merge(
#     top_products_per_retailer[['RETAILER_ID', 'PPG_NM']],
#     on=['RETAILER_ID', 'PPG_NM'],
#     how='inner'
# )


# # retailer_ppg_pairs = pos_df_filtered[['RETAILER_ID', 'PPG_NM']].drop_duplicates()
# # selected_pairs = retailer_ppg_pairs.head(100)
# # pos_df_filtered = pos_df_filtered.merge(selected_pairs, on=['RETAILER_ID', 'PPG_NM'], how='inner')
# print(f"Filtered (Columns, Retailer and PPGs) dataset shape: {pos_df_filtered.shape}")
# print(f"Length of unique PPGs: {len(pos_df_filtered['PPG_NM'].unique())}")
# print(f"Length of unique Retailer: {len(pos_df_filtered['RETAILER_ID'].unique())}")
# print(f"Length of unique UID: {len(pos_df_filtered[['RETAILER_ID', 'PPG_NM']].drop_duplicates())}")


# COMMAND ----------



# %sql
# -- Ret-PPG-Week -- 1 Row
# select RETAILER_ID , PPG_NM, period_id, count(*), collect_list(PRODUCT_ID) from dev_rgmx_etl.us_all_dishcare_econ_pos
# group by RETAILER_ID , PPG_NM, PERIOD_ID
# having count(*) > 1


# COMMAND ----------

# DBTITLE 1,Stacking the Years
import pandas as pd
import numpy as np

# 1️⃣ Convert PERIOD_ID to datetime
pos_df_filtered['PERIOD_ID'] = pd.to_datetime(pos_df_filtered['PERIOD_ID'])

# 2️⃣ Extract Year and Week
pos_df_filtered['Year'] = pos_df_filtered['PERIOD_ID'].dt.year
pos_df_filtered['Week'] = pos_df_filtered['PERIOD_ID'].dt.isocalendar().week

# 3️⃣ Keep only complete years (e.g., 2023 & 2024)
valid_years = [2023, 2024]
pos_df_filtered = pos_df_filtered[pos_df_filtered['Year'].isin(valid_years)].copy()

print(f"✅ Filtered dataset shape after keeping only valid years ({valid_years}): {pos_df_filtered.shape}")
print(f"Filtered (Columns, Retailer and PPGs) dataset shape: {pos_df_filtered.shape}")
print(f"Length of unique PPGs: {len(pos_df_filtered['PPG_NM'].unique())}")
print(f"Length of unique Retailer: {len(pos_df_filtered['RETAILER_ID'].unique())}")
print(f"Length of unique UID: {len(pos_df_filtered[['RETAILER_ID', 'PPG_NM']].drop_duplicates())}")

# 4️⃣ Pivot to get weekly sales for each (Retailer, Product, Year)
# pivot_df = (
#     pos_df_filtered.pivot_table(
#         index=['RETAILER_ID', 'PPG_NM', 'Year'],
#         columns='Week',
#         values='NON_PROMO_SALES_VOLUME',
#         fill_value=0
#     )
#     .reset_index()
# )
pivot_df = pos_df_filtered.pivot_table(
    index=['RETAILER_ID', 'PPG_NM', 'Year'],
    columns='Week',
    values='NON_PROMO_SALES_VOLUME',
    aggfunc='sum'
).fillna(0)

# 5️⃣ Rename week columns
pivot_df.columns.name = None
pivot_df = pivot_df.rename(columns=lambda x: f"W_{x}" if isinstance(x, (int, np.integer)) else x)

# 7️⃣ Remove rows with 0 sales across all 52 weeks
week_cols = [col for col in pivot_df.columns if col.startswith('W_')]
pivot_df = pivot_df.loc[pivot_df[week_cols].sum(axis=1) != 0]

print(f"📊 Weekly pivot dataset shape: {pivot_df.shape}")
print(f"🗓️ Years included: {pivot_df.index.unique().tolist()}")

print(f"✅ Filtered out rows with zero sales across all weeks. Final shape: {pivot_df.shape}")

# 6️⃣ Display sample
display(pivot_df)


# COMMAND ----------

# DBTITLE 1,Filter Seasonal, Perfect, NII and Delisted Items
import numpy as np
import pandas as pd

def classify_sales_sequence(sales):
    """Classify weekly sales pattern based on contiguous zero sequences."""
    sales = np.array(sales)
    n = len(sales)

    # If all zeros → no sales
    if np.all(sales == 0):
        return 'No Sales'

    # Detect where non-zero values occur
    nonzero = np.nonzero(sales)[0]
    first, last = nonzero[0], nonzero[-1]

    # Count continuous zeros from start and end
    start_zeros = 0
    for val in sales:
        if val == 0:
            start_zeros += 1
        else:
            break

    end_zeros = 0
    for val in sales[::-1]:
        if val == 0:
            end_zeros += 1
        else:
            break

    # Pattern detection based on sequences
    if np.all(sales > 0):
        return 'Perfect Product'
    elif start_zeros >= 4 and end_zeros == 0:
        return 'New Product'
    elif end_zeros >= 4 and start_zeros == 0:
        return 'Delisted Product'
    elif start_zeros >= 1 and end_zeros >= 1:
        return 'Seasonal Product'
    else:
        return 'Irregular/Mixed'

# --------------------------------------------------------
# Apply on your pivot_df
# --------------------------------------------------------
week_cols = [col for col in pivot_df.columns if col.startswith('W_')]

pivot_df['Pattern'] = pivot_df[week_cols].apply(classify_sales_sequence, axis=1)

# Optional: Summary count per category
pattern_summary = (
    pivot_df['Pattern']
    .value_counts()
    .reset_index()
    .rename(columns={'index': 'Pattern', 'Pattern': 'Count'})
)


# Filter only selectd pattern rows
# pivot_df = pivot_df[pivot_df['Pattern'] == 'Perfect Product'].copy()
# pivot_df = pivot_df[pivot_df['Pattern'] == 'Seasonal Product'].copy()
pivot_df = pivot_df[pivot_df['Pattern'] == 'Irregular/Mixed'].copy()

# Drop the pattern column (not needed for model training)
pivot_df = pivot_df.drop(columns=['Pattern'])

# 4Optional: reset index for clean processing
# perfect_df = perfect_df.reset_index(drop=False)


display(pattern_summary)
display(pivot_df)
print(f"✅ Filtered only Perfect Products: {pivot_df.shape[0]} rows retained.")
display(pivot_df)


# COMMAND ----------

pivot_df.shape

# COMMAND ----------

# Total UID Count
len(pos_df[['RETAILER_ID', 'PPG_NM']].drop_duplicates())


# COMMAND ----------



# For 3 Years Dataset
# import pandas as pd
# import numpy as np

# # Ensure correct order
# pos_df_filtered = pos_df_filtered.sort_values(['RETAILER_ID', 'PPG_NM', 'PERIOD_ID'])

# # Pivot to create matrix: rows = retailer-PPG, columns = time (PERIOD_ID)
# pivot_df = pos_df_filtered.pivot_table(
#     index=['RETAILER_ID', 'PPG_NM'],
#     columns='PERIOD_ID',
#     values='NON_PROMO_SALES_VOLUME',
#     aggfunc='sum'
# ).fillna(0)

# print(f"Original Pivot Data Shape: {pivot_df.shape}")
# display(pivot_df.reset_index())

# COMMAND ----------

# MAGIC %md
# MAGIC # Data Preprocessing
# MAGIC **In this example, time series are preprocessed using TimeSeriesScalerMeanVariance.**
# MAGIC This scaler is such that each output time series has zero mean and unit variance.
# MAGIC The assumption here is that the range of a given time series is uninformative and one only wants to compare shapes.

# COMMAND ----------

# DBTITLE 1,Smoothening and Scaling
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from tslearn.preprocessing import TimeSeriesScalerMeanVariance
# from scipy.signal import savgol_filter
# import random

# # Step 1️⃣ — Convert pivot to numpy 3D array
# print("Original shape:", pivot_df.shape)
# X = pivot_df.values[:, :, np.newaxis]  # shape: (n_series, n_weeks, 1)
# n_series, n_weeks, _ = X.shape
# print(f"Converted to 3D array: {X.shape}")

# # Step 2️⃣ — Apply smoothing for each series
# X_smooth = np.zeros_like(X)
# for i in range(n_series):
#     # Apply Savitzky-Golay smoothing (window=5, polynomial=2)
#     # X_smooth[i, :, 0] = savgol_filter(X[i, :, 0], window_length=5, polyorder=2, mode='nearest')
#     X_smooth[i, :, 0] = pd.Series(X[i, :, 0]).rolling(window=7, center=True, min_periods=1).mean()

# print("✅ Smoothing applied using Savitzky-Golay filter")

# # Step 3️⃣ — Compare before and after smoothing for 10 random series
# sample_indices = random.sample(range(n_series), min(10, n_series))

# plt.figure(figsize=(18, 12))
# for idx, i in enumerate(sample_indices):
#     plt.subplot(5, 2, idx + 1)
#     plt.plot(X[i, :, 0], color='gray', alpha=0.6, label='Before Smoothing')
#     plt.plot(X_smooth[i, :, 0], color='green', linewidth=2, label='After Smoothing')
#     plt.title(f"Series {i} (Retailer-PPG-Year Combo)")
#     plt.xlabel("Week")
#     plt.ylabel("Sales Volume")
#     plt.grid(alpha=0.3)
#     if idx == 0:
#         plt.legend()

# plt.suptitle("Before vs After Smoothing (10 Random Time Series)", fontsize=16, y=0.92)
# plt.tight_layout(rect=[0, 0, 1, 0.94])
# plt.show()

# # Step 4️⃣ — Standardize (mean=0, std=1)
# scaler = TimeSeriesScalerMeanVariance()
# X_scaled = scaler.fit_transform(X_smooth)

# print(f"✅ Final scaled shape: {X_scaled.shape}")


# COMMAND ----------

# DBTITLE 1,Scaling
# X = pivot_df.values  # shape (n_series, n_time_steps)
# X = X[:, :, np.newaxis]  # Add feature dimension → shape (n_series, n_time_steps, 1)

# print(X.shape)  # e.g. (500, 40, 1)

import numpy as np
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

# Each row = Retailer-PPG, each column = week (PERIOD_ID)
print("Original shape:", pivot_df.shape)

# Step 1️⃣ — Convert to numpy 3D array
X = pivot_df.values[:, :, np.newaxis]  # shape: (n_series, n_weeks, 1)
n_series, n_weeks, _ = X.shape

# # Step 2️⃣ — Define split ratio (e.g., 80% train, 20% test)
# train_ratio = 0.8
# split_point = int(n_weeks * train_ratio)

# # Step 3️⃣ — Split along the time axis (not randomly)
# X_train = X[:, :split_point, :]
# X_test = X[:, split_point:, :]

# print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# Step 4️⃣ — Scale each time series (important for shape clustering)
scaler = TimeSeriesScalerMeanVariance()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)
X_scaled = scaler.fit_transform(X)
print(f"X_scaled shape: {X_scaled.shape}")

# COMMAND ----------

X_scaled

# COMMAND ----------

# import pandas as pd

# # Ensure chronological order
# pos_df_filtered = pos_df_filtered.sort_values(['RETAILER_ID', 'PPG_NM', 'PERIOD_ID'])

# train_list, test_list = [], []

# # Loop through each Retailer–PPG combination
# for (retailer, ppg), group in pos_df_filtered.groupby(['RETAILER_ID', 'PPG_NM']):
#     # Sort by time to avoid leakage
#     group = group.sort_values('PERIOD_ID')
    
#     # Time-based split (e.g., 80% train, 20% test)
#     split_point = int(len(group) * 0.8)
    
#     train_part = group.iloc[:split_point]
#     test_part = group.iloc[split_point:]
    
#     # Append to lists
#     train_list.append(train_part)
#     test_list.append(test_part)

# # Combine all groups back into single DataFrames
# train_df = pd.concat(train_list).reset_index(drop=True)
# test_df = pd.concat(test_list).reset_index(drop=True)

# print(f"Train shape: {train_df.shape}")
# print(f"Test shape: {test_df.shape}")
# X_train_scaled

# COMMAND ----------

# pos_df_filtered[pos_df_filtered['PPG_NM'] == 'PRIVATE LABEL | DISHWASHER PRODUCTS | PRIVATE LABEL | RINSE AID | PRIVATE LABEL | LIQUID | CITRUS | 480 ML | 1'].display(    )

# COMMAND ----------

# X_train, y_train, X_test, y_test = CachedDatasets().load_dataset("Trace")

# X_train = X_train[y_train<4]
# y_train = y_train[y_train<4]

# X_test = X_test[y_test<4]
# y_test = y_test[y_test<4]

# COMMAND ----------

# y_train

# COMMAND ----------

# X_train.shape

# COMMAND ----------

# seed = 0
# np.random.seed(seed)
# X_train = TimeSeriesScalerMeanVariance().fit_transform(X_train)
# sz = X_train.shape[1]

# COMMAND ----------

X_scaled.shape

# COMMAND ----------

# DBTITLE 1,Actual Scaled Data Plot
plt.figure(figsize=(15, 20))  # make overall plot larger

for yi in range(50):
    plt.subplot(10, 5, yi + 1)
    plt.plot(X_scaled[yi].ravel(), "k-", alpha=.6)
#     plt.text(0.55, 0.85, f'Class Label: {y_train[yi]}', transform=plt.gca().transAxes, fontsize=8)
    plt.xticks(fontsize=6)   # small x-axis ticks
    plt.yticks(fontsize=6)   # small y-axis ticks

plt.tight_layout(pad=2.0)
plt.show()


# COMMAND ----------

# for yi in range(50):
#     plt.subplot(10, 5, yi + 1)
#     plt.plot(X_scaled[yi].ravel(), "k-", alpha=.2)
# #     plt.text(0.55, 0.85,'Class Label: %d' % (y_train[yi]))

# COMMAND ----------

# MAGIC %md
# MAGIC # Clusters Numbers Decision

# COMMAND ----------

# DBTITLE 1,Elbow Method to Choose Cluster (Using inertia (sum of squared distances)
# Sum_of_squared_distances = []
# K = range(2, 7)

# print("Running Elbow Method using DTW metric...\n")

# for i, k in enumerate(K, start=1):
#     print(f"→ Processing k = {k} ({i}/{len(K)})...", end=" ")
#     km = TimeSeriesKMeans(
#         n_clusters=k,
#         n_init=2,
#         metric="dtw",
#         verbose=False,
#         max_iter_barycenter=10,
#         random_state=0
#     )
#     km.fit(X_scaled)
#     Sum_of_squared_distances.append(km.inertia_)
#     print("✅ Done")

# print("\nAll cluster sizes processed successfully!")



# COMMAND ----------

# DBTITLE 1,Elbow Method Plot
# # --- Plot Elbow Curve ---
# plt.figure(figsize=(20, 10))
# plt.plot(K, Sum_of_squared_distances, 'bo-', markersize=8)
# plt.xlabel('Number of Clusters (k)', fontsize=12)
# plt.ylabel('Sum of Squared Distances (Inertia)', fontsize=12)
# plt.title('Elbow Method for Optimal k (DTW Metric)', fontsize=14)
# plt.grid(True)
# plt.show()


# COMMAND ----------

# DBTITLE 1,Prediction on Actual Data
# # Actual Prediction to get Cluster Labels
# n_clusters = 5
# model = TimeSeriesKMeans(n_clusters=n_clusters,
#                           n_init=2,
#                           metric="dtw",
#                           verbose=False,
#                           max_iter_barycenter=10,
#                           random_state=0)

# cluster_labels = model.fit_predict(X_scaled)  # X_scaled shape: (n_samples, n_timestamps, 1)
# print(cluster_labels)

# COMMAND ----------

# DBTITLE 1,Elbow Method to Choose Cluster (Using Silhouette Score)
# from sklearn.cluster import KMeans
# from sklearn.metrics import silhouette_samples

# import matplotlib.pyplot as plt
# import matplotlib.cm as cm
# import numpy as np


# train_pca = list(X_train.reshape(X_train.shape[0], X_train.shape[1]))
# pca = PCA(n_components=0.95)
# train_pca = pca.fit_transform(train_pca)

# X = train_pca

# range_n_clusters = [2, 3, 4, 5]

# for n_clusters in range_n_clusters:
#     # Create a subplot with 1 row and 2 columns
#     fig, (ax1, ax2) = plt.subplots(1, 2)
#     fig.set_size_inches(18, 7)

#     # The 1st subplot is the silhouette plot
#     # The silhouette coefficient can range from -1, 1 but in this example all
#     # lie within [-0.1, 1]
#     ax1.set_xlim([-0.1, 1])
#     # The (n_clusters+1)*10 is for inserting blank space between silhouette
#     # plots of individual clusters, to demarcate them clearly.
#     ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

#     # Initialize the clusterer with n_clusters value and a random generator
#     # seed of 10 for reproducibility.
#     clusterer = KMeans(n_clusters=n_clusters, random_state=0)
#     cluster_labels = clusterer.fit_predict(X)

#     # The silhouette_score gives the average value for all the samples.
#     # This gives a perspective into the density and separation of the formed
#     # clusters
#     silhouette_avg = silhouette_score(X, cluster_labels)
#     print(
#         "For n_clusters =",
#         n_clusters,
#         "The average silhouette_score is : {:.2f}"
#         .format(silhouette_avg),
#     )

#     # Compute the silhouette scores for each sample
#     sample_silhouette_values = silhouette_samples(X, cluster_labels)

#     y_lower = 10
#     for i in range(n_clusters):
#         # Aggregate the silhouette scores for samples belonging to
#         # cluster i, and sort them
#         ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
#         ith_cluster_silhouette_values.sort()


#         size_cluster_i = ith_cluster_silhouette_values.shape[0]
#         y_upper = y_lower + size_cluster_i

#         color = cm.nipy_spectral(float(i) / n_clusters)
#         ax1.fill_betweenx(
#             np.arange(y_lower, y_upper),
#             0,
#             ith_cluster_silhouette_values,
#             facecolor=color,
#             edgecolor=color,
#             alpha=0.7,
#         )

#         # Label the silhouette plots with their cluster numbers at the middle
#         ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

#         # Compute the new y_lower for next plot
#         y_lower = y_upper + 10  # 10 for the 0 samples

#     ax1.set_title("The silhouette plot for the various clusters.")
#     ax1.set_xlabel("The silhouette coefficient values")
#     ax1.set_ylabel("Cluster label")

#     # The vertical line for average silhouette score of all the values
#     ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

#     ax1.set_yticks([])  # Clear the yaxis labels / ticks
#     ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

#     # 2nd Plot showing the actual clusters formed
#     colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
#     ax2.scatter(
#         X[:, 0], X[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k"
#     )

#     # Labeling the clusters
#     centers = clusterer.cluster_centers_
#     # Draw white circles at cluster centers
#     ax2.scatter(
#         centers[:, 0],
#         centers[:, 1],
#         marker="o",
#         c="white",
#         alpha=1,
#         s=200,
#         edgecolor="k",
#     )

#     for i, c in enumerate(centers):
#         ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")

#     ax2.set_title("The visualization of the clustered data.")
#     ax2.set_xlabel("Feature space for the 1st feature")
#     ax2.set_ylabel("Feature space for the 2nd feature")

#     plt.suptitle(
#         "Silhouette analysis for KMeans clustering on sample data with n_clusters = %d"
#         % n_clusters,
#         fontsize=14,
#         fontweight="bold",
#     )

# plt.show()


# COMMAND ----------

# DBTITLE 1,Input clustering model params
n_clusters = 10

# COMMAND ----------

# DBTITLE 1,Prediction on Actual Data
# import matplotlib.pyplot as plt
# n_clusters = 4
# n_timestamps = X_scaled.shape[1]

# # Suppose you already fit TimeSeriesKMeans
# from tslearn.clustering import TimeSeriesKMeans
# model = TimeSeriesKMeans(n_clusters=n_clusters, metric="dtw", random_state=42)
# cluster_labels = model.fit_predict(X_scaled)  # X_scaled shape: (n_samples, n_timestamps, 1)


# plt.figure(figsize=(20, 6))
# plt.suptitle("Actual Data", fontsize=16, y=0.95)

# for yi in range(n_clusters):
#     plt.subplot(1, n_clusters, yi + 1)
#     for xx in X_scaled[cluster_labels == yi]:
#         plt.plot(xx.ravel(), "k-", alpha=0.2)  # flatten 3D series to 1D for plotting
#     plt.xlim(0, n_timestamps)
#     plt.ylim(np.min(X_scaled), np.max(X_scaled))  # dynamic y-limits
#     plt.text(0.5, 0.85, f'Cluster {yi + 1}', transform=plt.gca().transAxes)
#     if yi == 1:
#         plt.title("TimeSeriesKMeans Predicted Clusters")

# plt.tight_layout()
# plt.show()


# COMMAND ----------

# n_clusters = 3
# sz = X_scaled.shape[1]
# seed = 0

# ## Actual clusters(using labels) plot
# plt.figure()
# for yi in range(n_clusters):
#     plt.subplot(3, n_clusters, yi + 1)
#     for xx in X_scaled[y_train == yi+1]:
#         plt.plot(xx.ravel(), "k-", alpha=.2)
#     plt.xlim(0, sz)
#     plt.ylim(-4, 4)
#     plt.text(0.55, 0.85,'Cluster %d' % (yi + 1),
#              transform=plt.gca().transAxes)
#     if yi == 1:
#         plt.title("Acutal")

# COMMAND ----------

# MAGIC %md
# MAGIC # Model Training
# MAGIC

# COMMAND ----------

# DBTITLE 1,Model Training using Euclidean Distance
# Euclidean k-means
# n_clusters = 4
sz = X_scaled.shape[1]
seed = 0
print("Euclidean k-means")
km = TimeSeriesKMeans(n_clusters=n_clusters, verbose=True, random_state=seed)
y_pred_km = km.fit_predict(X_scaled)


print("Euclidean silhoutte: {:.2f}".format(silhouette_score(X_scaled, y_pred_km, metric="euclidean")))

plt.figure(figsize=(20, 10))  # Bigger figure for readability
plt.suptitle("Euclidean k-means", fontsize=16, y=0.95)

for yi in range(n_clusters):
    plt.subplot(3, n_clusters, yi + 1)
    for xx in X_scaled[y_pred_km == yi]:
        plt.plot(xx.ravel(), "k-", alpha=.2)
    plt.plot(km.cluster_centers_[yi].ravel(), "r-")
    plt.xlim(0, sz)
    plt.ylim(-4, 4)
    plt.text(0.55, 0.85,'Cluster %d' % (yi + 1),
             transform=plt.gca().transAxes)
    # if yi == 1:
    #     plt.title("Euclidean $k$-means")

# COMMAND ----------

import matplotlib.pyplot as plt

# Set up rows & columns dynamically n_cols per row
n_cols = 5
n_rows = int(np.ceil(n_clusters / n_cols))

plt.figure(figsize=(18, n_rows * 4))  # Increase height for readability
plt.suptitle("Euclidean K-Means Clusters (Scaled Time Series)", fontsize=18, y=1)

for yi in range(n_clusters):
    plt.subplot(n_rows, n_cols, yi + 1)
    
    # Plot all individual time series in light gray
    for xx in X_scaled[y_pred_km == yi]:
        plt.plot(xx.ravel(), color="green", alpha=0.3)
        
    # Plot cluster centroid in red
    plt.plot(km.cluster_centers_[yi].ravel(), color="red", linewidth=2)
    
    plt.title(f"Cluster {yi + 1}", fontsize=14)
    plt.xlabel("Week")
    plt.ylabel("Scaled Sales")
    plt.xlim(0, sz)
    plt.ylim(-4, 4)
    plt.grid(alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.97])  # Leave space for title
plt.show()


# COMMAND ----------

# DBTITLE 1,Model Training using DTW (Dynamic Time Wrapping) Distance
# # DTW-k-means
# # n_clusters = 4
# sz = X_scaled.shape[1]
# seed = 0


# print("DTW k-means")
# dba_km = TimeSeriesKMeans(n_clusters=n_clusters,
#                           n_init=2,
#                           metric="dtw",
#                           verbose=False,
#                           max_iter_barycenter=10,
#                           random_state=seed)
# y_pred_dba_km = dba_km.fit_predict(X_scaled)
# print("DTW silhoutte: {:.2f}".format(silhouette_score(X_scaled, y_pred_dba_km, metric="dtw")))

# plt.figure(figsize=(20, 10))  # Bigger figure for readability
# plt.suptitle("DTW k-means", fontsize=16, y=1)

# for yi in range(n_clusters):
#     plt.subplot(3, n_clusters, yi+1)
#     for xx in X_scaled[y_pred_dba_km == yi]:
#         plt.plot(xx.ravel(), "k-", alpha=.2)
#     plt.plot(dba_km.cluster_centers_[yi].ravel(), "r-")
#     plt.xlim(0, sz)
#     plt.ylim(-4, 4)
#     plt.text(0.55, 0.85,'Cluster %d' % (yi + 1),
#              transform=plt.gca().transAxes)
#     # if yi == 1:
#     #     plt.title("DBA $k$-means")

# plt.tight_layout()
# plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC # Model Validation (If we have actual label - not useful in real world)

# COMMAND ----------

# DBTITLE 1,Verifying performance using Rand Index (calculated using labels y_train, so can't be calculated in real world use cases
# from sklearn import metrics

# print("Adjusted Rand Index for Euclidean: {:.2f}".format(metrics.adjusted_rand_score(y_train, y_pred_km)))
# print("Adjusted Rand Index for DBA : {:.2f}".format(metrics.adjusted_rand_score(y_train, y_pred_dba_km)))

# COMMAND ----------

# MAGIC %md
# MAGIC # All plots together

# COMMAND ----------

# DBTITLE 1,All Plots Together
import matplotlib.pyplot as plt

plt.figure(figsize=(20, 8))  # Bigger figure for readability
plt.suptitle("Comparison of Time Series Clustering Techniques", fontsize=16, y=0.95)

# --- Euclidean K-Means ---
for yi in range(n_clusters):
    plt.subplot(2, n_clusters, yi + 1)
    # All individual time series in green
    for xx in X_scaled[y_pred_km == yi]:
        plt.plot(xx.ravel(), color="green", alpha=0.2)
    # Cluster centroid in red
    plt.plot(km.cluster_centers_[yi].ravel(), color="red", linewidth=2)
    plt.xlim(0, sz)
    plt.ylim(-4, 4)
    plt.title(f"Cluster {yi+1}", fontsize=10)
    if yi == 0:
        plt.ylabel("Euclidean K-Means", fontsize=12)
    plt.grid(alpha=0.3)

# # --- DTW K-Means ---
# for yi in range(n_clusters):
#     plt.subplot(2, n_clusters, n_clusters + yi + 1)
#     # All individual time series in green
#     for xx in X_scaled[y_pred_dba_km == yi]:
#         plt.plot(xx.ravel(), color="green", alpha=0.2)
#     # Cluster centroid in red
#     plt.plot(dba_km.cluster_centers_[yi].ravel(), color="red", linewidth=2)
#     plt.xlim(0, sz)
#     plt.ylim(-4, 4)
#     plt.title(f"Cluster {yi+1}", fontsize=10)
#     if yi == 0:
#         plt.ylabel("DTW K-Means", fontsize=12)
#     plt.grid(alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.93])  # Adjust spacing for suptitle
plt.show()


# COMMAND ----------

# # display(pos_df_filtered)
# display(pivot_df.columns)
# pivot_df = pivot_df.copy()

# # Add cluster labels
# pivot_df["Cluster_Euclidean"] = y_pred_km
# pivot_df["Cluster_DTW"] = y_pred_dba_km

# print("✅ Cluster labels added to pivot_df")
# display(pivot_df.head())

# pivot_long = (
#     pivot_df
#     .reset_index()
#     .melt(
#         id_vars=["RETAILER_ID", "PPG_NM", "Year", "Cluster_Euclidean", "Cluster_DTW"],
#         value_vars=[col for col in pivot_df.columns if col.startswith("W_")],
#         var_name="Week",
#         value_name="Sales"
#     )
# )

# # Extract week number (as integer)
# pivot_long["Week"] = pivot_long["Week"].str.replace("W_", "").astype(int)

# import matplotlib.pyplot as plt

# n_clusters = pivot_df["Cluster_Euclidean"].nunique()
# years = sorted(pivot_long["Year"].unique())

# for cluster in sorted(pivot_df["Cluster_Euclidean"].unique()):
#     print(cluster)
#     plt.figure(figsize=(20, 8))
#     plt.suptitle(f"Cluster {cluster} - Euclidean KMeans (Year-wise Patterns)", fontsize=16)
    
#     cluster_data = pivot_long[pivot_long["Cluster_Euclidean"] == cluster]
    
#     for i, year in enumerate(years):
#         plt.subplot(1, len(years), i + 1)
#         yearly_data = cluster_data[cluster_data["Year"] == year]
        
#         for key, grp in yearly_data.groupby(["RETAILER_ID", "PPG_NM"]):
#             plt.plot(grp["Week"], grp["Sales"], color="green", alpha=0.2)
        
#         plt.title(f"{year}")
#         plt.xlabel("Week Number")
#         plt.ylabel("Sales Volume")
#         plt.grid(alpha=0.3)
#         plt.xlim(1, 52)
    
#     plt.tight_layout(rect=[0, 0, 1, 0.93])
#     plt.show()


# COMMAND ----------

# for cluster in sorted(pivot_df["Cluster_Euclidean"].unique()):
#     plt.figure(figsize=(20, 8))
#     plt.suptitle(f"Cluster {cluster} - Euclidean KMeans (Year-wise Patterns)", fontsize=16)
    
#     cluster_data = pivot_long[pivot_long["Cluster_Euclidean"] == cluster]
    
#     for i, year in enumerate(years):
#         plt.subplot(1, len(years), i + 1)
#         yearly_data = cluster_data[cluster_data["Year"] == year]
        
#         for key, grp in yearly_data.groupby(["RETAILER_ID", "PPG_NM"]):
#             plt.plot(grp["Week"], grp["Sales"], color="green", alpha=0.2)
        
#         plt.title(f"{year}")
#         plt.xlabel("Week Number")
#         plt.ylabel("Sales Volume")
#         plt.grid(alpha=0.3)
#         plt.xlim(1, 52)
    
#     plt.tight_layout(rect=[0, 0, 1, 0.93])
#     plt.show()

# COMMAND ----------

# for cluster in sorted(pivot_df["Cluster_DTW"].unique()):
#     plt.figure(figsize=(20, 8))
#     plt.suptitle(f"Cluster {cluster} - DTW KMeans (Year-wise Patterns)", fontsize=16)
    
#     cluster_data = pivot_long[pivot_long["Cluster_DTW"] == cluster]
    
#     for i, year in enumerate(years):
#         plt.subplot(1, len(years), i + 1)
#         yearly_data = cluster_data[cluster_data["Year"] == year]
        
#         for key, grp in yearly_data.groupby(["RETAILER_ID", "PPG_NM"]):
#             plt.plot(grp["Week"], grp["Sales"], color="green", alpha=0.2)
        
#         plt.title(f"{year}")
#         plt.xlabel("Week Number")
#         plt.ylabel("Sales Volume")
#         plt.grid(alpha=0.3)
#         plt.xlim(1, 52)
    
#     plt.tight_layout(rect=[0, 0, 1, 0.93])
#     plt.show()


# COMMAND ----------

# import matplotlib.pyplot as plt

# plt.figure(figsize=(20, 8))  # Bigger figure for readability
# plt.suptitle("Comparison of Time Series Clustering Techniques", fontsize=16, y=0.95)

# # --- Euclidean K-Means ---
# for yi in range(n_clusters):
#     plt.subplot(2, n_clusters, yi + 1)
#     for xx in X_scaled[y_pred_km == yi]:
#         plt.plot(xx.ravel(), "k-", alpha=0.2)
#     plt.plot(km.cluster_centers_[yi].ravel(), "r-", linewidth=2)
#     plt.xlim(0, sz)
#     plt.ylim(-4, 4)
#     plt.title(f"Cluster {yi+1}", fontsize=10)
#     if yi == 0:
#         plt.ylabel("Euclidean K-Means", fontsize=12)
#     plt.grid(alpha=0.3)

# # --- DBA K-Means ---
# for yi in range(n_clusters):
#     plt.subplot(2, n_clusters, n_clusters + yi + 1)
#     for xx in X_scaled[y_pred_dba_km == yi]:
#         plt.plot(xx.ravel(), "k-", alpha=0.2)
#     plt.plot(dba_km.cluster_centers_[yi].ravel(), "r-", linewidth=2)
#     plt.xlim(0, sz)
#     plt.ylim(-4, 4)
#     plt.title(f"Cluster {yi+1}", fontsize=10)
#     if yi == 0:
#         plt.ylabel("DTW K-Means", fontsize=12)
#     plt.grid(alpha=0.3)

# plt.tight_layout(rect=[0, 0, 1, 0.93])  # Adjust spacing for suptitle
# plt.show()


# COMMAND ----------

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Create subplot grid: 2 rows (Euclidean, DBA) x n_clusters columns
fig = make_subplots(
    rows=2,
    cols=n_clusters,
    subplot_titles=[f"Cluster {i+1}" for i in range(n_clusters)] * 2,
    vertical_spacing=0.15,
    horizontal_spacing=0.05,
)

# --- Euclidean K-Means (Row 1) ---
for yi in range(n_clusters):
    # Add all series for this cluster
    cluster_indices = (y_pred_km == yi)
    for series in X_scaled[cluster_indices]:
        fig.add_trace(
            go.Scatter(
                y=series.ravel(),
                mode='lines',
                line=dict(color='green', width=1),
                opacity=0.2,
                showlegend=False
            ),
            row=1, col=yi + 1
        )

    # Add cluster center
    fig.add_trace(
        go.Scatter(
            y=km.cluster_centers_[yi].ravel(),
            mode='lines',
            line=dict(color='red', width=3),
            name=f"Cluster {yi+1} Center (Euclidean)" if yi == 0 else None,
            showlegend=(yi == 0)
        ),
        row=1, col=yi + 1
    )

# --- DTW K-Means (Row 2) ---
for yi in range(n_clusters):
    cluster_indices = (y_pred_dba_km == yi)
    for series in X_scaled[cluster_indices]:
        fig.add_trace(
            go.Scatter(
                y=series.ravel(),
                mode='lines',
                line=dict(color='green', width=1),
                opacity=0.2,
                showlegend=False
            ),
            row=2, col=yi + 1
        )

    fig.add_trace(
        go.Scatter(
            y=dba_km.cluster_centers_[yi].ravel(),
            mode='lines',
            line=dict(color='red', width=3),
            name=f"Cluster {yi+1} Center (DTW)" if yi == 0 else None,
            showlegend=(yi == 0)
        ),
        row=2, col=yi + 1
    )

# Layout styling
fig.update_layout(
    height=600,
    width=1500,
    title_text="Interactive Comparison of Time Series Clustering (Euclidean vs DBA K-Means)",
    title_x=0.5,
    plot_bgcolor='white',
    hovermode='x unified',
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.05,
        xanchor="center",
        x=0.5
    ),
)

# Add row labels
fig.add_annotation(
    text="Euclidean K-Means",
    xref="paper", yref="paper",
    x=-0.04, y=0.78,
    textangle=-90,
    font=dict(size=14),
    showarrow=False
)
fig.add_annotation(
    text="DBA K-Means",
    xref="paper", yref="paper",
    x=-0.04, y=0.1,
    textangle=-90,
    font=dict(size=14),
    showarrow=False
)

fig.show()


# COMMAND ----------

## Actual clusters(using labels) plot
# plt.figure()
# for yi in range(n_clusters):
#     plt.subplot(3, n_clusters, yi + 1)
#     for xx in X_train[y_train == yi+1]:
#         plt.plot(xx.ravel(), "k-", alpha=.2)
#     plt.xlim(0, sz)
#     plt.ylim(-4, 4)
#     plt.text(0.55, 0.85,'Cluster %d' % (yi + 1),
#              transform=plt.gca().transAxes)
#     if yi == 1:
#         plt.title("Acutal")

# plt.figure()
# for yi in range(n_clusters):
#     plt.subplot(3, n_clusters, yi + 4)
#     for xx in X_scaled[y_pred_km == yi]:
#         plt.plot(xx.ravel(), "k-", alpha=.2)
#     plt.plot(km.cluster_centers_[yi].ravel(), "r-")
#     plt.xlim(0, sz)
#     plt.ylim(-4, 4)
#     plt.text(0.55, 0.85,'Cluster %d' % (yi + 1),
#              transform=plt.gca().transAxes)
#     if yi == 1:
#         plt.title("Euclidean $k$-means")

# for yi in range(n_clusters):
#     plt.subplot(3, n_clusters, yi + 7)
#     for xx in X_scaled[y_pred_dba_km == yi]:
#         plt.plot(xx.ravel(), "k-", alpha=.2)
#     plt.plot(dba_km.cluster_centers_[yi].ravel(), "r-")
#     plt.xlim(0, sz)
#     plt.ylim(-4, 4)
#     plt.text(0.55, 0.85,'Cluster %d' % (yi + 1),
#              transform=plt.gca().transAxes)
#     if yi == 1:
#         plt.title("DBA $k$-means")



# COMMAND ----------

# MAGIC %md
# MAGIC # Profiling

# COMMAND ----------

# DBTITLE 1,Read Cluster Labeled Data
pivot_df['Cluster'] = y_pred_km
pivot_df['Cluster'] = pivot_df['Cluster'].astype(int)
pivot_df_org = pivot_df.copy()
print(f"pivot_df_org shape: {pivot_df_org.shape}")

# Write Pivot with cluster labels
# pivot_df.reset_index().to_csv('/dbfs/FileStore/tables/subhash_dixit/baseline_forecast_project/baseline_shape_analysis/us_ret_ppg_year_level_20_cluster_labels_reckitt_data.csv', index=False)

# Read the sam Pivot with cluster labels
# pivot_df = pd.read_csv("/dbfs/FileStore/tables/subhash_dixit/baseline_forecast_project/baseline_shape_analysis/us_ret_ppg_year_level_20_cluster_labels_reckitt_data.csv")
pos_df = pd.read_csv('/dbfs/FileStore/tables/subhash_dixit/baseline_forecast_project/baseline_shape_analysis/us_all_category_pos_reckitt_data.csv')

pos_df['PERIOD_ID'] = pd.to_datetime(pos_df['PERIOD_ID'])
pos_df['Year'] = pos_df['PERIOD_ID'].dt.year
pos_df['Week'] = pos_df['PERIOD_ID'].dt.isocalendar().week
valid_years = [2023, 2024]
pos_df = pos_df[pos_df['Year'].isin(valid_years)].copy()

# Aggregation at Retailer + PPG + Year level
agg_df = (
    pos_df.groupby(['RETAILER_ID', 'PPG_NM', 'Year'], as_index=False)
    .agg({
        'NET_SALES_WITH_TAXES': 'sum',  # sum sales
        'CATEGORY': 'first'             # keep category as-is
    })
)

print(f"pos_df shape: {pos_df.shape}")
print(f"agg_df shape: {agg_df.shape}")
display(agg_df.head())


# COMMAND ----------

pos_df[['RETAILER_ID', 'PPG_NM', 'Year']].drop_duplicates().shape

# COMMAND ----------

# DBTITLE 1,Merge Pivot with POS Data
# pos_df_with_clusters = pos_df.merge(
#     pivot_df,
#     on=['RETAILER_ID', 'PPG_NM', 'Year'],
#     how='left'
# )
pivot_df = pivot_df.merge(
    agg_df[['RETAILER_ID', 'PPG_NM', 'Year', 'CATEGORY', 'NET_SALES_WITH_TAXES']],
    on=['RETAILER_ID', 'PPG_NM', 'Year'],
    how='left'
)


# pos_df_with_clusters = pos_df_filtered.merge(
#     pivot_df_reset[['RETAILER_ID', 'PPG_NM', 'Year', 'Cluster']],
#     on=['RETAILER_ID', 'PPG_NM', 'Year'],
#     how='left'
# )

print(pivot_df['CATEGORY'].isnull().sum())
print(f"pivot_df shape: {pivot_df.shape}")

# COMMAND ----------

pivot_df.display()

# COMMAND ----------

# 🔹 Combine Retailer and PPG
pivot_df['ret_ppg_year'] = (
    pivot_df['RETAILER_ID'] + '_' + pivot_df['PPG_NM'] + '_' + pivot_df['Year'].astype(str)
)

# 🔹 Total per category
cat_summary = (
    pivot_df.groupby('CATEGORY')
    .agg(
        total_ret_ppg=('ret_ppg_year', 'nunique'),
        total_sales=('NET_SALES_WITH_TAXES', 'sum')
    )
    .reset_index()
)

# 🔹 Cluster-wise stats (count only)
cluster_summary = (
    pivot_df.groupby(['CATEGORY', 'Cluster'])
    .agg(cluster_ret_ppg=('ret_ppg_year', 'nunique'))
    .reset_index()
)

# Pivot to wide format — keep only counts
cluster_pivot = cluster_summary.pivot(
    index='CATEGORY', 
    columns='Cluster',
    values='cluster_ret_ppg'
).fillna(0)

# Rename columns for clarity
cluster_pivot.columns = [f"Cluster_{int(c)}_ret_ppg_count" for c in cluster_pivot.columns]

# Merge category totals with cluster info
final_df = cat_summary.merge(cluster_pivot, on='CATEGORY', how='left')

# 🔹 Identify week columns (W_1 to W_52)
week_cols = [col for col in pivot_df.columns if str(col).startswith('W_')]

# 🔹 Compute number of non-zero weeks per row
pivot_df['weeks_sold'] = (pivot_df[week_cols] > 0).sum(axis=1)

# 🔹 Aggregate by category
weeks_summary = (
    pivot_df.groupby('CATEGORY')
    .agg(
        total_weeks_sold=('weeks_sold', 'sum'),          # total count of weeks sold across all rows
        rows_in_category=('weeks_sold', 'count')         # number of ret-ppg-year rows in that category
    )
    .reset_index()
)

# 🔹 Compute fraction of available weeks (float 0–1)
weeks_summary['frac_weeks_sold'] = (
    weeks_summary['total_weeks_sold'] / (weeks_summary['rows_in_category'] * 52)
)

# 🔹 Merge week stats
final_df = final_df.merge(
    weeks_summary[['CATEGORY', 'total_weeks_sold', 'frac_weeks_sold']],
    on='CATEGORY', how='left'
)

# ✅ Display final output
display(final_df)


# COMMAND ----------

# # 🔹 Combine Retailer and PPG
# pivot_df['ret_ppg_year'] = pivot_df['RETAILER_ID'] + '_' + pivot_df['PPG_NM'] + '_' + pivot_df['Year'].astype(str)

# # 🔹 Total per category
# cat_summary = (
#     pivot_df.groupby('CATEGORY')
#     .agg(total_ret_ppg=('ret_ppg_year', 'nunique'),
#          total_sales=('NET_SALES_WITH_TAXES', 'sum'))
#     .reset_index()
# )

# # 🔹 Category fraction of total (as float 0–1)
# total_ret_ppg_all = cat_summary['total_ret_ppg'].sum()
# cat_summary['frac_of_total_ret_ppg'] = cat_summary['total_ret_ppg'] / total_ret_ppg_all

# # 🔹 Cluster-wise stats
# cluster_summary = (
#     pivot_df.groupby(['CATEGORY', 'Cluster'])
#     .agg(cluster_ret_ppg=('ret_ppg_year', 'nunique'))
#     .reset_index()
# )

# # Merge category totals
# cluster_summary = cluster_summary.merge(
#     cat_summary[['CATEGORY', 'total_ret_ppg', 'frac_of_total_ret_ppg']],
#     on='CATEGORY', how='left'
# )

# # Fraction within CATEGORY (0–1)
# cluster_summary['frac_within_CATEGORY'] = cluster_summary['cluster_ret_ppg'] / cluster_summary['total_ret_ppg']

# # Fraction relative to overall total (0–1)
# cluster_summary['frac_of_total'] = cluster_summary['frac_within_CATEGORY'] / cluster_summary['frac_of_total_ret_ppg']

# # Pivot to wide format
# cluster_pivot = cluster_summary.pivot(
#     index='CATEGORY', 
#     columns='Cluster',
#     values=['cluster_ret_ppg', 'frac_within_CATEGORY', 'frac_of_total']
# ).fillna(0)

# # Flatten columns
# cluster_pivot.columns = [f"Cluster_{c[1]}_{c[0]}" for c in cluster_pivot.columns]

# # Merge category summary with cluster info
# final_df = cat_summary.merge(cluster_pivot, on='CATEGORY', how='left')


# # 🔹 Identify week columns (W_1 to W_52)
# week_cols = [col for col in pivot_df.columns if str(col).startswith('W_')]

# # 🔹 Compute weeks sold per row (Retailer-PPG-Year)
# pivot_df['weeks_sold'] = (pivot_df[week_cols] > 0).sum(axis=1)

# # 🔹 Aggregate by category
# weeks_summary = (
#     pivot_df.groupby('CATEGORY')
#     .agg(
#         total_weeks_sold=('weeks_sold', 'sum'),          # sum of weeks sold across all ret-ppg-year
#         max_possible_weeks=('weeks_sold', 'count')       # number of ret-ppg-year rows
#     )
#     .reset_index()
# )

# # 🔹 Compute fraction of weeks sold wrt total possible weeks (52 weeks per row)
# weeks_summary['frac_weeks_sold'] = weeks_summary['total_weeks_sold'] / (weeks_summary['max_possible_weeks'] * 52)

# # 🔹 Merge with existing final_df
# final_df = final_df.merge(weeks_summary[['CATEGORY', 'total_weeks_sold', 'frac_weeks_sold']], on='CATEGORY', how='left')


# # ✅ Display final table
# display(final_df)


# COMMAND ----------

# Compare same Retailer-PPG combination across years
cluster_transition = (
    pivot_df.pivot_table(index=['RETAILER_ID', 'PPG_NM'],
                         columns='Year',
                         values='Cluster',
                         aggfunc='first')
    .reset_index()
)

# Count transitions like (Cluster in 2023 -> Cluster in 2024)
cluster_transition['Transition'] = cluster_transition[2023].astype(str) + ' (2023)  → ' + cluster_transition[2024].astype(str) + ' (2024)'

transition_summary = cluster_transition['Transition'].value_counts().reset_index()
transition_summary.columns = ['Transition', 'Count']
print("🔁 Cluster Transitions Across Years:")
display(transition_summary)


# COMMAND ----------

# MAGIC %md
# MAGIC 1 → 1	The Retailer–PPG was in Cluster 1 in Year 1 (e.g., 2023) and remained in Cluster 1 in Year 2 (e.g., 2024) — stable behavior ✅
# MAGIC 1 → 3	It was in Cluster 1 in 2023, but shifted to Cluster 3 in 2024 — changed annual pattern 🔄
# MAGIC 4 → 2	It was in Cluster 4 in 2023, and moved to Cluster 2 in 2024 — another transition

# COMMAND ----------

import pandas as pd

# 1️⃣ Reset index to expose RETAILER_ID, PPG_NM, and Year as columns
pivot_df_reset = pivot_df.reset_index()

# 2️⃣ Split PPG_NM into multiple attributes (assuming '|' delimiter)
# Example: "Beverage|Cola|500ml" → Attr1 = Beverage, Attr2 = Cola, Attr3 = 500ml
attr_cols = pivot_df_reset['PPG_NM'].str.split('|', expand=True)
attr_cols = attr_cols.apply(lambda x: x.str.strip())  # remove extra spaces

# 3️⃣ Assign proper column names dynamically
attr_cols.columns = [f'Attr{i+1}' for i in range(attr_cols.shape[1])]

# 4️⃣ Concatenate attributes with main pivot dataframe
pivot_df_reset = pd.concat([pivot_df_reset, attr_cols], axis=1)

# 5️⃣ Add cluster info (from your clustering output)
# pivot_df_reset['Cluster'] = y_pred_km  # or y_pred_dba_km depending on your model

# 6️⃣ Compute profiling metrics
cluster_profile = (
    pivot_df_reset
    .groupby('Cluster')
    .agg({
        'CATEGORY': lambda x: x.value_counts().index[0] if not x.empty else None,
        'RETAILER_ID': ['nunique', lambda x: x.value_counts().index[0] if not x.empty else None],
        'PPG_NM': 'nunique',
        'Year': 'nunique',
        'Attr1': lambda x: x.value_counts().index[0] if not x.empty else None,
        'Attr2': lambda x: x.value_counts().index[0] if not x.empty else None,
        'Attr3': lambda x: x.value_counts().index[0] if not x.empty else None,
        'Attr4': lambda x: x.value_counts().index[0] if not x.empty else None,
        'Attr5': lambda x: x.value_counts().index[0] if not x.empty else None,
        'Attr6': lambda x: x.value_counts().index[0] if not x.empty else None,
        'Attr7': lambda x: x.value_counts().index[0] if not x.empty else None,
        'Attr8': lambda x: x.value_counts().index[0] if not x.empty else None,
        'Attr9': lambda x: x.value_counts().index[0] if not x.empty else None,
        'Attr10': lambda x: x.value_counts().index[0] if not x.empty else None,
        'Attr11': lambda x: x.value_counts().index[0] if not x.empty else None,

    })
    .reset_index()
)

# 7️⃣ Rename columns for clarity
cluster_profile = cluster_profile.rename(columns={
    'CATEGROY': 'Top Category',
    'RETAILER_ID': 'Unique Retailers',
    'RETAILER_ID': 'Top Retailer',
    'PPG_NM': 'Unique Products',
    'Year': 'Years Covered',
    'Attr1': 'Top Attribute 1',
    'Attr2': 'Top Attribute 2',
    'Attr3': 'Top Attribute 3',
    'Attr4': 'Top Attribute 4',
    'Attr5': 'Top Attribute 5',
    'Attr6': 'Top Attribute 6',
    'Attr7': 'Top Attribute 7',
    'Attr8': 'Top Attribute 8',
    'Attr9': 'Top Attribute 9',
    'Attr10': 'Top Attribute 10',
    'Attr11': 'Top Attribute 11',

})
cluster_profile['Cluster'] = cluster_profile['Cluster'] + 1
print("📊 Cluster Profiling Summary:")
display(cluster_profile)


# COMMAND ----------

import pandas as pd
import numpy as np

# Identify attribute columns dynamically
attr_cols = [col for col in pivot_df_reset.columns if col.startswith('Attr') or col.startswith('RETAILER_ID') or col.startswith('CATEGORY')]

final_rows = []

# Loop through each cluster
for cluster in sorted(pivot_df_reset['Cluster'].unique()):
    cluster_data = pivot_df_reset[pivot_df_reset['Cluster'] == cluster]
    
    # Dictionary of top5 per attribute for this cluster
    top_attr_dict = {'Cluster': cluster}
    attr_top_values = {attr: [] for attr in attr_cols}
    
    for attr in attr_cols:
        vc = cluster_data[attr].value_counts(normalize=True).dropna() * 100
        vc = vc.head(5).round(2)
        
        for val, pct in zip(vc.index, vc.values):
            attr_top_values[attr].append(f"{val} – {pct:.1f}%")
        
        # If less than 5, pad with blanks
        while len(attr_top_values[attr]) < 5:
            attr_top_values[attr].append("")

    # Now create 5 rows for each cluster (Top 5 values)
    for i in range(5):
        row = {'Cluster': cluster+1}
        for attr in attr_cols:
            row[attr] = attr_top_values[attr][i]
        final_rows.append(row)

# Convert to final DataFrame
cluster_top5_df = pd.DataFrame(final_rows)

# Display cleanly
pd.set_option('display.max_columns', None)
display(cluster_top5_df)


# COMMAND ----------

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8,5))
sns.countplot(data=pivot_df.reset_index(), x='Cluster', hue='Year', palette='Set2')
plt.title("Cluster Distribution Across Years")
plt.xlabel("Cluster")
plt.ylabel("Count of Retailer-PPG Combinations")
plt.legend(title="Year")
plt.show()


# COMMAND ----------

END.......

# COMMAND ----------

# MAGIC %md
# MAGIC # Experiments

# COMMAND ----------

import pandas as pd
import numpy as np

# Seed for reproducibility
np.random.seed(42)

# Dummy data: 3 products, 3 years, 10 weeks
products = ['A', 'B', 'C']
years = [2021, 2022, 2023]
weeks = list(range(1, 11))

data = []
for prod in products:
    for year in years:
        # generate a base seasonal pattern + random noise
        base_pattern = np.linspace(100, 200, len(weeks))  # rising trend
        if prod == 'B':
            base_pattern += 50  # B is higher
        elif prod == 'C':
            base_pattern = np.linspace(200, 150, len(weeks))  # declining
        sales = base_pattern + np.random.normal(0, 5, len(weeks))
        for w, s in zip(weeks, sales):
            data.append([prod, year, w, s])

df = pd.DataFrame(data, columns=['Product', 'Year', 'Week', 'Sales'])
df.head(10)


# COMMAND ----------

annual_df = df.pivot_table(
    index=['Product','Year'],
    columns='Week',
    values='Sales'
).fillna(0)

annual_df


# COMMAND ----------

uid_1   3 years 1 2 .....157 -- 1
uid_2   3 years 1 2 .....157 -- 0
uid_3   3 years 1 2 .....157 -- 2



uid_1   1 years 1 2 .....52 -- 1
uid_1   1 years 1 2 .....52 -- 0
uid_1   1 years 1 2 .....52 -- 1
uid_2   1 years 1 2 .....52 -- 2
uid_2   1 years 1 2 .....52 -- 0
uid_2   1 years 1 2 .....52 -- 1
uid_3   1 years 1 2 .....52 -- 2
uid_3   1 years 1 2 .....52 -- 0
uid_3   1 years 1 2 .....52 -- 1

uid_1_2021
uid_1_2022
uid_1_2023





# COMMAND ----------

from tslearn.preprocessing import TimeSeriesScalerMeanVariance

X = annual_df.values.astype(float)
scaler = TimeSeriesScalerMeanVariance()
X_scaled = scaler.fit_transform(X[:, :, np.newaxis])  # shape = (n_series, n_timesteps, 1)


# COMMAND ----------

X_scaled

# COMMAND ----------

from tslearn.clustering import TimeSeriesKMeans

n_clusters = 3
km = TimeSeriesKMeans(n_clusters=n_clusters, metric="dtw", random_state=42)
y_pred = km.fit_predict(X_scaled)


# COMMAND ----------

y_pred

# COMMAND ----------

annual_df['Cluster'] = y_pred
annual_df


# COMMAND ----------

import matplotlib.pyplot as plt

colors = ['green', 'blue', 'orange']  # one color per cluster
plt.figure(figsize=(12,6))

for cl in range(n_clusters):
    # Plot all series in this cluster
    for xx in X_scaled[y_pred == cl]:
        plt.plot(xx.ravel(), color=colors[cl], alpha=0.3)
    # Plot centroid
    plt.plot(km.cluster_centers_[cl].ravel(), color='red', linewidth=2, label=f'Centroid Cluster {cl+1}')

plt.title("Annual Shape Clusters for Products")
plt.xlabel("Week")
plt.ylabel("Scaled Sales")
plt.legend()
plt.show()


# COMMAND ----------

for cl in range(n_clusters):
    plt.figure(figsize=(10,5))
    subset = X[y_pred==cl]
    for ts in subset:
        plt.plot(ts.ravel(), color='green', alpha=0.3)
    plt.plot(km.cluster_centers_[cl].ravel(), color='red', linewidth=2)
    plt.title(f"Cluster {cl} - Annual Shape")
    plt.xlabel("Week")
    plt.ylabel("Scaled Sales")
    plt.show()


# COMMAND ----------

# Recover approximate original scale for centroids
centroids_scaled_back = np.zeros_like(km.cluster_centers_)
for i in range(n_clusters):
    # Approximate by using global mean & std across series (since centroid is synthetic)
    centroids_scaled_back[i] = km.cluster_centers_[i] * X.std() + X.mean()

# COMMAND ----------

# for cl in range(n_clusters):
#     plt.figure(figsize=(10,5))
#     subset = X[y_pred==cl]
#     for ts in subset:
#         plt.plot(ts.ravel(), color='green', alpha=0.3)
#     plt.plot(km.cluster_centers_[cl].ravel(), color='red', linewidth=2)
#     plt.title(f"Cluster {cl} - Annual Shape")
#     plt.xlabel("Week")
#     plt.ylabel("Scaled Sales")
#     plt.show()

for cl in range(n_clusters):
    plt.figure(figsize=(10,5))
    for ts in X[y_pred==cl]:
        plt.plot(ts.ravel(), color='green', alpha=0.3)
    plt.plot(centroids_scaled_back[cl].ravel(), color='red', linewidth=2)
    plt.title(f"Cluster {cl} - Annual Shape")
    plt.show()


# COMMAND ----------

