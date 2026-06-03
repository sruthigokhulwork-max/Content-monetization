# Step 1: Importing libraries

# from operator import le

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Step 2: Data loading 
df = pd.read_csv('youtube_ad_revenue_dataset.csv')

# Step 3: Data analysis
print(df.shape)
print(df.head())
print(df.info()) 
print(df.describe())

# # Step 3.1: Checking the missing values
missing_values =df.isnull().sum()
percent_missing = (missing_values / len(df)) * 100
missing_data = pd.DataFrame({'Missing Values': missing_values, 'Percentage': percent_missing.round(2)})
print(missing_data)

# # Step 3.2 Finding duplicates
duplicates = df.duplicated().sum()
percent_duplicates = (duplicates / len(df)) * 100
missing_duplicates = pd.DataFrame({'Duplicates': [duplicates], 'Percentage': [percent_duplicates.round(2)]})
print(missing_duplicates)

# Step 4: Data cleaning

# Step 4.1: Filling missing values
# Here we fill the missing values with median for numerical columns since its middle value is more stable than outliers.
# we found the missing values from the percent missing  from the previous step.

# likes - 5% missing
df['likes'] = df['likes'].fillna(df['likes'].median())

# comments - 4.99% missing
df['comments'] = df['comments'].fillna(df['comments'].median())

# watch_time_minutes - 4.99% missing
df['watch_time_minutes'] = df['watch_time_minutes'].fillna(df['watch_time_minutes'].median())

# checking for missing values after filling them with median
missing_values_after = df.isnull().sum()
print(missing_values_after)

# Step 4.2: Removing duplicates

#  Here we remove the duplicates from the dataset 
df.drop_duplicates(inplace=True)
print("df shape after removing duplicates:", df.shape)

# Step 5: Feature engineering
# It means creating new columns or features from the existing data to improve the performance.

# ENGAGEMENT RATE
# Here we create a new column called 'engagement_rate' which is the sum of likes and comments divided by the number of views.
# This metric helps us understand how engaged the audience is with the content.
df['engagement_rate'] = (df['likes'] + df['comments']) / df['views']

# LIKE TO VIEW RATIO
# Here we create a new column called "like_ratio"
# This metric helps us understand how many people liked the video giving us insight into the video's popularity 
# and audience reception.
df['like_ratio'] = df['likes']/df['views']

# COMMENT TO VIEW RATIO
# Here we create a new column called "comment_ratio"
# This metric helps us understand how many people commented on the video giving us insight into the video's engagement and audience interaction.
df['comment_ratio'] = df['comments']/df['views']

# WATCH TIME TO VIEW RATIO
# Here we create a new column called "watch_time_ratio"
# This metric helps us understand how much time viewers spend watching the video compared to the number of views.
df['watch_time_ratio'] = df['watch_time_minutes']/df['video_length_minutes']

# SUBSCRIBER TO VIEW RATIO
# Here we create a new column called "subscriber_ratio"
# This metric helps us understand how many subscribers the channel has compared to the number of views.
df['subscriber_ratio'] = df['subscribers']/df['views']

# Date-based features 
# Here we create new columns for the day of the week and month from the date needed
# This metric helps to understand how time patterns affect the ad revenue.
df['date'] = pd.to_datetime(df['date'])
df['day_of_week'] = df['date'].dt.dayofweek
df['month'] = df['date'].dt.month

print(df[['engagement_rate', 'like_ratio', 'comment_ratio', 
          'watch_time_ratio', 'subscriber_ratio', 'day_of_week', 'month']].head())

# Step 6: Outlier handling
# Here we fix the watch time ratio, views and subscribers outliers.
# Some values in the watch time is greater than the video lenght which is not posssible. So we cap them.

# Droping the problematic ratio column
df.drop(columns=['watch_time_ratio'], inplace=True)

# Instead, cap the raw watch_time_minutes at 99th percentile
cap_watch = df['watch_time_minutes'].quantile(0.99)
df['watch_time_minutes'] = df['watch_time_minutes'].clip(upper=cap_watch)

# Also cap views and subscribers (these also tend to have outliers)
cap_views = df['views'].quantile(0.99)
df['views'] = df['views'].clip(upper=cap_views)

cap_subs = df['subscribers'].quantile(0.99)
df['subscribers'] = df['subscribers'].clip(upper=cap_subs)

# Recalculate subscriber_ratio after capping
df['subscriber_ratio'] = df['subscribers'] / df['views']

print(df[['engagement_rate', 'like_ratio', 'comment_ratio', 
          'watch_time_minutes', 'subscriber_ratio', 'day_of_week', 'month']].head())

# print(df.columns.tolist())
# print(df.shape)

# Step 7: EDA 

# Step 7.1: Revenue distribution
# Here we visualize the distribution of ad revenue using histograms to identify the outliers.
# We also apply a log transformation to identify the skewness in the data and to make it more normally distributed, 
# which can be helpful for modeling.

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
sns.histplot(df['ad_revenue_usd'], bins=50, color='steelblue', kde=True)
plt.title('Revenue Distribution (Original)')
plt.xlabel('Ad Revenue (USD)')
plt.ylabel('Number of Videos')

plt.subplot(1, 2, 2)
sns.histplot(np.log1p(df['ad_revenue_usd']), bins=50, color='orange', kde=True)
plt.title('Revenue Distribution (Log Scale)')
plt.xlabel('Log of Ad Revenue')
plt.ylabel('Number of Videos')

plt.tight_layout()
plt.savefig('eda_1_revenue_distribution.png', dpi=150)
plt.show()


# Step 7.2: Correlation heatmap
# Here we visualize the correlation between the numerical features and the target variable (ad revenue) using a heatmap.
# It helps in calculating the correlation matrix, the Pearson correlation coefficient(r).
# The correlation coefficient ranges from -1 to 1, where:
# +1.0: Perfect positive correlation (as one goes up, the other goes up).
# -1.0: Perfect negative correlation (as one goes up, the other goes down).
# 0: No linear relationship at all.

# After the calculatiins, we use a heatmap to visualize the correlation values, where the color intensity 
# indicates the strength of the correlation.


numeric_cols = [
    'views', 'likes', 'comments', 'watch_time_minutes',
    'video_length_minutes', 'subscribers', 'ad_revenue_usd',
    'engagement_rate', 'like_ratio', 'comment_ratio',
    'subscriber_ratio', 'day_of_week', 'month'
]

plt.figure(figsize=(13, 9))
correlation = df[numeric_cols].corr()
sns.heatmap(
    correlation,
    annot=True,
    fmt='.2f',
    cmap='coolwarm',
    center=0,
    linewidths=0.5
)
plt.title('Correlation Heatmap — Which Features Relate to Revenue?')
plt.tight_layout()
plt.savefig('eda_2_correlation_heatmap.png', dpi=150)
plt.show()

# Step 7.3: Revenue by category
# this helps visualize which video categories generate more ad revenue on average, providing insights.
# This satisfies why we should use category as a feature in our linear regression model.

# Here we group the  rows by category and calculate the mean ad revenue for each category. 
# Then we sort the categories by their average revenue in descending order.
# Finally, we create a bar plot to visualize the average ad revenue for each category to identify which 
# categories are more uswully for generating ad revenue.

plt.figure(figsize=(13, 5))
category_revenue = (
    df.groupby('category')['ad_revenue_usd']
    .mean()
    .sort_values(ascending=False)
    .reset_index()
)
sns.barplot(data=category_revenue, x='category', y='ad_revenue_usd', palette='viridis')
plt.title('Average Ad Revenue by Content Category')
plt.xlabel('Category')
plt.ylabel('Average Revenue (USD)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('eda_3_revenue_by_category.png', dpi=150)
plt.show()

# Step 7.4: Revenue by device type----------------
# This helps visualize which device types generate more ad revenue on average, providing insights.
# This satisfies why we should use device type as a feature in our linear regression model.

# Here we group the rows by device type and calculate the mean ad revenue for each device type.
# Then we sort the device types by their average revenue in descending order.
# Finally, we create a bar plot to visualize the average ad revenue for each device type to
# identify which device types are more useful for generating ad revenue.


plt.figure(figsize=(8, 5))
device_revenue = (
    df.groupby('device')['ad_revenue_usd']
    .mean()
    .sort_values(ascending=False)
    .reset_index()
)
sns.barplot(data=device_revenue, x='device', y='ad_revenue_usd', palette='Set2')
plt.title('Average Ad Revenue by Device Type')
plt.xlabel('Device')
plt.ylabel('Average Revenue (USD)')
plt.tight_layout()
plt.savefig('eda_4_revenue_by_device.png', dpi=150)
plt.show()

# Step 7.5: Views vs Revenue Scatter Plot------------------
#  Here we create a scatter plot to visualize the direct relationship between the number of views and ad revenue.
# This helps us understand how strongly views correlate with revenue and whether there are any outliers or patterns in the data.

# Each point on the scatter plot represents a video, with the x-axis representing the number of views and
# the y-axis representing the ad revenue.
# A positive correlation would indicate that as views increase, ad revenue tends to increase as well, 
# which is a common expectation in YouTube monetization.

plt.figure(figsize=(9, 5))
plt.scatter(df['views'], df['ad_revenue_usd'], alpha=0.05, color='purple', s=5)
plt.title('Views vs Ad Revenue')
plt.xlabel('Views')
plt.ylabel('Ad Revenue (USD)')
plt.tight_layout()
plt.savefig('eda_5_views_vs_revenue.png', dpi=150)
plt.show()

# Step 7.6: Top 10 Countries by Revenue------------------
# Here we create a bar plot to show the 10 countries geenrating the most ad revenue on youtube.
# It helps us prove if geographical location significantly impacts revenue.

#  Here we group the rows by country and calculate the mean ad revenue for each country.
#  First we filter to get the top 10 countries that are more profitable in terms of ad revenue.
# then we group them to get the average ad revenue.
#  Finally, we create a bar plot to visualize the average ad revenue for the top 10 countries to identify which
# countries are more useful for generating ad revenue.

plt.figure(figsize=(12, 5))
top_countries = (
    df.groupby('country')['ad_revenue_usd']
    .mean()
    .sort_values(ascending=False)
    .head(10)
    .reset_index()
)
sns.barplot(data=top_countries, x='country', y='ad_revenue_usd', palette='Blues_d')
plt.title('Top 10 Countries by Average Ad Revenue')
plt.xlabel('Country')
plt.ylabel('Average Revenue (USD)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('eda_6_revenue_by_country.png', dpi=150)
plt.show()


# ----------INSIGHTS SUMMARY FROM EDA----------

# 1. From the revenue distribution, we see that ad revenue is uniformly distributed between $130-$370, 
# with no significant skewness. This means we can use the original ad_revenue_usd values for modeling without 
# needing a log transformation.

# 2. From the correlation heatmap, we find that views, watch time, and subscribers have positive correlation with ad revenue, 
# while engagement rate and like/comment ratios show weaker correlations. 
# Watch_time_minutes alone drives 96% of revenue correlation.

# 3. From the revenue by category, all categories earn ~$252 average ad revenue, with no significant differences between them.
# This may not be useful for our model.

# 4. From the revenue by device , same as the revenue by category, all the devices earn $252 average ad revenue
# showing no significant differences between them. This may not be useful for our model.

# 5. From the views vs revenue scatter plot, we can understand that views alone is not enough for the revenue.
# we can see that the views matvhes with the heatmap correlation where it has a positive correlation with the revenue 
# but not a strong one.

# 6. From the top 10 countries by revenue, we can understand that they also earn the same average ad revenue of $252, 
# showing no significant differences between them. This may not be useful for our model.


# Step 8: Model Building

# Here we use 5 models to predict the ad revenue:
# 1. Linear Regression
# 2. Ridge Regression
# 3. Lasso Regression
# 4. Random Forest Regressor
# 5. Gradient Boosting Regressor

# We will evaluate the performance of these models using Mean Absolute Error (MAE),Root Mean Squared Error(RMSE), and R-squared (R²) metrics 
# to determine which model best predicts ad revenue based on the features we have engineered and selected.
# Then the best performing model will be used for the streamlit.

# R2 = values should be closer to 1.
# MAE and RMSE = they are quite same but rmse is more sensitive to outliers. The lower the better for the both of them.

# Step 8.1: Importing the model libraries.
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# # Step8.2: Getting the necessary columns for the model building.
# # We will use the features that have correlation with the ad revenue and the ones that we have engineered.
# # They are mostly numerical features since linear regression works better with numerical data.

# Here we encode the categorical features (category, device, country) using Label Encoding to convert them into numerical format.
# from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

# Create the encoded columns
df['category_encoded'] = le.fit_transform(df['category'])
df['device_encoded'] = le.fit_transform(df['device'])
df['country_encoded'] = le.fit_transform(df['country'])

print("Encoded columns created!")
# print(df[['category', 'category_encoded', 
#           'device', 'device_encoded',
#           'country', 'country_encoded'
#           ]].head())

features = [
    'views', 'likes', 'comments', 'watch_time_minutes',
    'video_length_minutes', 'subscribers',
    'like_ratio', 'comment_ratio',
    'subscriber_ratio',
    'category_encoded', 'device_encoded', 'country_encoded'
]

x = df[features]
y = df['ad_revenue_usd']

print(f"Features shape: {x.shape}")
print(f"Target shape: {y.shape}")
print(f"\nFeatures used: {features}")

# Step 8.3: Splitting the data into training and testing sets.
# We will use 80% of the data for training and 20% for testing to evaluate the model's performance.

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# print(f"\nTraining set: {X_train.shape[0]} rows")
# print(f"Testing set:  {X_test.shape[0]} rows")

from sklearn.preprocessing import StandardScaler
import pickle

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 8.4: Definig the models.

models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Lasso Regression': Lasso(alpha=1.0),
    'Random Forest Regressor': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting Regressor': GradientBoostingRegressor(n_estimators=100,learning_rate=0.1, random_state=42)
}

# Step 8.5: Training the models and evaluating their performance.


print("\nMODEL TRAINING AND EVALUATION")

results = {}
for model_name, model in models.items():
    print(f"\nTraining {model_name}...")
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2   = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)

    results[model_name] = {
        'R2': round(r2, 4),
        'RMSE': round(rmse, 4),
        'MAE': round(mae, 4)
    }


    print(f"   R²   = {r2:.4f}  (closer to 1.0 = better)")
    print(f"   RMSE = {rmse:.4f} (lower = better, in USD)")
    print(f"   MAE  = {mae:.4f}  (lower = better, in USD)")

# Step 8.6: Comparing the model performance.

print("\nMODEL PERFORMANCE COMPARISON:")

results_df = pd.DataFrame(results).T
results_df = results_df.sort_values('R2', ascending=False)
print(results_df)

best_model_name = results_df.index[0]
print(f"\n Best Model: {best_model_name}")
print(f"   R² = {results_df.loc[best_model_name, 'R2']}")
print(f"   RMSE = {results_df.loc[best_model_name, 'RMSE']}")

# From the results, we can see that the best performing model is the one with the highest R² and lowest RMSE and MAE.

# Step 9: Saving the best model for deployment in Streamlit.

import pickle

best_model = models['Linear Regression']

with open('best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('features.pkl', 'wb') as f:
    pickle.dump(features, f)

print("\nBest model and scaler saved for deployment!")


# Step 10: Visualizing the model performance comparison.


model_names = list(results.keys())
r2_scores = [results[m]['R2'] for m in model_names]
rmse_scores = [results[m]['RMSE'] for m in model_names]
mae_scores = [results[m]['MAE'] for m in model_names]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# R² comparison (higher is better)
axes[0].barh(model_names, r2_scores, color='steelblue')
axes[0].set_title('R² Score (Higher = Better)')
axes[0].set_xlabel('R² Score')
axes[0].axvline(x=max(r2_scores), color='red', 
                linestyle='--', alpha=0.5)
for i, v in enumerate(r2_scores):
    axes[0].text(v - 0.005, i, f'{v:.4f}', 
                va='center', color='white', fontweight='bold')

# RMSE comparison (lower is better)
axes[1].barh(model_names, rmse_scores, color='coral')
axes[1].set_title('RMSE (Lower = Better)')
axes[1].set_xlabel('RMSE (USD)')
for i, v in enumerate(rmse_scores):
    axes[1].text(v - 0.5, i, f'{v:.2f}', 
                va='center', color='white', fontweight='bold')

# MAE comparison (lower is better)
axes[2].barh(model_names, mae_scores, color='mediumseagreen')
axes[2].set_title('MAE (Lower = Better)')
axes[2].set_xlabel('MAE (USD)')
for i, v in enumerate(mae_scores):
    axes[2].text(v - 0.1, i, f'{v:.4f}', 
                va='center', color='white', fontweight='bold')

plt.suptitle('Model Performance Comparison', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('model_1_comparison.png', dpi=150)
plt.show()

# From the visual comparison, we can easily see which model performs best across the different metrics,
# with the Linear Regression model showing the highest R² and lowest RMSE and MAE, confirming it as the best model 
# for predicting ad revenue based on our features.

# Step 11: Visualizing the actual vs predicted values for the best model.

best_model = models['Linear Regression']
y_pred = best_model.predict(X_test_scaled)

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.1, color='steelblue', s=5)

# Draw the perfect prediction line (diagonal)
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], 
         'r--', linewidth=2, label='Perfect Prediction')

plt.xlabel('Actual Revenue (USD)')
plt.ylabel('Predicted Revenue (USD)')
plt.title('Actual vs Predicted Revenue\n(Closer to red line = Better)')
plt.legend()
plt.tight_layout()
plt.savefig('model_2_actual_vs_predicted.png', dpi=150)
plt.show()

# if predictions are good,dots form a diagonal straight line.
# if predictions are bad, dots are scattered and do not follow the diagonal line.

# Step 12: Visualizing feature importance for the best model.

coefficients = best_model.coef_
feature_importance = pd.DataFrame({
    'Feature': features,
    'Importance': abs(coefficients)
}).sort_values('Importance', ascending=True)

plt.figure(figsize=(10, 6))
plt.barh(feature_importance['Feature'], 
         feature_importance['Importance'], 
         color='mediumpurple')
plt.title('Feature Importance\n(Which features matter most?)')
plt.xlabel('Coefficient Magnitude')
plt.tight_layout()
plt.savefig('model_3_feature_importance.png', dpi=150)
plt.show()

# From the feature importance plot, we can identify which features have multicollinearity issues.

# Step 13: Visualizing the residuals for the best model.
# This is done to check if model errors are random (good) or patterned (bad)
# Residual = Actual - Predicted
# If residuals are randomly scattered around zero, it indicates a good fit.
# If residuals show a pattern (e.g., funnel shape, curve), it suggests model is not good.

residuals = y_test - y_pred

plt.figure(figsize=(9, 5))
plt.scatter(y_pred, residuals, alpha=0.1, color='orange', s=5)
plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
plt.xlabel('Predicted Revenue (USD)')
plt.ylabel('Residuals (Actual - Predicted)')
plt.title('Residual Plot\n(Random scatter around 0 = Good model)')
plt.tight_layout()
plt.savefig('model_4_residuals.png', dpi=150)
plt.show()

# from this we can understand that multicollinearity.


# From all these 3 visualizations,we can see that the watch time minutes was overshadowed by the engagement rate and the like/comment ratios,
# so we will remove the like_ratio, comment_ratio and engagement_rate features to see if the model performance improves.
# As a reult, we will have to retrain the model with the cleaned features.

# STEP 10: Retrain with Cleaned Features

# Simplified features — removing redundant ratios that were causing multicollinearity issues.
# multicollinearity - two or more of your input features are highly correlated with each other.

features = [
    'watch_time_minutes',
    'views',
    'likes',
    'comments',
    'subscribers',
    'video_length_minutes',
    'category_encoded',
    'device_encoded',
    'country_encoded'
]

x = df[features]
y = df['ad_revenue_usd']

X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

best_model = LinearRegression()
best_model.fit(X_train_scaled, y_train)
y_pred = best_model.predict(X_test_scaled)

r2   = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae  = mean_absolute_error(y_test, y_pred)

print(f"R²:   {r2:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE:  {mae:.4f}")
print(f"Min prediction: {y_pred.min():.2f}")
print(f"Max prediction: {y_pred.max():.2f}")
print(f"Min actual:     {y_test.min():.2f}")
print(f"Max actual:     {y_test.max():.2f}")

# Save updated model and scaler
import pickle
with open('best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('features.pkl', 'wb') as f:
    pickle.dump(features, f)

print("\nUpdated model saved!")

df.to_csv('cleaned_youtube_data.csv', index=False)
print(" Cleaned data saved!")