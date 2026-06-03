Content Monetization Modeler

Description:- 

A synthetic dataset having 122,000 rows under the name YouTube Monetization Modeler is given where we are asked to find the ad_revenue_usd using the LINEAR REGRESSION model.

Approach:-

Step 1: The necessary libraries for handling the datas were imported.

Step 2: The dataset in the csv form was loaded to understand and inspect the data and its columns using pandas.
Step 3: Data Cleaning.

        -The duplicates were found and then dropped.
        -The missing values were found and then filled them with median. Our project contains metrics like views,likes.Median was used to fill the missing values since median is not affected by outliers. 

Step 4: After the data cleaning, the cleaned csv is used for further operations.

Step 5: Feature Engineering.
        - Here we created new columns from the existing columns to get accurate results
        - The features we created were,
            1.'engagement_rate' 
            2.'like_ratio' 
            3.'comment_ratio'
            4.'subscriber_ratio'
            5.'date','day_of_week','month'
            6.'watch_time_ratio"

Step 6: Outliers handling.
        - We detected outliers in the watch_time_ratio.Some values in the watch time were greater than the video lenght, so we dropped the watch time ratio and used the raw watch time minutes and capped them to 99 percentile.
        - We also capped the views and subscribers since they also tend to have outliers. After capping them the subscriber ratio was recalculated.

Step 7: Exploratory Data Analysis (EDA)
        -Here we created 6 charts to understand the data.

            1. Revenue distribution: From the revenue distribution, we found that ad revenue was uniformly distributed between $130-$370, with no significant skewness. This meant that we could use the original ad_revenue_usd values for modeling without needing a log transformation.

            2. Coorelation mapping:  we found that views, watch time, and subscribers had positive correlation with ad revenue, while engagement rate and like/comment ratios showed weaker correlations. Watch_time_minutes alone drives 96% of revenue correlation.

            3. Revenue by CATEGORY, DEVICE and COUNTRY - We found that all the categories, devices and top 10 countries earned ~$252 average ad revenue, with no significant differences between them. They were not much useful for our model.

            4. Views vs revenue scatter plot:  From the views vs revenue scatter plot, we understood that views alone was not enough for the revenue. We could see that the views matvhes with the heatmap correlation where it had a positive correlation with the revenue but not a strong one.

Step 8: Model building.
        - Here we used 5 models and trained them.
            1.linear regression
            2.ridge regression
            3.lasso regression
            4.random forest regressor
            5.gradient boosting regressor.

Step 9: Categorical encoding.
        - Here we encoded the categorical features (category, device, country) using Label Encoding to convert them into numerical format from sklearn.preprocessing import LabelEncoder.

Step 10: Performance evaluation.
         -We evaluated the performance of these models using Mean Absolute Error (MAE),Root Mean Squared Error(RMSE), and R-squared (R²) metrics to determine which model best predicted the ad revenue based on the features we engineered and selected.
        - R2 = values should be closer to 1.
        - MAE and RMSE = they are quite same but rmse is more sensitive to outliers. The lower the better for the both of them.
        

              Model           - R²  -  RMSE  -  MAE -

         Linear Regression   -0.9526 - 13.48 - 3.12 - *****
         Ridge Regression    -0.9526 - 13.48 - 3.12 - *****
         Lasso Regression    -0.9526 - 13.48 - 3.11 -
         Gradient Boosting   -0.9522 - 13.53 - 3.63 -
         Random Forest       -0.9496 - 13.90 - 3.60 -

From the results, we found that the best performing model was the one with the highest R² and lowest RMSE and MAE.
- Here we could see both the ridge and linear has same values but we chose linear regression as the best model.
- the reason is that linear regression is more simpler that ridge as ridge is used when features are highly correlated and       overfitting.
- Since we have a synthetic data, we might not have complex features so we use the simpler one that is the 
 LINEAR REGRESSION MODEL.

Step 11: Saving the model for deployment in stramlit using pickle.

Step 12: Visualization.
     - Using model visualzation, we once angain found that linear regression is the best fit.
     - We once again use visualization to find the best model using actual vs predicted values, feature importance and the residuals(actual-predicted)
From the visualizations we found that the watch time minutes were overshadowed by the ratios of likes,comment and engagement.
So we retrained the models removing the ratios(likes,engagemnet,comment)columns.

Step 12: Model retraining.
        - the model was retrained and was updated.
        - the model was scaled and saved.

Step 13: Saving the cleaned csv.

Step 14: Building the stramlit application.
        -We created a simple interface where users can input the performance metrics of their YouTube video and get an estimated ad revenue prediction based on our trained model.

Step 14.1: Importing the necessary libraries for it.

Step 14.2: loaded the trained model, scaler, and features list.
        - we used the cache resourses so that we can reuse it everytime.

Step 14.3: loaded the cleaned dataset for the visualizations.

Step 14,4: we created the app layout with a side bar having 3 pages -  home, predict revenue and the data insights.

        1.home - Project description and Model performance metrics (R², RMSE, MAE).
        2.predict revenue -  input video performance metrics
                          -  get instant revenue prediction
                          -  view video stats summary
        3.data insights - Dataset overview (total videos, avg/max revenue)
                        - Revenue distribution chart
                        - Revenue by category chart
                        - Revenue by device chart

Technologies used:

    Python — Core programming language
    Pandas — Data manipulation
    NumPy — Numerical operations
    Matplotlib / Seaborn — Data visualization
    Scikit-learn — Machine learning models
    Streamlit — Web application framework
    Pickle — Model serialization










