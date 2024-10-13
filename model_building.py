# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 15:25:19 2021

@author: Abhiram
"""

import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
import statsmodels.api as sm
import seaborn as sns
import pandas as pd

pd.set_option('display.max_columns', None)

df = pd.read_csv(r'C:\Users\abhir\Projects\DS_Salary_Project\eda_data.csv')

len(df.columns)

# choose relevant columns 
df_model = df[['avg_salary','Rating','Size','Type of ownership','Industry','Sector','Revenue','num_competitors','hourly','employer_provided',
             'job_state','same_state','age','python_yn','spark','aws','excel','job_simple','seniority','desc_len']]


# get dummy data 
df_dum = pd.get_dummies(df_model)

X = df_dum.drop('avg_salary', axis =1)
y = df_dum.avg_salary.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# multiple linear regression using OLS
# Prepare data for statsmodels
X_train_sm = sm.add_constant(X_train)  # Add constant for intercept
X_train_sm = X_train_sm.astype(float)   # Ensure float type
y_train = y_train.astype(float)          # Ensure float type

# Fit the model
model = sm.OLS(y_train, X_train_sm).fit()

# Print summary of the model
print(model.summary())


# Initialize a DataFrame to store the results
results = pd.DataFrame(columns=['Model', 'MAE', 'MSE', 'RMSE', 'R-squared'])

# Multiple Linear Regression
lm = LinearRegression()
lm.fit(X_train, y_train)
y_pred = lm.predict(X_test)

# Evaluate
results = pd.concat([results, pd.DataFrame([{
    'Model': 'Linear Regression',
    'MAE': mean_absolute_error(y_test, y_pred),
    'MSE': mean_squared_error(y_test, y_pred),
    'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
    'R-squared': r2_score(y_test, y_pred)
}])], ignore_index=True)

# Lasso Regression
lasso = Lasso()
param_grid = {'alpha': [0.09, 0.1, 0.11, 0.12, 0.13]}
grid_search = GridSearchCV(estimator=lasso, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

best_lasso = grid_search.best_estimator_
y_pred_lasso = best_lasso.predict(X_test)

results = pd.concat([results, pd.DataFrame([{
    'Model': 'Lasso',
    'MAE': mean_absolute_error(y_test, y_pred_lasso),
    'MSE': mean_squared_error(y_test, y_pred_lasso),
    'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_lasso)),
    'R-squared': r2_score(y_test, y_pred_lasso)
}])], ignore_index=True)

# Ridge Regression
ridge = Ridge()
ridge.fit(X_train, y_train)
y_pred_ridge = ridge.predict(X_test)

results = pd.concat([results, pd.DataFrame([{
    'Model': 'Ridge',
    'MAE': mean_absolute_error(y_test, y_pred_ridge),
    'MSE': mean_squared_error(y_test, y_pred_ridge),
    'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_ridge)),
    'R-squared': r2_score(y_test, y_pred_ridge)
}])], ignore_index=True)

# Random Forest Regression
rf = RandomForestRegressor(random_state=42)
param_grid_rf = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

grid_search_rf = GridSearchCV(estimator=rf, param_grid=param_grid_rf, cv=5, n_jobs=-1, verbose=2)
grid_search_rf.fit(X_train, y_train)

y_pred_rf = grid_search_rf.best_estimator_.predict(X_test)

results = pd.concat([results, pd.DataFrame([{
    'Model': 'Random Forest',
    'MAE': mean_absolute_error(y_test, y_pred_rf),
    'MSE': mean_squared_error(y_test, y_pred_rf),
    'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_rf)),
    'R-squared': r2_score(y_test, y_pred_rf)
}])], ignore_index=True)

# XGBoost Regression
xgb = XGBRegressor(random_state=42)
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)

results = pd.concat([results, pd.DataFrame([{
    'Model': 'XGBoost',
    'MAE': mean_absolute_error(y_test, y_pred_xgb),
    'MSE': mean_squared_error(y_test, y_pred_xgb),
    'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_xgb)),
    'R-squared': r2_score(y_test, y_pred_xgb)
}])], ignore_index=True)


# Hyperparameter grid for XGBoost
param_grid_xgb = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7, 10],
    'min_child_weight': [1, 3, 5],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'gamma': [0, 0.1, 0.2]
}

grid_search_xgb = GridSearchCV(estimator=XGBRegressor(random_state=42), 
                                 param_grid=param_grid_xgb, 
                                 cv=5, n_jobs=-1, verbose=2)
grid_search_xgb.fit(X_train, y_train)

y_pred_xgb = grid_search_xgb.best_estimator_.predict(X_test)

# Update results DataFrame
results = pd.concat([results, pd.DataFrame([{
    'Model': 'XGBoost_grid',
    'MAE': mean_absolute_error(y_test, y_pred_xgb),
    'MSE': mean_squared_error(y_test, y_pred_xgb),
    'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_xgb)),
    'R-squared': r2_score(y_test, y_pred_xgb)
}])], ignore_index=True)


from sklearn.model_selection import RandomizedSearchCV

random_search_xgb = RandomizedSearchCV(estimator=XGBRegressor(random_state=42),
                                       param_distributions=param_grid_xgb,
                                       n_iter=100,
                                       cv=5,
                                       n_jobs=-1,
                                       verbose=2,
                                       random_state=42)

random_search_xgb.fit(X_train, y_train)

y_pred_random_xgb = random_search_xgb.best_estimator_.predict(X_test)

# Update results DataFrame with Randomized Search results
results = pd.concat([results, pd.DataFrame([{
    'Model': 'XGBoost_random_grid',
    'MAE': mean_absolute_error(y_test, y_pred_random_xgb),
    'MSE': mean_squared_error(y_test, y_pred_random_xgb),
    'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_random_xgb)),
    'R-squared': r2_score(y_test, y_pred_random_xgb)
}])], ignore_index=True)


from lightgbm import LGBMRegressor

# Rename columns to remove special characters and spaces
X_train.columns = X_train.columns.str.replace(' ', '_').str.replace('[^A-Za-z0-9_]', '', regex=True)
X_test.columns = X_test.columns.str.replace(' ', '_').str.replace('[^A-Za-z0-9_]', '', regex=True)


lgbm_model = LGBMRegressor()
lgbm_model.fit(X_train, y_train)
y_pred_lgbm = lgbm_model.predict(X_test)

# Update results for LightGBM
results = pd.concat([results, pd.DataFrame([{
    'Model': 'LightGBM',
    'MAE': mean_absolute_error(y_test, y_pred_lgbm),
    'MSE': mean_squared_error(y_test, y_pred_lgbm),
    'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_lgbm)),
    'R-squared': r2_score(y_test, y_pred_lgbm)
}])], ignore_index=True)

# Print updated results
print(results)




# Plotting the results
results.set_index('Model')[['MAE', 'MSE', 'RMSE', 'R-squared']].plot(kind='bar', figsize=(12, 6))
plt.title('Model Comparison')
plt.ylabel('Metric Value')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()