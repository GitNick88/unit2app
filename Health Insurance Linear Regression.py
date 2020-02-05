#!/usr/bin/env python
# coding: utf-8

# In[102]:


get_ipython().system('pip install category_encoders')
get_ipython().system('pip install xgboost')
get_ipython().system('pip install plotly_express')
get_ipython().system('pip install dash')


# In[103]:


import pandas as pd
import plotly_express as px
import numpy as np

df = pd.read_csv('https://raw.githubusercontent.com/GitNick88/GitNick88.github.io/master/insurance.csv')

# Display more columns 
pd.set_option('display.max_rows', 500)

print(df.shape)
df.head(25)


# In[104]:


df['smoker'] = df['smoker'].astype(str)


# In[105]:


# Types of data in the df
df.dtypes


# In[106]:


df.describe()


# In[107]:


# Checking for high cardinality columns
df.describe(exclude='number').T.sort_values(by='unique')


# In[108]:


# Train/test split 80/20
from sklearn.model_selection import train_test_split

train, test = train_test_split(df, test_size=0.2)

print(train.shape)
test.shape


# In[109]:


# Train/val split 80/20
from sklearn.model_selection import train_test_split

train, val = train_test_split(train, test_size=0.2)

print(train.shape)
print(val.shape)
test.shape


# In[110]:


from sklearn.linear_model import LinearRegression

target = 'charges'
features = train.columns.drop(target)
X_train = train[features]
y_train = train[target]
X_val = val[features]
y_val = val[target]
X_test = test[features]
y_test = test[target]

model = LinearRegression()

print(X_train.shape)
print(X_val.shape)
print(X_test.shape)
print(y_val.shape)
print(y_train.shape)
y_test.shape


# In[111]:


from sklearn.metrics import r2_score
from xgboost import XGBRegressor

gb = make_pipeline(
    ce.OrdinalEncoder(),
    StandardScaler(),
    XGBRegressor(n_estimators=200, objective='reg:squarederror', n_jobs=-1)
)

gb.fit(X_train, y_train)
y_pred = gb.predict(X_val)
print('Gradient Boosting R^2', r2_score(y_val, y_pred))


# In[112]:


# BASELINE
guess = df['charges'].mean()
print(f'Just guessing, we would predict that insurance will cost ${guess:,.2f} per customer.  This is our model baseline.')


# In[113]:


# Train Error
from sklearn.metrics import mean_absolute_error
y_pred = [guess] * len(y_train)
mae_train = mean_absolute_error(y_train, y_pred)
print(f'Train Error: ${mae_train:.2f}')

# Test Error
y_pred = [guess] * len(y_test)
mae_test = mean_absolute_error(y_test, y_pred)
print(f'Test Error: ${mae_test:.2f}')
print('The train and test error are better than our baseline:')


# In[114]:


# What would be the average cost if I didn't insure anyone who smokes?
df_no_smoke= df.loc[df['smoker'] == 'no']

df_no_smoke.head()


# In[115]:


# New baseline to my costs if I do not insure smokers
no_smoke_y_pred = df_no_smoke['charges'].mean()

no_smoke_y_pred


# In[116]:


# Let's see what happens to average costs when we remove obesity values > 30
df_obese = df.loc[df['bmi'] < 26]

df_obese.head()


# In[117]:


# Average healthcare costs for non obese people

df_obese_cost = df_obese['charges'].mean()

df_obese_cost


# In[118]:


# Remove obese (bmi > 30) and smokers to see total charges
# Baseline prediction of avg total charges was $13,152.18

df_goodbmi_nosmoke = df.loc[df['smoker'] == 'no']

df_goodbmi_nosmoke.head(25)


# In[119]:


df_goodbmi_nosmoke = df_goodbmi_nosmoke[df_goodbmi_nosmoke['bmi'] < 26]

df_goodbmi_nosmoke.head(25)


# In[120]:


# Average cost to the insurance company without obese (bmi > 30) and without smoker is $7977.
# What if we dropped the bmi paramter down to 26 (still considered overweight)?

df_goodbmi_nosmoke['charges'].mean()


# In[121]:


df.head()


# In[122]:


# Who is more expensive?  Men or women?

df_male = df.loc[df['sex'] == 'male']

df_female = df.loc[df['sex'] == 'female']


# In[123]:


df_male['charges'].mean()


# In[124]:


df_female['charges'].mean()


# In[125]:


df.describe()


# In[126]:


gb.fit(X_train, y_train)


# In[127]:


# Predict on age	sex	bmi	children	smoker	region	charges

# Apply the model to new data
age = 27
sex = 1
bmi = 24
children = 2
smoker = 1
region = 1
X_test1 = [age, sex, bmi, children, smoker, region]
y_pred = pipeline.predict(X_test1)
print(y_pred1)


# In[ ]:


# Visual Ideas:
# At least one visual of my code
# Feature importances chart
# A visual of the r^2 score
# A visual for the mean of costs with and without smokers


# In[128]:


# Get feature importances
rf = gb.named_steps['xgbregressor']
importances = pd.Series(rf.feature_importances_, X_train.columns)

# Plot feature importances
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

n = 6
plt.figure(figsize=(10,n/2))
plt.title(f'Top features')
importances.sort_values()[-n:].plot.barh(color='grey');


# In[137]:


import category_encoders as ce
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

pipeline = make_pipeline(
    ce.TargetEncoder(),  
    LinearRegression()
)

pipeline.fit(X_train, y_train)
print('Linear Regression R^2', pipeline.score(X_val, y_val))


# In[185]:


# Create ex_data as a dataframe with column headers and "predictions"
ex_data = {'age': [27], 'sex': ['male'], 'bmi': [24], 'children': [2], 'smoker': ['no'], 'region': ['northwest']}


ex_data = pd.DataFrame(ex_data)

new_prediction = np.array2string(pipeline.predict(ex_data))
# new_prediction
print('This is the cost for someone with the above features:', new_prediction)


# In[167]:


# Shap value


# In[ ]:


# Create ex_data as a dataframe with column headers and "predictions"

sex = input("Gender? male or female: ")
bmi = input("What's your bmi ?")
children = input("How many children do you have?")
age = input("What's your age?")
region = input("Region? ex: northwest:")
smoker = input("Are you a smoker? yes or no:")

ex_data = {'age': [age], 'sex': [sex], 'bmi': [bmi], 'children': [children], 'smoker': [smoker], 'region': [region]}

ex_data = pd.DataFrame(ex_data)

new_prediction = np.array2string(pipeline.predict(ex_data))
# new_prediction
print('This is the cost for someone with the above features:', new_prediction)


# In[ ]:


df['bmi'].value_counts


# In[ ]:




