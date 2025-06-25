#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 12:43:55 2025

@author: miacompton
"""

## importing modules
import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
import seaborn as sns

##--------------------- IMPORTING AND READING DATA -------------------------##

skiprows = [0,1,2,3] # skip first lines in data to remove =# and column titles
usecols = [0,1,2,3,4,5,6,7] # ensures use of only necessary columns

# define column names
columnNames = ['material', 'density (kgm^-3)', 'radius (m)', 
               'mass (kg)', 'temperature (K)', 'pressure (Pa)', 
               'height (m)', 'time (s)']
# read using pandas
freefallData = pd.read_csv('exercise3data.csv', skiprows = skiprows, usecols= usecols, names= columnNames)
print(f'this is the information on the original dataset:\n')
freefallData.info() # describe the data
print('--------')

materialList = freefallData['material'].unique() # identify unique materials
print(f'\nthese are the different materials in the original dataset: \n{materialList}')

#---------------------------- CLEANING DATA ------------------------------------#

data = pd.DataFrame(freefallData) # convert to pandas dataframe

data = data.drop(data[data['material'] == 'm'].index) # remove random m

## convert to practical values 
data.iloc[:, 1:] = data.iloc[:, 1:].apply(pd.to_numeric, errors='coerce') 
data = data.dropna()  # drop rows with null values
data = data[(data.iloc[:, 1:] >= 0).all(axis=1)] # remove negative values, excluding material column

## showing clean data
print(f'\n--------')
print(f'\nthis is the info on the clean data: ')
print({data.info()})

#------------------------ FINDING MAX AND MIN VALUES --------------------------------#

print(f'\n------------------------------------------------')
print('These are the maximum values for each column:')
print(data.iloc[:, 1:].max())  # find max for only numeric columns
print('------------------------------------------------')
print('These are the minimum values for each column:')
print(data.iloc[:, 1:].min()) # find min for only numeric columns


##-------------------------- PLOTTING -------------------------- ##

## group by material and show together on scatter plot
for material, group in data.groupby('material'):                          
    plt.scatter(group['height (m)'], group['time (s)'], label=material, alpha=0.7)
    
plt.ylabel('Fall Time (s)')
plt.xlabel('Drop Height (m)')
plt.title('Plot of time against fall height for each material')
plt.legend(title="Material", bbox_to_anchor=(1, 1)) # bbox puts legend outside 
plt.show()

# create separate plots for each material
# show different radii separately
for material, material_group in data.groupby('material'):
    plt.figure(figsize=(4, 3))      

    # group by radius
    for radius, radius_group in material_group.groupby('radius (m)'):
        plt.scatter(x=radius_group['height (m)'], y=radius_group['time (s)'], label=f'Radius= {radius} m', alpha=0.7)    
        
    plt.xlabel('Drop Height (m)')
    plt.ylabel('Fall Time (s)')
    plt.title(f'Fall time against initial drop height for {material}')
    plt.legend(title="Radius (m)", bbox_to_anchor=(1, 1))  # Legend outside
    plt.show()


## use seaborn heatmap for correlation

sns.heatmap(data.iloc[:, 1:].corr(), annot=True) #ignore material column
                            
## import necessary modules

from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor

##################   OLS Regression   ########################

## define a function to model regression

def lin_reg_model(data, X):
    """
    function to find linear regression, where:
    data = dataframe
    attributes =  arrays of values to plot against time
    coeff_dict = returned dictionary of coefficients, intercept, attribute columns, and MSE
    """

    ## define X (arrays of attibutes in data) and y (time)
    y = data['time (s)']
    
    ## using the model
    model = linear_model.LinearRegression()
    model.fit(X, y)
    
    ## get coefficients (beta values) and intercept
    coeffs = model.coef_
    intercept = model.intercept_
    
    ## find model predicted values and MSE
    y_true = y.values
    y_pred = model.predict(X)
    mse = mean_squared_error(y_true, y_pred)
    
    ## create dictionary of coefficients with each attribute and store in results
    coeff_dict = dict(zip(attributes, coeffs)) # 'zip' zips together corresponding data
    
    results = {'coefficients': coeff_dict, 'intercept': intercept, 'mse': mse, 'model': model}

    ## return dictionary of results
    return results

## use numeric attribute columns
attributes = ['density (kgm^-3)', 'radius (m)', 'mass (kg)', 'temperature (K)', 'pressure (Pa)', 'height (m)']

X = data[attributes]

## call function
results = lin_reg_model(data, X)

## print results
print("OLS Regression Coefficients:")
print('-----------------------------')

## loop over attributes and each coefficient in results dictionary to print
for attribute, coeff in results['coefficients'].items():
    print(f'{attribute}: {coeff:.5f}')

## display results
print(f'Intercept: {results['intercept']:.5f}')
print('\nMean squared error in OLS regression:')
print(f'\n{results['mse']:.3f}')

##################   Seaborn Regression Plot   ########################

attributes = ['density (kgm^-3)', 'radius (m)', 'mass (kg)', 'temperature (K)', 'pressure (Pa)', 'height (m)']
for i, attribute in enumerate(attributes):

    sns.regplot(data=data, x=attribute, y='time (s)', line_kws={'color': 'red', 'label': 'True regression'})
        
    ## add calculated reegression line
    x_range = np.linspace(data[attribute].min(), data[attribute].max(), 100)
    y_pred = coeffs[i] * x_range + intercept
    plt.plot(x_range, y_pred, 'k--', label='Model regression')
        
    plt.xlabel(attribute)
    plt.ylabel('Fall Time (s)')
    plt.title(f'fall time against {attribute}')
    plt.legend()
    plt.show()
    
    
from scipy.stats import linregress

def radius_regression(data, material_x):
    """
    define regression plots for each material
    keeps density constant
    one plot for each radius
    where:
    data = pandas dataframe
    material_x = material to input for plotting
    """
    
    ## extract separate data
    material_data = data[data['material'] == material_x] # finds data for material x
    unique_radii = material_data['radius (m)'].unique() # finds unique radii for each x

    ## loop over each radius to plot separately
    for radius in unique_radii:
       radius_group = material_data[material_data['radius (m)'] == radius]

       ## seaborn regression plot; plots height against fall time for each radius
       sns.regplot(x=radius_group['height (m)'], y=radius_group['time (s)'], 
                  label=f'Radius= {radius} m', line_kws={'color': 'red'})
        
       result = linregress(x=radius_group['height (m)'], y=radius_group['time (s)'])
       slope = result.slope
       intercept = result.intercept
        
       plt.xlabel('Drop Height (m)')
       plt.ylabel('Fall Time (s)')
       plt.title(f'Fall time against initial drop height for {material_x} sphere\nr = {radius}m')
       plt.show()
       print(f"Slope of the regression line for {radius}m: {slope:.2f}")

radius_regression(data, 'iron')
radius_regression(data, 'polycarbonate')

from sklearn.preprocessing import StandardScaler

## mean absolute error

mae = mean_absolute_error(ytrue, ypred) # MSE in sklearn package
print(f'mean absolute error:\n {mae}')


#huber = modified_huber(ytrue, ypred)
#print(f'modified huber loss:\n{huber}')

## -------------------------------lasso ----------------------------------- ##

## lasso regression
lasso = linear_model.Lasso().fit(X, y)
lasso_coefs = lasso.coef_
lasso_loss = lasso.score(X, y)
print(f'\nlasso loss coefficients:')
print('----------')
for i, col in enumerate(X.columns):
    print(f"{col}: {lasso_coefs[i]:.5f}")
print(f'Lasso regression loss: \n{lasso_loss}')


## ------------------------------- ridge ----------------------------------- ##
## ridge regression  

ridge = linear_model.Ridge().fit(X, y)
ridge_coefs = ridge.coef_
ridge_loss = ridge.score(X, y)
print(f'\n------------------------------------')
print(f'\nRidge loss coefficients:')
print('----------')
for i, col in enumerate(X.columns):
    print(f"{col}: {ridge_coefs[i]:.5f}")
print(f'Ridge regression loss: \n{ridge_loss:.3f}')


#--------------------------------------------------------------------------------------------
#                             STOCHASTIC GRADIENT DESCENT 
#--------------------------------------------------------------------------------------------

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

SGDmodel = linear_model.SGDRegressor(loss = 'squared_error')
SGDmodel.fit(X_scaled, y)
SGDcoeffs = SGDmodel.coef_

# Print intercept
print('coefficients in Stochastic Gradient Descent:')
for i, col in enumerate(X.columns):
    print(f"{col}: {SGDcoeffs[i]:.5f}")
print('-------------------------------------------------')
print("Intercept:", SGDmodel.intercept_)


#--------------------------------------------------------------------------------------------
#                             MINI-BATCH GRADIENT DESCENT
#--------------------------------------------------------------------------------------------

## generate parameters
batch_size = 20  # batch size
n_epochs = 1000  # number of epochs
eta0 = 0.01  # learning rate

## compute the regression
mini_batch_model = SGDRegressor(
    loss='squared_error',
    learning_rate='constant',
    eta0=eta0,
    random_state=42,
    average=True)
y = data['time (s)'].values

n_samples = X_scaled.shape[0]

## computing the mini-batches
for epoch in range(n_epochs):
    shuffled_indices = np.random.permutation(n_samples)    # shuffle data for randomness
    X_shuffled = X_scaled[shuffled_indices]
    y_shuffled = y[shuffled_indices]

    ## iterate over batches
    for i in range(0, n_samples, batch_size):
        X_batch = X_shuffled[i:i + batch_size]
        y_batch = y_shuffled[i:i + batch_size]

        ## update using batches
        mini_batch_model.partial_fit(X_batch, y_batch)

## compare predictions
y_pred_mini_batch = mini_batch_model.predict(X_scaled)
mse_mini_batch = mean_squared_error(y, y_pred_mini_batch)

## show results
print('------------------------------')
print(f'\nMini-Batch Gradient Descent:')
for col, coef in zip(X.columns, mini_batch_model.coef_):
    print(f"{col}: {coef:.5f}")
print(f"MSE: {mse_mini_batch:.5f}")

from sklearn.model_selection import train_test_split

modelP4 = linear_model.LinearRegression()
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = 0.1)
modelP4.fit(X_train, y_train)

betas = np.zeros(6)
betas = modelP4.coef_
testIntercept = modelP4.intercept_
residuals = X_test.dot(betas) - y_test +testIntercept

print(f'\nTest coefficients:')
print('------------------------------------')
for i, col in enumerate(X.columns):
    print(f"{col}: {betas[i]:.5f}")

print(f'intercept: {testIntercept}')

residual_mean = np.mean(residuals)
residual_std = np.std(residuals)
print(f"Residual Mean: {residual_mean:.5f}")
print(f"Residual Standard Deviation: {residual_std:.5f}")

# Plot residuals
plt.figure(figsize=(10, 6))
plt.scatter(range(len(residuals)), residuals, alpha=0.7)
plt.axhline(y=0, color='r', linestyle='--')  # Add a horizontal line at y=0
plt.title("Residuals Plot")
plt.xlabel("Data Point Index")
plt.ylabel("Residuals")
plt.show()


## define constants
g = 9.81  # gravitational acceleration (m/s^2)
drag = 0.47  # drag coefficient for sphere
R = 8.314462  # molar gas constant (J/K/mol)
Mmol = 0.0289652  # molar mass of dry air (kg/mol)


## create function for comparisons
def predicted_time(h, m, radius, pressure, temperature):
    """ 
    function to find time using equation (8)
    h = height 
    """
    # Calculate air density using ideal gas law
    rho = (pressure * Mmol) / (R * temperature)
    
    # Calculate cross-sectional area (assuming sphere)
    A = np.pi * radius**2
    
    # Calculate drag coefficient k
    k = (drag * rho * A) / 2
    
    # Calculate time using analytical solution
    t = np.sqrt(m/(k*g)) * np.arccosh(np.exp(h*k/m))
    return t

# Generate heights outside data range
h_low = np.linspace(0, 500, 100)
h_high = np.linspace(1000, 1500, 100)

# Get sample data point for parameters (using first row)
sample = data.iloc[0]
m = sample['mass (kg)']
r = sample['radius (m)']
p = sample['pressure (Pa)']
T = sample['temperature (K)']

# Calculate analytical times
t_low = [predicted_time(h, m, r, p, T) for h in h_low]
t_high = [predicted_time(h, m, r, p, T) for h in h_high]

#X = data[attributes]

# Fit linear model
test_model = linear_model.LinearRegression()
test_model.fit(X, y)

# Create prediction dataframes with same features as training
X_low = pd.DataFrame(columns=attributes)
X_high = pd.DataFrame(columns=attributes)

# Fill with sample data
for attribute in attributes:
    if attribute != 'height (m)':
        X_low[attribute] = [sample[attribute]] * len(h_low)
        X_high[attribute] = [sample[attribute]] * len(h_high)

# Add heights
X_low['height (m)'] = h_low
X_high['height (m)'] = h_high

# Make predictions
below_dataRange = model.predict(X_low)
above_dataRange = model.predict(X_high)

# Plot comparisons
plt.figure(figsize=(12,5))

# Low heights
plt.subplot(121)
plt.plot(h_low, t_low, 'b-', label='Analytical')
plt.plot(h_low, below_dataRange, 'r--', label='Linear Model')
plt.xlabel('Height (m)')
plt.ylabel('Time (s)')
plt.title('Model Comparison (h < 500m)')
plt.legend()


# High heights
plt.subplot(122)
plt.plot(h_high, t_high, 'b-', label='Analytical')
plt.plot(h_high, above_dataRange, 'r--', label='Linear Model')
plt.xlabel('Height (m)')
plt.ylabel('Time (s)')
plt.title('Model Comparison (h > 1000m)')
plt.legend()

plt.show()

# Calculate error metrics
mse_low = mean_squared_error(t_low, below_dataRange)
mse_high = mean_squared_error(t_high, above_dataRange)
print(f"MSE for h < 500m: {mse_low:.2f}")
print(f"MSE for h > 1000m: {mse_high:.2f}")


