## Name : Kishore S
## Reg No : 212222240050
# Ex.No: 05  IMPLEMENTATION OF TIME SERIES ANALYSIS AND DECOMPOSITION
### Date: 


## AIM:
To Illustrates how to perform time series analysis and decomposition on the monthly average temperature of a city/country and for airline passengers.

## ALGORITHM:
1. Import the required packages like pandas and numpy
2. Read the data using the pandas
3. Perform the decomposition process for the required data.
4. Plot the data according to need, either seasonal_decomposition or trend plot.
5. Display the overall results.

## PROGRAM:
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# Load the dataset
df = pd.read_csv('NFLX.csv')  # Assuming the file name is 'NFLX.csv'

# Output FIRST FIVE ROWS
print("FIRST FIVE ROWS:")
print(df.head())

# Convert 'Date' column to datetime and set it as index
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Extract the 'Open' column to perform time series analysis
stock_data = df['Open']

# Create a new index if necessary (adjust frequency as per data characteristics)
new_index = pd.date_range(start=stock_data.index.min(), periods=len(stock_data), freq='B')[:len(stock_data)]
stock_data.index = new_index

# Perform decomposition (additive model)
decomposition = seasonal_decompose(stock_data, model='additive', period=30)  # Adjust period for monthly seasonality

# Extract trend, seasonal, and residuals components
trend = decomposition.trend
seasonal = decomposition.seasonal
residuals = decomposition.resid

# PLOTTING THE DATA
print("\nPLOTTING THE DATA:")
plt.figure(figsize=(12, 6))
plt.plot(stock_data, label='Original Stock Prices')
plt.title('Original Stock Prices')
plt.legend()
plt.show()

# SEASONAL PLOT REPRESENTATION
print("\nSEASONAL PLOT REPRESENTATION:")
plt.figure(figsize=(12, 6))
plt.plot(seasonal, label='Seasonal Component', color='orange')
plt.title('Seasonal Component')
plt.legend()
plt.show()

# TREND PLOT REPRESENTATION
print("\nTREND PLOT REPRESENTATION:")
plt.figure(figsize=(12, 6))
plt.plot(trend, label='Trend Component', color='green')
plt.title('Trend Component')
plt.legend()
plt.show()

# OVERALL REPRESENTATION
print("\nOVERALL REPRESENTATION:")

plt.subplot(3, 1, 3)
plt.plot(residuals, label='Residuals', color='red')
plt.title('Residuals Component')
plt.legend(loc='best')

plt.tight_layout()
plt.show()

```

## OUTPUT:
### FIRST FIVE ROWS:
![image](https://github.com/user-attachments/assets/a7a9bad3-b94a-49d7-9aa9-40b5e0934811)


### PLOTTING THE DATA:
![image](https://github.com/user-attachments/assets/32527eb0-ad0c-48f8-aac3-c943d6eb9cdf)


### SEASONAL PLOT REPRESENTATION :
![image](https://github.com/user-attachments/assets/0faddda7-801d-4b91-a225-954fdce6dab8)


### TREND PLOT REPRESENTATION :
![image](https://github.com/user-attachments/assets/14b0a79e-33f0-43ef-a1cd-3f937f9b22db)


### RESIDUAL PLOT  REPRESENTATION:
![image](https://github.com/user-attachments/assets/575a0a15-46e2-4b9f-852a-42c8b78c481b)


### RESULT:
The python code for the time series analysis and decomposition was executed successfully.
