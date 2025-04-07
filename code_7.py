import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
#If any of this libraries is missing from your computer. Please install them using pip.

filename = 'Flight_Delays_2018.csv'

#ARR_DELAY is the column name that should be used as dependent variable (Y).

df = pd.read_csv(filename)

df['FL_DATE'] = pd.to_datetime(df['FL_DATE'], format = "%m/%d/%y") # Converts 'FL_DATE' column to date time
df['FL_DATE']= df.FL_DATE.dt.month # Returns 'FL_DATE' column into months

# Part 1: Descriptive Analytics

print(df['ARR_DELAY'].describe()) # Prints out descriptive analytics on Arrival Delay's column

# Function that returns the descriptive analytics between an independent variable, 'column' and the dependent variable 'ARR_DELAY'
# Shows if there might be some significance between variables 
def get_describe_column(column):

    column = column.upper()
    return df.groupby(column)['ARR_DELAY'].describe()

# Prints out descriptive analytics between the independent variables and the depentent variable 'ARR_DELAY'
print(get_describe_column('fl_date'))
print(get_describe_column('op_carrier_name'))
print(get_describe_column('taxi_in'))
print(get_describe_column('taxi_out'))
print(get_describe_column('carrier_delay'))
print(get_describe_column('weather_delay'))
print(get_describe_column('late_aircraft_delay'))

#Vizualization based on insights from descriptive analytics
df.plot.scatter(x= 'TAXI_OUT', y='ARR_DELAY')
df.plot.scatter(x= 'CARRIER_DELAY', y='ARR_DELAY')
df.plot.scatter(x= 'WEATHER_DELAY', y='ARR_DELAY')
df.plot.scatter(x= 'LATE_AIRCRAFT_DELAY', y='ARR_DELAY')
df.boxplot(column='ARR_DELAY', by= 'OP_CARRIER_NAME')
plt.xticks(rotation = 90)
df.boxplot(column='ARR_DELAY', by= 'FL_DATE')

# Step 2: Predictive analytics 

# Define independent and dependent variables
y = df['ARR_DELAY']
x = df['CARRIER_DELAY']

x= sm.add_constant(x) #adds constant to predictor variable

# fit linear regression model
model = sm.OLS(y,x).fit()

# view model summary
print(model.summary())
print(model.params)

#Visualizing OLS model
fig, ax = plt.subplots()
fig = sm.graphics.plot_fit(model, 0, ax=ax)

# Plotting scatterplot with matplotlib
plt.scatter(df['CARRIER_DELAY'], df['ARR_DELAY'])
plt.show()


### Vizualization of OLS Model, Statology method###

# #find line of best fit
# a, b = np.polyfit(df['CARRIER_DELAY'], df['ARR_DELAY'], 1)

# #add points to plot
# plt.scatter(df['CARRIER_DELAY'], df['ARR_DELAY'], color = 'purple')

# #add line of best fit to plot
# plt.plot(df['CARRIER_DELAY'], a*df['CARRIER_DELAY'] + b)

# # add the fitted regression equation to the scatter plot
# plt.text(0, 2500, 'y = ' + '{:.3f}'.format(b) + ' + {:.3f}'.format(a) + 'x', size = 12)

# # add axis labels
# plt.xlabel('Carrier Delay')
# plt.ylabel('Arrival Delay')