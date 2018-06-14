import pandas as pd
import matplotlib.pyplot as plt
import quandl

root = 'baseballdatabank-2017.1/core/'
all_salaries = pd.read_csv(root+'Salaries.csv')
all_batting = pd.read_csv(root+'Batting.csv')
all_pitching = pd.read_csv(root+'Pitching.csv')

yearly_average_salary = all_salaries.groupby('yearID').mean()

# Calculate and plot the percentage change from the first entry
yearly_salary_pct_change = 100.0 *(yearly_average_salary - yearly_average_salary.iloc[0]) / yearly_average_salary.iloc[0]

median_household_income = quandl.get('FRED/MEHOINUSA646N', start_date = '1985-01-01')

# Calculate and plot the percentage change of median household income from the first entry
median_household_pct_change = 100.0 * (median_household_income - median_household_income.iloc[0]) / \
                              median_household_income.iloc[0]

# collect data to plot on the same graph
years = [y for y in range(1985, 2017)]
yearly_salary_pct_changes = [x for x in yearly_salary_pct_change['salary']]
median_household_pct_changes = [x for x in median_household_pct_change['Value']]

#process US inflation rate data
inflation = open('US_Inflation.txt','r')
sum = 0
yearly_inflation_increase = []
for line in inflation:
    percent_increase = float(line.split('\t')[1])
    yearly_inflation_increase.append(sum+percent_increase)
    sum += percent_increase

plt.plot(years,yearly_salary_pct_changes, 'b', label='MLB Player Salary')
plt.plot(years,median_household_pct_changes, 'r', label='US Household Income')
plt.plot(years,yearly_inflation_increase, 'k', label='USD Inflation')

plt.title('Median Income')
plt.ylabel('Percent Change'), plt.xlabel('Year')
plt.legend()
plt.show()
