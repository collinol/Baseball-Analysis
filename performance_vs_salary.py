import numpy
import pandas as pd
import matplotlib.pyplot as plt

root = 'baseballdatabank-2017.1/core/'
all_batting = pd.read_csv(root + 'Batting.csv')
all_pitching = pd.read_csv(root + 'Pitching.csv')
all_salaries = pd.read_csv(root + 'Salaries.csv')

past_five_years = [2012, 2013, 2014, 2015, 2016]
batting = all_batting[['playerID', 'yearID', 'RBI', 'H', 'HR', 'G']]
batting = batting[batting['yearID'].isin(past_five_years)]

pitching = all_pitching[['playerID', 'yearID', 'ERA', 'W', 'SO', 'IPouts']]
pitching = pitching[pitching['yearID'].isin(past_five_years)]

batting = batting.groupby(['playerID', 'yearID'], as_index=False).sum()
pitching = pitching.groupby(['playerID', 'yearID'], as_index=False).sum()

# Only look at pitchers with more than 120 innings pitched, and players with more than 100 games played ( for batters ).
batting = batting[batting['G'] > 100]
pitching = pitching[(pitching['IPouts'] / 3) > 120]


# Remove all players from the dataframes who don't have data listed for all five past years.

def players_with_all_five_years(records):
    # First create a list with all of the playerIDs
    Id_list = list(records['playerID'])
    players_with_five_years = set()
    # there are no double entries for the same year, so we can be sure that if there are five entries
    # it corresponds to each of the past five years
    for player in Id_list:
        if (Id_list.count(player)) == len(past_five_years):
            players_with_five_years.add(player)

    return records[records['playerID'].isin(players_with_five_years)]


# Create the new DataFrames including only players with records in years in the analysis
batting = players_with_all_five_years(batting)
pitching = players_with_all_five_years(pitching)

print batting.head()
print '--'
print pitching.head()

salaries_2014 = all_salaries[all_salaries['yearID'] == 2014]
salaries_2014 = salaries_2014.drop(['yearID', 'teamID', 'lgID'], axis=1)

# Merge the salary and performance  DataFrames
batting = batting.merge(salaries_2014, on='playerID')
batting = batting.rename(columns={'H': 'Hits', 'HR': 'Home Runs', 'G': 'Games', 'salary': 'Salary'})
pitching = pitching.merge(salaries_2014, on='playerID')
pitching = pitching.rename(columns={'W': 'Wins', 'SO': 'Strikeouts', 'salary': 'Salary'})

print batting.head()
print '--'
print pitching.head()

previous_years = [2012, 2013]
following_years = [2015, 2016]


# Return averages for data, grouped by the previous and following year brackets that we're interested in
def create_seasons_averages(records):
    five_year_average = records.groupby('playerID', as_index=False).mean()

    previous_two_years = records[records['yearID'].isin(previous_years)]
    previous_two_years_average = previous_two_years.groupby('playerID', as_index=False).mean()

    following_two_years = records[records['yearID'].isin(following_years)]
    following_two_years_average = following_two_years.groupby('playerID', as_index=False).mean()

    return five_year_average, previous_two_years_average, following_two_years_average


# Create the average batting and pitching DataFrames
batting_five_year, batting_previous, batting_following = create_seasons_averages(batting)
pitching_five_year, pitching_previous, pitching_following = create_seasons_averages(pitching)


print('There are', len(batting_previous), 'batters in the wrangled batting datasets.')
print('There are', len(pitching_previous), 'pitchers in the wrangled pitching datasets.')


def analyze_previous_records(record_df, statistics_list):
    for n,statistic in enumerate(statistics_list):
        n += 1
        x = record_df[statistic]
        y = record_df['Salary'] / 1e6
        plt.subplot(2, 2, n)
        plt.scatter(x, y)
        plt.title('Previous Two Seasons Average '+ statistic+' vs Salary')
        plt.ylabel('Salary, in millions of dollars')
        plt.xlabel(statistic)
        z = numpy.polyfit(x, y, 1)
        p = numpy.poly1d(z)
        plt.plot(x, p(x), 'orange')
        print('The correlation between average {} over the previous two years and salary is {:0.3f}' \
         .format(statistic, record_df.corr()['Salary'][statistic]))


analyze_previous_records(batting_previous, ['RBI', 'Hits', 'Home Runs', 'Games'])
plt.show()
analyze_previous_records(pitching_previous, ['ERA', 'Wins', 'Strikeouts', 'IPouts'])
plt.show()
