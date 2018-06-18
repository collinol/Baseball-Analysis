import pandas as pd
import numpy as np
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


root = 'baseballdatabank-2017.1/core/'
all_pitching = pd.read_csv(root + 'Pitching.csv')
all_salaries = pd.read_csv(root + 'Salaries.csv')

salaries_2014 = all_salaries[all_salaries['yearID'] == 2014]
salaries_2014 = salaries_2014.drop(['yearID', 'teamID', 'lgID'], axis=1)

testing_set = all_pitching[['yearID', 'playerID', 'ERA', 'W', 'SO', 'IPouts', 'G']]
testing_set = testing_set[(testing_set['yearID'] == 2012) | (testing_set['yearID'] == 2013)]
testing_set = testing_set.merge(salaries_2014, on='playerID')
testing_set = testing_set.groupby(['playerID', 'yearID'], as_index=False).mean()
years_to_examine = [2012, 2013]

# Same function used to find players with records in all five years used in the analysis section.
def players_with_all_five_years(records):
    # First create a list with all of the playerIDs (many will be repeated) in the playerID column of the DataFrame
    Id_list = list(records['playerID'])
    players_with_five_years = set()

    # Again, want to filter out players with only data in each of the five years.
    for player in Id_list:
        if (Id_list.count(player)) == len(years_to_examine):
            players_with_five_years.add(player)

    return records[records['playerID'].isin(players_with_five_years)]


testing_set = players_with_all_five_years(testing_set)
# Average the player statistics across the two years
testing_set = testing_set.groupby('playerID', as_index=False).mean()



testing_set['salary_sd'] = (testing_set['salary'] - testing_set['salary'].mean()) / testing_set['salary'].std()
testing_set['labels'] = testing_set['salary_sd'] > 0
testing_set['labels'] = testing_set['labels'].astype(int)
testing_set.drop(['yearID', 'playerID', 'G', 'salary'], axis=1, inplace=True)
print testing_set.head


testing_set.drop('salary_sd', axis=1, inplace=True)
# Name the features X and the labels y
X = np.array(testing_set.drop('labels', axis=1))
y = np.array(testing_set['labels'])
# Preprocessing
X = scale(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# Test the classifier on the testing set of data
accuracy = gnb.score(X_test, y_test)
print("Accuracy: {:.2f}".format(accuracy*100))


# Calculate the average accuracy over a thousand classification runs
runs = 1000
accuracies = []
for i in range(runs):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    gnb.fit(X_train, y_train)
    accuracies.append(gnb.score(X_test, y_test))


print 'over 1000 training runs: \nmin accuracy: {:0.2f}% \navg accuracy: {:0.2f}% \nmax accuracy: {:0.2f}%'\
    .format(min(accuracies)*100, 100 * np.mean(accuracies), max(accuracies)*100)
