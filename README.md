# Baseball Data Analysis

## [How Salaries Compare](https://github.com/collinol/Baseball-Analysis/salaries.py)
A quick look at how US household incomes and MLB Player Salaries have increased over the years. Measured as cumulative percent change since 1985.
![alttext](/home/collinol/practice/Baseball-Analysis/Salaries.png)
  
  Not a big surprise here. I think these results are a bit trivial and in the future
  it'd be more interesting to compare salaries of other professional sports
  against the viewership levels of each of those sports. I think a visualization of that data
  would give a good look into trends among US sports over the past few decades.
## [How Batting Performance Affects Salary](https://github.com/collinol/Baseball-Analysis/performance_vs_salary.py)
Here we take a look at four metrics of batters to see which has the most impact on salaries. 
The salary data is taken after two year's worth of the respective performance data. 
This gives us a look into how you can anticipate a player's salary to change in future years, given his current performance.

![alttext](/home/collinol/practice/Baseball-Analysis/Batting_Performance.png)
  
 While home runs are exciting and paying top dollar for a big hitter could be help drive a team's ticket sales,
 it makes sense that the correlation to an increased salary is slightly higher for RBI.
 Teams pay player's to help them win, and which performance metric is most vital to winning?
  Well, I'll let this quote from *MoneyBall* answer that.
 > Billy Beane: You get on base, we win. You don't, we lose. And I *hate* losing, Chavy. I *hate* it. I hate losing more than I even wanna win.



## [How Pitching Performance Affects Salary](https://github.com/collinol/Baseball-Analysis/performance_vs_salary.py)
In the same script, I took a similar look at how past pitching performance affected a player's current salary   

![alttext](/home/collinol/practice/Baseball-Analysis/Pitching_Performance.png)

Naturally, it makes sense that wins would be the highest indicator of an increased salary and 
a higher ERA would earn lower salaries,
but I found it interesting that innings pitched and strikeouts have similar correlations.
I guess neither of those metrics on their own tell you anything about how good a pitcher is. 
A lot of innings pitched doesn't tell you how good the pitcher was; 
and likewise, if a player threw 100 strikeouts, without knowing how many total pitches they threw, 
that information doesn't tell us much about the quality of the player.

## Scikit Machine Learning
Based on the previous results, I want to see how well I can predict that, 
given a player's performance over the past two years, whether or not 
they'll have an above average salary the following year.
My initial reservations about this are that; 
1. I don't have a lot of data by ML standards
2. The scatter plots do not have particularly strong correlations, so I don't 
anticipate being able to achieve a remarkably high accuracy.

I'll stick with just pitchers for now and look at the same four metrics as before,
ERA, strikouts, wins and innings pitched.

First, getting the data cleaned up is the same process as it was in the previous analysis. 
We want only players with data for all 5 years, and then to average out their performances for
the two years prior to the salary that we're looking at.
```angular2html

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
print testing_set.head()

```
Gives us 
```
    playerID  yearID    ERA    W    SO  IPouts     G     salary
0   abadfe01  2012.5  4.220  0.0  35.0   125.5  38.0   525900.0
1  adamsmi03  2012.5  3.615  3.0  34.0   116.0  44.5  7000000.0
2  affelje01  2012.5  3.220  1.0  39.0   145.5  53.0  6000000.0
3  albural01  2012.5  2.635  2.0  44.0    93.5  30.5   837500.0
4  allenco01  2012.5  3.075  3.0  57.5   149.0  52.0   515400.0
```
Obviously, there's no year 2012.5, and that's the average of 2012 and 2013, the two previous years that we're looking at.
I'm not using that as an input for classification though, so I'm not worried about it for now.
There are 300 entries in this dataframe. Not a lot, as I mentioned, but with a 3:1 
train:test split, it should be enough to give us something to work with.

Based on [this flow chart](http://scikit-learn.org/stable/tutorial/machine_learning_map/),
 I'll use scikit's Naive Bayes for classification.
 
 ```angular2html
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


output: Accuracy: 74.67
```
After one run, we have almost 75% accuracy, which is not great, 
but better certainly better than random chance, and a little better than
what I had initially expected. Next is to run this classifier a few thousand times
and see what our average accuracy is.

```angular2html
runs = 1000
accuracies = []
for i in range(runs):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    gnb.fit(X_train, y_train)
    accuracies.append(gnb.score(X_test, y_test))
    
print 'over 1000 training runs: \nmin accuracy: {:0.2f}% \navg accuracy: {:0.2f}% \nmax accuracy: {:0.2f}%'\
    .format(min(accuracies)*100, 100 * np.mean(accuracies), max(accuracies)*100)


output: 
min accuracy: 65.33% 
avg accuracy: 77.88% 
max accuracy: 90.67%
```

### Additional improvements to the model.
Of course, it should be stated that just because there is correlation 
between stats and salary, we cannot say for certain that one implies the
other, without having examined a lot of other data.
The other important thing to note is the large range in accuracies obtained from our model.
Theres an almost 30 point swing between the lowest accuracy and highest. I predict
that this is mainly caused by an insufficient amount of data. A larger training set
would almost certainly be a key factor in reducing this variance.  
Additionally, for future work, I'd like to look at post-season performance metrics, 
and whether or not those stats for a player weigh more heavily in 
future salaries. 