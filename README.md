# Standardizing data

Some models, like K-nearest neighbors (KNN) & neural networks, work better with scaled data -- so we'll standardize our data.

We'll also remove unimportant variables (day of week), according to feature importances, by indexing the features DataFrames with .iloc[]. KNN uses distances to find similar points for predictions, so big features outweigh small ones. Scaling data fixes that.

sklearn's scale() will standardize data, which sets the mean to 0 and standard deviation to 1. Ideally we'd want to use StandardScaler with fit_transform() on the training data, and fit() on the test data, but we are limited to 15 lines of code here.

Once we've scaled the data, we'll check that it worked by plotting histograms of the data.

Instructions

1.    Remove day of week features from train/test features using .iloc (day of week are the last 4 features).
2.    Standardize train_features and test_features using sklearn's scale(); store scaled features as scaled_train_features and scaled_test_features.
3.    Plot a histogram of the 14-day RSI moving average (indexed at [:, 2]) from unscaled train_features on the first subplot (ax[0]]).
4.    Plot a histogram of the standardized 14-day RSI moving average on the second subplot (ax[1]).

```
from sklearn.preprocessing import scale

# Remove unimportant features (weekdays)
train_features = train_features.iloc[:, :-4]
test_features = test_features.iloc[:, :-4]

# Standardize the train and test features
scaled_train_features = scale(train_features)
scaled_test_features = scale(test_features)

# Plot histograms of the 14-day SMA RSI before and after scaling
f, ax = plt.subplots(nrows=2, ncols=1)
train_features.iloc[:, 2].hist(ax=ax[0])
ax[1].hist(scaled_train_features[:, 2])
plt.show()
```

