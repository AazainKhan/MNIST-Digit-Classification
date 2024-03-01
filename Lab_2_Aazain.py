
# # Lab assignment #2: “Classification”
# Aazain Ullah Khan - 301277063 - COMP 247
# <hr>


# ### Load and check the data


# from sklearn import minst dataset using fetch_openml

from sklearn.datasets import fetch_openml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


# load mnist dataset into pandas dataframe
np.random.seed(42)
mnist = fetch_openml('mnist_784', version=1)


# list the keys of the dataset
mnist.keys()


# assign data to ndarray
X_aazain = mnist['data']

# assign target to ndarray
y_aazain = mnist['target']
print(y_aazain.head(20))


# types
print(type(X_aazain))
print(type(y_aazain))


# shape
print(X_aazain.shape)
print(y_aazain.shape)


# assign target to ndarray
some_digit1 = X_aazain.iloc[7].to_numpy()
some_digit2 = X_aazain.iloc[5].to_numpy()
some_digit3 = X_aazain.iloc[0].to_numpy()

print(some_digit1)


# use imshow method to display the image of the digit and plot it

some_digit_image1 = some_digit1.reshape(28, 28)
some_digit_image2 = some_digit2.reshape(28, 28)
some_digit_image3 = some_digit3.reshape(28, 28)

plt.imshow(some_digit_image1, cmap=mpl.cm.binary)
plt.axis("off")
plt.show()

plt.imshow(some_digit_image2, cmap=mpl.cm.binary)
plt.axis("off")
plt.show()

plt.imshow(some_digit_image3, cmap=mpl.cm.binary)
plt.axis("off")
plt.show()


# ### Pre-process the data


# change type of y_aazain to unit8

y_aazain = y_aazain.astype(np.uint8)



# transform the target
y_aazain = np.where(y_aazain <= 3, 0, np.where(y_aazain <= 6, 1, 9))


# print the frequencies of each of the 3 target classes after transformation
class_counts = pd.Series(y_aazain).value_counts()

for label, count in class_counts.items():
    print(f"Class {label}: {count} instances")


# return count is the frequency of each class
unique, counts = np.unique(y_aazain, return_counts=True)
plt.bar(unique, counts)
plt.title('Frequency of Each Class')
plt.xlabel('Class')
plt.ylabel('Frequency')
plt.xticks(unique, ['0-3', '4-6', '7-9'])
plt.show()


# Split the data into training and test sets
# Assign the first 50,000 records for training and the last 20,000
# records for testing

X_train, X_test, y_train, y_test = X_aazain[:50000], X_aazain[50000:
                                                              70000], y_aazain[:50000], y_aazain[50000:70000]


# ### Build Classification Models


# **Naive Bayes**


from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB


# create a MultinomialNB classifier fitting the training data
NB_clf_aazain = MultinomialNB()
NB_clf_aazain.fit(X_train, y_train)


# cross validation score

cross_val_score(NB_clf_aazain, X_train, y_train, cv=3, scoring="accuracy")


# accuracy score against test data

y_pred = NB_clf_aazain.predict(X_test)
accuracy_score(y_test, y_pred)


# confusion matrix for the accuracy of the model

confusion_matrix(y_test, y_pred)


# predict the 3 variables

print(NB_clf_aazain.predict([some_digit1, some_digit2, some_digit3]))


NB_clf_aazain.classes_


# **Logistic Regression**


# initialise the logistic regression model
from sklearn.linear_model import LogisticRegression

LR_clf_aazain = LogisticRegression()


# fit the model with training data
# uses lbfgs
LR_clf_aazain = LogisticRegression(
    solver='lbfgs', max_iter=1200, tol=0.1, multi_class='multinomial')
LR_clf_aazain.fit(X_train, y_train)


# fit model with training data
# uses saga
LR_clf_aazain = LogisticRegression(
    solver='saga', max_iter=1200, tol=0.1, multi_class='multinomial')
LR_clf_aazain.fit(X_train, y_train)


# cross validation score

cross_val_score(LR_clf_aazain, X_train, y_train, cv=3, scoring="accuracy")


# accuracy score against test data

y_pred = LR_clf_aazain.predict(X_test)
accuracy_score(y_test, y_pred)


# confusion matrix for the accuracy of the model
# saga

confusion_matrix(y_test, y_pred)


LR_clf_aazain.predict([some_digit1, some_digit2, some_digit3])


