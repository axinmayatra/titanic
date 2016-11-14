import random
import pandas
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split

train = pandas.read_csv('train2.csv')
train = train.dropna()
y, X = train['Survived'], train[['Age','SibSp','Fare','Sex','Embarked','Parch','Pclass']].fillna(0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)

# clf = LogisticRegression()
# clf = GaussianNB()
# clf = DecisionTreeClassifier()
# clf = KNeighborsClassifier(n_neighbors=5)
# clf = MLPClassifier(solver='lbfgs', alpha=10,hidden_layer_sizes=(5,5,2), random_state=1,learning_rate_init=.01)
clf.fit(X_train, y_train)                         
print(accuracy_score(clf.predict(X_test), y_test))
# print(X_test[0:10],clf.predict(X_test)[0:10])
print(X[0:20])