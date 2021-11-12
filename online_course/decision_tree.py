from itertools import combinations

import graphviz
import pandas as pd
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier

from online_course.utils import plot_decision_regions

data = pd.read_csv('data/titanic.csv', index_col='PassengerId')

input_fields = ['Pclass', 'Age', 'Sex', 'Embarked']
expected_result_field = ['Survived']

influential_fields = input_fields + expected_result_field

prepared_dataset = data.filter(influential_fields)
prepared_dataset = prepared_dataset.dropna()
prepared_dataset.loc[prepared_dataset['Embarked'] == 'S', 'Embarked'] = 0
prepared_dataset.loc[prepared_dataset['Embarked'] == 'Q', 'Embarked'] = 1
prepared_dataset.loc[prepared_dataset['Embarked'] == 'C', 'Embarked'] = 2

prepared_dataset['Sex'] = (prepared_dataset['Sex'] == 'female').astype('int')

X = prepared_dataset.filter(input_fields)
y = prepared_dataset.filter(expected_result_field)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, shuffle=True)

tree_param = {'criterion': ['gini', 'entropy'],
              'max_depth': [4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 20, 30, 40, 50, 70, 90, 120, 150]}
dt = GridSearchCV(DecisionTreeClassifier(), tree_param, )

dt.fit(X_train, y_train)
dot_data = tree.export_graphviz(dt.best_estimator_, out_file=None, filled=True, rounded=True,
                                feature_names=input_fields,
                                class_names=['Survived', 'Died'])
graph = graphviz.Source(dot_data)
graph.render("Titanic")

importance = dt.best_estimator_.feature_importances_
# summarize feature importance
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))

plot_decision_regions(X_test, y_test, classifier=dt)

print("decision tree accuracy:", accuracy_score(y_test, dt.predict(X_test)))
