import pickle
import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier


random.seed(1)


# save() function to save the trained network to a file
def save(ann, file_name):
    with open(file_name, 'wb') as fp:
        pickle.dump(ann, fp)


# restore() function to restore the file
def load(file_name):
    with open(file_name, 'rb') as fp:
        nn = pickle.load(fp)
    return nn


def get_data(with_birad=True):
    data = pd.read_csv('mammo_filtered.csv', header=0, dtype=float)

    # split out label
    y = data.Severity
    data = data.drop(columns=['Severity'])

    # removing the professional opinion for a more pure dataset
    if not with_birad:
        # TODO test without the "professional" recommendation
        data = data.drop(columns=['BIRADS'])

    x_train, x_test, y_train, y_test = train_test_split(data, y, test_size=0.2, random_state=1)
    return x_train, x_test, y_train, y_test


def display_data():
    # how are the features distributed?
    data = get_data(with_birad=True)
    sns.countplot(x='BIRADS', data=data)
    plt.title('BIRADS - Professional Assessment Score:1=benign,5=malignant')
    plt.show()

    sns.countplot(x='Age', data=data)
    plt.title('Patients Age')
    plt.show()

    sns.countplot(x='Shape', data=data)
    plt.title('Tumor Shape: Round=1 Oval=2 Lobular=3 Irregular=4')
    plt.show()

    sns.countplot(x='Margin', data=data)
    plt.title('Tumor Margin: Solid=1 Veiny=2 Obscured=3 Ill-defined=4 Spikey=5')
    plt.show()

    sns.countplot(x='Density', data=data)
    plt.title('Density (Lobule/Gland density): Extreme=1 Iso=2 Scattered=3 Fatty=4')
    plt.show()

    sns.countplot(x='Severity', data=data)
    plt.title('Severity: 1=Malignant, 0=Benign')
    plt.show()


# reusable method to do fit, evaluate, and cross over validation
def train_mod(mod, x, y, filename, do_save=True):
    score = cross_val_score(mod, x, y, cv=200)
    mod.fit(x, y)
    print(sum(score) / len(score))
    if do_save:
        save(mod, filename)


def train(with_birad):
    print("Beginning Training: ")
    x_train, x_test, y_train, y_test = get_data(with_birad)

    # train on several different models
    logreg_m = LogisticRegression(C=1e5, solver='lbfgs', multi_class='warn')  # 82.125
    svm_m = svm.SVC(C=0.5, kernel='linear')  # 82.83
    tree_m = DecisionTreeClassifier(criterion='entropy', max_depth=4, min_samples_leaf=3)  # 84.20
    forest_m = RandomForestClassifier(n_estimators=200, criterion='entropy', max_depth=4, min_samples_leaf=3)  # 84.08

    # this is an ensemble
    vote_m = VotingClassifier(estimators=[
        ('lr', logreg_m),
        ('svm', svm_m),
        ('tree', tree_m),
        ('forest', forest_m)
    ])

    train_mod(logreg_m, x_train, y_train, 'logreg.pck')
    train_mod(svm_m, x_train, y_train, 'svm.pck')
    train_mod(tree_m, x_train, y_train, 'tree.pck')
    train_mod(forest_m, x_train, y_train, 'forest.pck')
    train_mod(vote_m, x_train, y_train, 'vote.pck')
    print("Training Completed:")


# secondary method for model validation
def validate_mod(pck_name, x, y):
    mod = load(pck_name)
    preds = mod.predict(x)
    print(sum(preds == y.values) / len(y))
    plt.scatter(x.values, preds.reshape(-1,1))
    plt.show()


# validate each model with validation data
def validate(with_birad):
    print("Beginning Validation: ")
    x_train, x_test, y_train, y_test = get_data(with_birad)
    validate_mod('logreg.pck', x_test, y_test)
    validate_mod('svm.pck', x_test, y_test)
    validate_mod('tree.pck', x_test, y_test)
    validate_mod('forest.pck', x_test, y_test)
    validate_mod('vote.pck', x_test, y_test)
    print("Validation Completed:")


use_birad = True
# NOTE: only train or validate, not at the same time
# display_data()
# train(use_birad)
validate(use_birad)
