import pandas as pd
import numpy as np

# Data viz:
import matplotlib.pyplot as plt
import seaborn as sns

import prepare as prep
import pandas as pd
# random forest imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# knn imports
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix


def create_x_y(train, validate, test, target, drop_col=[]):
    """
    This function creates x and y variables for either a decision tree or a random forest, 
    by using the unsplit df, target variable columns name and column to drop, for multiple columns that need to be 
    dropped create a list of the columns0
    The arguments taken in are train, validate, test, target, drop_col=[])
    The function returns x_train, y_train, x_validate, y_validate, x_test, y_test
    """
    # separates train target variable
    x_train = train.drop(columns=drop_col)
    y_train = train[target]
    # validate 
    x_validate = validate.drop(columns=drop_col)
    y_validate = validate[target]

    # test
    x_test = test.drop(columns=drop_col)
    y_test = test[target]
    return x_train, y_train, x_validate, y_validate, x_test, y_test



def rf_max_depth_function(x_train, y_train, x_validate, y_validate, max_range=11):
    """
    This function takes in four arguments (x_train, y_train, x_validate, y_validate)
    an optional argument is max_range is the highest for the range method
    """
    for x in range(1, max_range):
        print(x)
        tree = RandomForestClassifier(max_depth=x, random_state=123)

        tree.fit(x_train, y_train)
        acc = tree.score(x_train, y_train)
        print(acc)
        #validate
        val_acc = tree.score(x_validate, y_validate)
        print(f"max depth- {x} accuracy- {round(acc, 2)} validation accuracy {round(val_acc, 2)}")
      

def rf_min_leaf_function(x_train, y_train, x_validate, y_validate, set_depth=5, max_range=11):
    """
    This function takes in five arguments (x_train, y_train, x_validate, y_validate)
    max_range is the highest for the range method
    """
    for x in range(1,max_range):
        tree = RandomForestClassifier(max_depth=set_depth,min_samples_leaf=x)
        tree.fit(x_train, y_train)
        acc = tree.score(x_train, y_train)
        #validate
        val_acc = tree.score(x_validate, y_validate)
        print(f"min samples leaf- {x} accuracy- {round(acc, 2)} validation accuracy {round(val_acc, 2)}")


def rf_tree_predict(clf, x_train, y_train):
    """
    This function takes in 3 arguments:
    clf = RandomForestClassifier() with your parameters included and fitted (REQUIRED)
    The function also prints the 
    This function returns four positions (y_pred, conf, report, pretty)
    """
    clf.predict(x_train)
    y_pred = clf.predict(x_train)
    conf = confusion_matrix(y_train, y_pred)
    labels = sorted(y_train.unique())
    pretty = pd.DataFrame(conf,
            index=[f"{label} _actual" for label in labels],
            columns=[f"{label} _predict" for label in labels])
    print(classification_report(y_train, y_pred))
    return y_pred, conf, pretty 

def knn_model_accuracies(x_train, y_train, x_validate, y_validate):
    '''
    This function is an aid for KNN models.
    Takes arguments x_train, y_train, x_validate, y_validate, which iterates through 10 n_neighbors and fits 
    and then produces a dataframe that shows the train score, validation score and their differences
    '''
    model_accuracies = {}

    for i in range(1,11):
        #MAKE THE THING
        knn = KNeighborsClassifier(n_neighbors=i)

        #FIT THE THING
        knn.fit(x_train, y_train)

        #USE THE THING
        model_accuracies[f'{i}_neighbors'] = {
            'train_score': round(knn.score(x_train, y_train),2),
            'validate_score':round(knn.score(x_validate, y_validate),2),
            'diff_score': round(round(knn.score(x_train, y_train),2) - round(knn.score(x_validate, y_validate),2), 2)
        }
    return pd.DataFrame(model_accuracies).T

def knn_viz_20(x_train, y_train, x_validate, y_validate):
    '''
    This function helps visualize your knn results by fitting the given arguments (x_train, y_train, x_validate, y_validate)
    and using these features with the KNeighborsClassifier to calculate through an iteration of 20 and find a train score and validation score and producing the vsual plot
    '''
    metrics = []

    for k in range(1,21):

        # MAKE the thing
        knn = KNeighborsClassifier(n_neighbors=k)

        # FIT the thing (remember only fit on training data)
        knn.fit(x_train, y_train)

        # USE the thing (calculate accuracy)
        train_score = knn.score(x_train, y_train)
        validate_score = knn.score(x_validate, y_validate)

        output = {
            "k": k,
            "train_score": train_score,
            "validate_score": validate_score
        }

        metrics.append(output)

    #conver to df
    results = pd.DataFrame(metrics)

    # plot the data
    results.set_index('k').plot(figsize = (16,9))
    plt.ylabel('Accuracy')
    plt.xticks(np.arange(0,21,1))
    plt.grid()