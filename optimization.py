# imports
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier

# function designed to conduct the optimization
def optimization_grid_search(f,l):
    print("\nfinding optimization possibilities, please wait ...\n")
    features = f
    labels = l
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=0)

    mlp_gs = MLPClassifier(max_iter=1000)

    # Set the parameters by cross-validation
    parameter_space = {
        'hidden_layer_sizes': [(10,30,10),(20,)],
        'activation': ['tanh', 'relu'],
        'solver': ['sgd', 'adam'],
        'alpha': [0.0001, 0.05],
        'learning_rate': ['constant','adaptive'],
    }

    # Objective metrics
    scores = ['precision']
    clf = GridSearchCV(mlp_gs, parameter_space, cv=5, n_jobs=-1)
    clf.fit(X_train, y_train)

    print("Grid scores are:")
    means = clf.cv_results_['mean_test_score']
    for mean,params in zip(means, clf.cv_results_['params']):
      print("%0.3f for %r" % (mean, params))

    print("\nBest Hyperparameters found is:")
    print(clf.best_params_)


if __name__ == '__main__':
    print("grid search")
