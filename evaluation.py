# imports
from sklearn.metrics import accuracy_score
from sklearn.metrics import multilabel_confusion_matrix

# function designed to perform accuracy based evaluation
def accuracy(p,f_test,l_test):
    # accuracy
    pipe = p
    X_test = f_test
    y_test = l_test
    print("Accuracy Score is: ", accuracy_score(pipe.predict(X_test), y_test))

# function designed to conduct multiclass confusion matrix based evaluation
def multiclass_confusion_matrix(p,f_test,l_test):
    # multiclass confusion matrix
    pipe = p
    X_test = f_test
    y_test = l_test
    y_pred = pipe.predict(X_test)
    mcm = multilabel_confusion_matrix(y_test, y_pred, labels=["Strong", "Week", "Non_Inhibitor"])
    print("multilabel_confusion_matrix is: ", mcm)


if __name__ == '__main__':
    print("evaluation")

