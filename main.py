# imports
import data
import optimization
import training
import evaluation
from joblib import dump


if __name__ == '__main__':
    features = data.features()
    labels = data.labels()
    optimization.optimization_grid_search(features, labels)
    print('Press 1 to proceed:')
    optimization_check = input()
    if optimization_check == "1":
        model, test_features, test_labels = training.train_pipe(features, labels)
        print('\nWould you like to find "accuracy" based evaluation: Press 1 to proceed or 0 to pass:')
        accu = input()
        if accu =="1":
            evaluation.accuracy(model, test_features, test_labels)
        else:
            pass
        print('\nWould you like to find "multiclass_confusion_matrix" '
              'based evaluation: Press 1 to proceed or 0 to pass:')
        mcm = input()
        if mcm == "1":
            evaluation.multiclass_confusion_matrix(model, test_features, test_labels)
        else:
            pass
        print('\nWould you like to save the model: Press 1 to proceed or 0 to pass:')
        sm = input()
        if sm == "1":
            dump(model, 'lrrk2_model.joblib')
            print("model saved in your current directory\n")
        else:
            pass
        # saving model
    else:
        print("Process did not proceed, run again")
