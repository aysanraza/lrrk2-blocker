# imports
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# function designed to train MLP model
def train_pipe(f, l):
    features = f
    labels = l
    pipe = make_pipeline(StandardScaler(), MLPClassifier(learning_rate_init = 0.001, learning_rate='adaptive', activation='tanh', hidden_layer_sizes=(10,3), alpha=0.001, solver='adam', max_iter=2000))
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=0)
    pipe.fit(X_train,y_train)
    return pipe, X_test, y_test


if __name__ == '__main__':
    print("training")
