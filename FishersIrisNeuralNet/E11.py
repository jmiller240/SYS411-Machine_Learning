''' Daniel Gibson and Jack Miller
    E11
'''

import pandas
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense
from keras import regularizers
from keras import optimizers
from keras.layers import BatchNormalization


# load the dataset
path = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv'
df = pandas.read_csv(path, header=None)

# split into input and output columns
X, y = df.values[:, :-1], df.values[:, -1]

# ensure all data are floating point values
X = X.astype('float32')

# encode strings to integer
y = LabelEncoder().fit_transform(y)


# Create, fit, and evaluate neural network with the specified configurations
def runExperiment(num_hidden_nodes, lambda_val, alpha, num_epochs):
    print(f'{num_hidden_nodes = }  {lambda_val = }  {alpha = }  {num_epochs = }')

    # split into train, cross-val datasets
    X_train, X_cv, y_train, y_cv = train_test_split(X, y, test_size=0.33)

    # determine the number of input features
    n_features = X_train.shape[1]

    # Specify the lambda (L1) regularization parameter
    regularizer = regularizers.L1(lambda_val)

    # define model
    model = Sequential()
    model.add(BatchNormalization())
    model.add(Dense(num_hidden_nodes, activation='sigmoid', kernel_initializer='he_normal', kernel_regularizer = regularizer, input_shape=(n_features,)))
    model.add(Dense(3, activation='softmax'))

    # Specify the learning rate for the optimizing function
    opt = optimizers.SGD(learning_rate=alpha)

    # compile the model
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # fit the model
    model.fit(X_train, y_train, epochs=num_epochs, batch_size=32, verbose=0)

    # evaluate the model
    loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
    #print('Train Accuracy: %.3f' % acc)
    loss, cv_acc = model.evaluate(X_cv, y_cv, verbose=0)
    #print('Cross-val Accuracy: %.3f' % acc)

    return cv_acc


# Tune model configurations and report best results 
# Finds and stores the best performing value for each configuration option, uses each best value 
#   at the end to find overall best results
def main():
    # Tune number of hidden nodes
    accuracy = 0
    best_num_nodes = 0
    for i in range(1, 100, 2):
        results = runExperiment(i, 10, 0.01, 1)
        if results > accuracy:
            accuracy = results
            best_num_nodes = i

    # Tune lambda
    accuracy = 0
    best_lambda_val = 0
    for i in range(1, 11):
        val = (1 / (10 ** i))
        results = runExperiment(best_num_nodes, val, 0.01, 1)
        if results > accuracy:
            accuracy = results
            best_lambda_val = val

    # Tune learning rate alpha
    accuracy = 0
    best_alpha_val = 0
    for i in range(1, 11):
        val = (1 / (10 ** i))
        results = runExperiment(best_num_nodes, best_lambda_val, val, 1)
        if results > accuracy:
            accuracy = results
            best_alpha_val = val

    # Tune number of epochs
    accuracy = 0
    best_num_epochs = 0
    for i in range(10, 10000, 1000):
        results = runExperiment(best_num_nodes, best_lambda_val, best_alpha_val, i)
        if results > accuracy:
            accuracy = results
            best_num_epochs = i

    # Evaluate model with chosen parameters
    test_acc = runExperiment(best_num_nodes, best_lambda_val, best_alpha_val, best_num_epochs)

    print('Best Results:')
    print('Number of Hidden Nodes: %d' % best_num_nodes)
    print('Lambda (L1) Regularization Value: %.10f' % best_lambda_val)
    print('Alpha Learning Rate: %.10f' % best_alpha_val)
    print('Number of Epochs: %d' % best_num_epochs)
    print('Resulting Model True Accuracy Estimation: %.3f' % test_acc)


main()

