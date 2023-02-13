'''
E13
Jack Miller
Daniel Gibson
'''

#From https://datascience.stackexchange.com/questions/26640/how-to-check-for-overfitting-with-svm-and-iris-data
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn import svm
from sklearn.metrics import accuracy_score, det_curve
from sklearn.model_selection import cross_val_score

iris = load_iris()
X = iris.data[:, :4]
y = iris.target

# Evaluate an SVM on the data with the specified model configurations
def run_experiment(C, kernel, degree=0):
    total_train_error = 0
    total_test_error = 0

    # Fit each configuration 30 times, find the average of the results
    for i in range(30):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        svm_model = svm.SVC(kernel=kernel, C=C, gamma='auto', probability=True, degree=degree)
        svm_model.fit(X_train, y_train)

        #scores = cross_val_score(svm_model, X, y, cv=30)
        #print(scores.mean())

        predictions = svm_model.predict(X_train)
        train_error = 1 - accuracy_score(predictions, y_train)

        predictions = svm_model.predict(X_test)
        test_error = 1 - accuracy_score(predictions, y_test)

        total_train_error += train_error
        total_test_error += test_error

    avg_train_error = total_train_error / 30
    avg_test_error = total_test_error / 30

    return avg_train_error, avg_test_error

# For each of the kernel, C, and degree options, call run_experiment() and record results.
# Return the best model configuration
def get_combo():
    kernels = ['rbf', 'linear', 'poly']
    C_options = [0.1, 1, 10, 100]
    degrees = [2, 3, 4]

    lowest_test_error = 100
    lowest_train_error = 0
    best_combination = []
    for c in C_options:
        for k in kernels:
            if k == 'poly':
                for d in degrees:
                    train_error, test_error = run_experiment(c, k, d)
                    if test_error < lowest_test_error:
                        lowest_test_error = test_error
                        lowest_train_error = train_error
                        best_combination = [c, k, d]
            else:
                train_error, test_error = run_experiment(c, k)
                if test_error < lowest_test_error:
                    lowest_test_error = test_error
                    lowest_train_error = train_error
                    best_combination = [c, k, -1]

    final_train_error, final_test_error = 0, 0
    if best_combination[2] == -1:
        final_train_error, final_test_error = run_experiment(best_combination[0], best_combination[1])
    else:
        final_train_error, final_test_error = run_experiment(best_combination[0], best_combination[1], best_combination[2])

    print("Results: \n")
    print("Optimal Model Configuration: C = %(c).2f, kernel = %(kernel)s, degree = %(d)d" % {'c': best_combination[0], 'kernel': best_combination[1], 'd': best_combination[2]})
    print("Testing Error Rate: %(error).4f" % {'error': lowest_train_error})
    print("Apparent Error Rate: %(error).4f" % {'error': lowest_test_error})

    best_combination.append(lowest_train_error)
    best_combination.append(lowest_test_error)
    return best_combination

# Calls get_combo() 50 times, records each resulting best combination out to CSV file
def main():
    with open('results.csv', 'w') as file:
        file.write("C,kernel,degree\n")
        for i in range(50):
            results = get_combo()
            new_results = [str(x) for x in results]
            file.write(f"{','.join(new_results)}\n")


main()

