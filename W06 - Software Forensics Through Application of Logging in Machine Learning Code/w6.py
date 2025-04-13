from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets, linear_model
import pandas as pd
import numpy as np 
import mnist
import myLogger
from sklearn.model_selection import KFold
from sklearn.datasets import load_iris
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow import keras

# Initialize logger
logObj  = myLogger.giveMeLoggingObject()

def readData():

    # (1) Log the start of the dataset loading
    logObj.info("Starting to read the Iris dataset.")

    iris = datasets.load_iris()

    # (2) Log the successful loading of the dataset
    logObj.info("Iris dataset loaded successfully. Shape: data={}, target={}".format(iris.data.shape, iris.target.shape))

    print(type(iris.data), type(iris.target))
    X = iris.data
    Y = iris.target
    df = pd.DataFrame(X, columns=iris.feature_names)

    # (3) Log the conversion of the dataset to a DataFrame
    logObj.info("Iris dataset converted to DataFrame. Columns: {}".format(df.columns.tolist()))

    print(df.head())

    return df 

def makePrediction():

    # (4) Log the start of the KNN prediction
    logObj.info("Starting KNN prediction on Iris dataset." )

    iris = datasets.load_iris()
    knn = KNeighborsClassifier(n_neighbors=6)
    knn.fit(iris['data'], iris['target'])

    # (5) Log the successful training of the KNN model
    logObj.info("KNN model trained successfully with n_neighbors=6.")

    X = [
        [5.9, 1.0, 5.1, 1.8],
        [3.4, 2.0, 1.1, 4.8],
    ]

    # (6) Log input data for predictions
    logObj.info("Prediction input data: {}".format(X))

    prediction = knn.predict(X)

    # (7) Log the prediction result
    logObj.info("KNN prediction completed. Results: {}".format(prediction))

    print(prediction)    

def doRegression():

    # (8) Log the start of the regression analysis
    logObj.info("Starting regression analysis on Diabetes dataset.")

    diabetes = datasets.load_diabetes()
    diabetes_X = diabetes.data[:, np.newaxis, 2]
    diabetes_X_train = diabetes_X[:-20]
    diabetes_X_test = diabetes_X[-20:]
    diabetes_y_train = diabetes.target[:-20]
    diabetes_y_test = diabetes.target[-20:]

    # (9) Log the successful splitting of the dataset
    logObj.info("Diabetes dataset split into training and testing sets.")

    regr = linear_model.LinearRegression()
    regr.fit(diabetes_X_train, diabetes_y_train)

    # (10) Log the successful training of the regression model
    logObj.info("Linear regression model trained successfully.")

    diabetes_y_pred = regr.predict(diabetes_X_test)

    # (11) Log the prediction results
    logObj.info("Regression predictions completed. Predictions: {}".format(diabetes_y_pred))


def doDeepLearning():
    
    # (12) Log the start of the deep learning model training
    logObj.info("Starting deep learning model training on MNIST dataset.")


    (train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()

    # (13) Log the successful loading of the MNIST dataset
    logObj.info("MNIST dataset loaded successfully. Training Samples: {}, Testing Samples: {}".format(len(train_images), len(test_images)))

    train_images = (train_images / 255) - 0.5
    test_images = (test_images / 255) - 0.5


    train_images = np.expand_dims(train_images, axis=3)
    test_images = np.expand_dims(test_images, axis=3)

    num_filters = 8
    filter_size = 3
    pool_size = 2

    model = Sequential([
    Conv2D(num_filters, filter_size, input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=pool_size),
    Flatten(),
    Dense(10, activation='softmax'),
    ])

    # Compile the model.

    # (14) Log the start of the model compilation
    logObj.info("Compiling the deep learning model.")

    model.compile(
    'adam',
    loss='categorical_crossentropy',
    metrics=['accuracy'],
    )

    # (15) Log the successful compilation of the model
    logObj.info("Deep learning model compiled successfully.")

    # Train the model.

    # (16) Log the start of the model training
    logObj.info("Training the deep learning model.")

    model.fit(
    train_images,
    to_categorical(train_labels),
    epochs=3,
    validation_data=(test_images, to_categorical(test_labels)),
    )

    # (17) Log the successful training of the model
    logObj.info("Deep learning model trained successfully.")

    model.save_weights('cnn.weights.h5')

    # (18) Log model weights saving
    logObj.info("Model weights saved to 'cnn.weights.h5'.")

    predictions = model.predict(test_images[:5])

    # (19) Log the prediction results
    logObj.info("Deep learning model predictions completed. Predictions: {}".format(np.argmax(predictions, axis=1)))

    print(np.argmax(predictions, axis=1)) # [7, 2, 1, 0, 4]

    print(test_labels[:5]) # [7, 2, 1, 0, 4]

def k_fold_cv_mlp(n_splits):
  
    # (20) Log the start of K-Fold Cross-Validation
    logObj.info("Starting K-Fold Cross-Validation with MLPClassifier.")

    iris_data = load_iris()
    X_data = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)
    ## to numpy
    X=  X_data.to_numpy()
    y = iris_data.target


    kf = KFold(n_splits)
    folds = []

    for train_index, test_index in kf.split(X):
        folds.append((train_index, test_index))


    # Initialize machine learning model, MLP
    model = MLPClassifier(hidden_layer_sizes=(256,128,64,32),activation="relu",random_state=1)

    # Initialize a list to store the evaluation scores
    scores = []
    ## Initialize fold index
    fold_index = 0

    
    # Iterate through each fold
    for train_indices, test_indices in folds:
        X_train, y_train = X[train_indices], y[train_indices]
        X_test, y_test = X[test_indices], y[test_indices]


        fold_index += 1
        print(f"Fold {fold_index}:")

        # (21) Log the folding preparation
        logObj.info(f"Fold {fold_index}: Training and testing data prepared.")
        
        # scale data
        sc_X = StandardScaler()
        X_train_scaled=sc_X.fit_transform(X_train)
        X_test_scaled=sc_X.transform(X_test)

        # Train the model on the training data
        model.fit(X_train_scaled, y_train)

        # (22) Log the successful training of the model for each fold
        logObj.info(f"Fold {fold_index}: Model trained successfully.")

        # Make predictions on the test data
        y_pred = model.predict(X_test_scaled)

        # Calculate the accuracy score for this fold
        fold_score = accuracy_score(y_test, y_pred)
        print(f"Fold test score {fold_score}:")

        # (23) Log the accuracy score for each fold
        logObj.info(f"Fold {fold_index}: Accuracy score: {fold_score}")

        # Append the fold score to the list of scores
        scores.append(fold_score)

    # Calculate the mean accuracy across all folds
    mean_accuracy = np.mean(scores)

    # (24) Log the mean accuracy across all folds
    logObj.info("K-Fold Cross-Validation completed. Mean Accuracy: {}".format(mean_accuracy))


    print("K-Fold Cross-Validation Scores:", scores)
    print("Mean Accuracy:", mean_accuracy)

if __name__=='__main__': 

    # (25) Log the start of the main program
    logObj.info("Main program started.")

    data_frame = readData()
    makePrediction() 
    doRegression() 
    my_k_fold = input('Type in k for cross valdation: ') 
    my_k_fold = int(my_k_fold)
    k_fold_cv_mlp(my_k_fold)
    doDeepLearning() 

    # (26) Log the end of the main program
    logObj.info("Main program finished.")