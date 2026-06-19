import os
import math
import random
import xgboost
import numpy as np 
from hyperopt import fmin, tpe, hp, rand, anneal, partial, Trials
from sklearn.decomposition import PCA
import  xgboost as xgb
import warnings
import pandas as pd
warnings.filterwarnings(module='sklearn*', action='ignore', category=DeprecationWarning)
warnings.filterwarnings(action='ignore', category=DeprecationWarning)
np.set_printoptions(suppress=True)
import matplotlib as mpl

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial']
mpl.rcParams['mathtext.fontset'] = 'custom'
mpl.rcParams['mathtext.it'] = 'Arial:italic'
mpl.rcParams['mathtext.rm'] = 'Arial'
mpl.rcParams['mathtext.bf'] = 'Arial:bold'

# ------------------------------------------------------------------------------
# --- Build Dataset
# ------------------------------------------------------------------------------
import pandas as pd
import numpy as np

def data_load(filename, Sheet):
    data = pd.read_excel(filename, sheet_name=Sheet)  # Read the specified sheet from the excel file
    result = data.values.tolist()  # Convert the data to a nested list
    nrows, ncols = data.shape  # Get the number of rows and columns

    catalyst = [x[0] for x in result] 
    cat_type = [x[1] for x in result]  
    SA = [x[2] for x in result]  
    substrate = [x[3] for x in result]
    label = np.array([x[4] for x in result])
    features = np.array([x[5:ncols] for x in result])  
    features_name = np.array(data.columns[5:ncols])  

    return catalyst, cat_type, SA, substrate, features, label, features_name

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# Handle missing values in the dataset using mean imputation
def imputer(dataArr):
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp.fit(dataArr)
    dataArr_full = imp.transform(dataArr)
    return dataArr_full

# Normalize/standardize the features in X and the target variable Y
def normData(dataX, dataY):
    X = dataX
    Y = dataY
    data_Y_filename = open(result_path + "data_Y" + ".dat", "a")
    for i in dataY:
        print(i, file=data_Y_filename)
    
    data_Y_filename.close()

    return X, Y

# Split the dataset into training and testing sets using the hold-out method
def splitDataHO(X, Y, testSize, random_state):
    trainX, testX, trainY, testY = train_test_split(X, Y, test_size=testSize, random_state=random_state)
    
    trainY_filename = open(result_path + "trainY" + ".dat", "a")
    testY_filename = open(result_path + "testY" + ".dat", "a")
    for i in trainY:
        print(i, file=trainY_filename)
    for i in testY:
        print(i, file=testY_filename)
    trainY_filename.close()
    testY_filename.close()
    
    return trainX, testX, trainY, testY

# Split the dataset into training/testing data and prediction data
def split_train_predict(X, Y, number_train_test):
    items_number = len(X)
    train_test_data_Arr = []
    predict_data_Arr = []
    train_test_label_Arr = []
    predict_label_Arr = []

    for i in range(number_train_test):
        train_test_data = X[i]
        train_test_label = Y[i]
        train_test_data_Arr.append(train_test_data)
        train_test_label_Arr.append(train_test_label)

    for j in range(number_train_test, items_number):
        predict_data = X[j]
        predict_label = Y[j]
        predict_data_Arr.append(predict_data)
        predict_label_Arr.append(predict_label)

    return np.array(train_test_data_Arr), np.array(train_test_label_Arr), np.array(predict_data_Arr), np.array(predict_label_Arr)

# ------------------------------------------------------------------------------
# --- Model Evaluation
# ------------------------------------------------------------------------------
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score, mean_absolute_error
import matplotlib.pyplot as plt
from matplotlib.axes._axes import _log as matplotlib_axes_logger

def ModelEvaluationRegression(testY, prediction):
    # Calculate R2 score
    R2 = r2_score(testY, prediction)
    # Calculate mean squared error
    MSE = mean_squared_error(testY, prediction)
    # Calculate mean absolute error
    MAE = mean_absolute_error(testY, prediction)
    # Calculate explained variance
    EV = explained_variance_score(testY, prediction)

    return R2, MSE, MAE, EV

def hyper_parameters_plot(parameters, trials, Loop_Step, screen_step):
    matplotlib_axes_logger.setLevel('ERROR')
    
    # Create subplots for parameter visualization
    f, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10), dpi=330)
    cmap = plt.cm.jet
    
    for i, val in enumerate(parameters):
        # Extract parameter values and corresponding losses
        xs = np.array([t['misc']['vals'][val] for t in trials.trials]).ravel()
        ys = [-t['result']['loss'] for t in trials.trials]
        xs, ys = zip(*sorted(zip(xs, ys)))

        # Scatter plot of parameter values and losses
        axes[int(i / 3), int(i % 3)].scatter(xs, ys, s=50, linewidth=0.01, alpha=0.5, norm=0.5,
                                             c=cmap(float(i) / len(parameters)))
        axes[int(i / 3), int(i % 3)].set_title(val, fontsize=20, fontweight="bold", fontname='Arial')

        # Set tick parameters and axis limits
        axes[int(i / 3), int(i % 3)].tick_params(labelsize=18, direction='out', width=2, length=6)
        axes[int(i / 3), int(i % 3)].set_ylim((0.5, 1))
        labels = axes[int(i / 3), int(i % 3)].get_xticklabels() + axes[int(i / 3), int(i % 3)].get_yticklabels()
        [label.set_fontname('Arial') for label in labels]
        [label.set_fontweight('bold') for label in labels]
        axes[int(i / 3), int(i % 3)].spines['top'].set_linewidth(2.5)
        axes[int(i / 3), int(i % 3)].spines['bottom'].set_linewidth(2.5)
        axes[int(i / 3), int(i % 3)].spines['right'].set_linewidth(2.5)
        axes[int(i / 3), int(i % 3)].spines['left'].set_linewidth(2.5)
    
    # Adjust subplot layout and save the figure
    plt.tight_layout()
    plt.savefig(result_path + 'hyper_parameters_trials_' + str(Loop_Step) + "_" + str(screen_step) + '.png', format='png')
    plt.close()

def plot_regression_results(trainY, testY, pred_trainY, std_pred_trainY, pred_testY, std_pred_testY, Loop_Step, screen_step):
    # Calculate R2 score and mean absolute error for training set
    R2_average_train = r2_score(trainY, pred_trainY)
    MAE_average_train = mean_absolute_error(trainY, pred_trainY)
    # Calculate R2 score and mean absolute error for test set
    R2_average_test = r2_score(testY, pred_testY)
    MAE_average_test = mean_absolute_error(testY, pred_testY)

    plt.figure(figsize=(4, 3), dpi=300)
    plt.scatter(trainY, pred_trainY, c='blue', alpha=0.6,
               label=f'Train ($R^2$={R2_average_train:.2f}, $MAE$={MAE_average_train:.2f} eV)')
    plt.scatter(testY, pred_testY, c='red', alpha=0.6,
               label=f'Test ($R^2$={R2_average_test:.2f}, $MAE$={MAE_average_test:.2f} eV)')    
    
    min_val = min(np.min(trainY), np.min(testY))
    max_val = max(np.max(trainY), np.max(testY))
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=1)
    
    plt.xlabel('DFT Calculated $E_{\mathrm{TS}}$', fontsize=12)
    plt.ylabel('ML Predicted $E_{\mathrm{TS}}$', fontsize=12)
    plt.legend(fontsize=8)
    plt.tight_layout()

    # Save the figure and close the plot
    plt.savefig(result_path + "Regression_" + str(Loop_Step) + "_" + str(screen_step) + ".jpg", format="jpg"
                , bbox_inches='tight')
    plt.close()
    
# ------------------------------------------------------------------------------
# --- Parameter Selection
# ------------------------------------------------------------------------------

def GBRegression(trainX, testX, trainY, testY, max_evals, Loop_Step, screen_step):
    # Open files for storing results
    GBR_model_score_filename = open(result_path + "hyper_parameter_selection_" + str(Loop_Step) + "_" + str(screen_step) + ".dat", "a")
    predicted_testY_file = open(result_path + "predicted_testY_" + str(Loop_Step) + "_" + str(screen_step) + ".dat", 'a')
    predicted_trainY_file = open(result_path + "predicted_trainY_" + str(Loop_Step) + "_" + str(screen_step) + ".dat", 'a')

    # Define the parameter space for hyperopt ## need to be modified to a wider range
    parameter_space_gbr = {
        "colsample_bytree": hp.uniform("colsample_bytree", 0.685, 0.705),
        "max_depth": hp.quniform("max_depth", 6, 8, 1),
        "n_estimators": hp.quniform("n_estimators", 180, 200, 1),
        "learning_rate": hp.uniform("learning_rate", 0.47, 0.49),
        "subsample": hp.uniform("subsample", 0.72, 0.75),
        "min_child_weight": hp.uniform("min_child_weight", 1.42, 1.44),
        "gamma": hp.uniform("gamma", 0.30, 0.33)
    }

    count = 0  # Counter for parameter combinations

    def function(argsDict):
        # Extract parameter values from argsDict
        colsample_bytree = argsDict["colsample_bytree"]
        max_depth = argsDict["max_depth"]
        n_estimators = argsDict['n_estimators']
        learning_rate = argsDict["learning_rate"]
        subsample = argsDict["subsample"]
        min_child_weight = argsDict["min_child_weight"]
        gamma = argsDict["gamma"]

        # Create and train the XGBoost regression model
        clf = xgb.XGBRegressor(nthread=4,
                               colsample_bytree=colsample_bytree,
                               max_depth=int(max_depth),
                               n_estimators=int(n_estimators),
                               learning_rate=learning_rate,
                               subsample=subsample,
                               min_child_weight=min_child_weight,
                               gamma=gamma,
                               random_state=int(42),
                               objective="reg:squarederror"
                               )
        clf.fit(trainX, trainY)
        prediction = clf.predict(testX)

        nonlocal count
        count += 1

        # Evaluate the model's performance
        R2, MSE, MAE, EV = ModelEvaluationRegression(testY, prediction)
        print("No.%s, R2: %f, MSE: %f, MAE: %f, EV: %f" % (str(count), R2, MSE, MAE, EV), argsDict, file=GBR_model_score_filename)

        # Return the negative R2 value for hyperopt to maximize
        return -R2

    trials = Trials()
    best = fmin(function, parameter_space_gbr, algo=tpe.suggest, max_evals=max_evals, trials=trials, verbose=False)
    parameters = ['colsample_bytree', 'max_depth', 'n_estimators', 'learning_rate', 'gamma', 'min_child_weight']
    # hyper_parameters_plot(parameters, trials, Loop_Step, screen_step)

    # Retrieve the best hyperparameters
    colsample_bytree = best["colsample_bytree"]
    max_depth = best["max_depth"]
    n_estimators = best['n_estimators']
    learning_rate = best["learning_rate"]
    subsample = best["subsample"]
    min_child_weight = best["min_child_weight"]
    gamma = best["gamma"]

    print("The_best_parameter：", best, file=GBR_model_score_filename)

    # Train the best model using the best hyperparameters
    best_model = xgb.XGBRegressor(nthread=4,
                                  colsample_bytree=colsample_bytree,
                                  max_depth=int(max_depth),
                                  n_estimators=int(n_estimators),
                                  learning_rate=learning_rate,
                                  subsample=subsample,
                                  min_child_weight=min_child_weight,
                                  gamma=gamma,
                                  random_state=int(42),
                                  objective="reg:squarederror"
                                  )
    best_model.fit(trainX, trainY)
    best_model_pred_testY = best_model.predict(testX)
    best_model_pred_trainY = best_model.predict(trainX)
    R2_test, MSE_test, MAE_test, EV_test = ModelEvaluationRegression(testY, best_model_pred_testY)
    R2_train, MSE_train, MAE_train, EV_train = ModelEvaluationRegression(trainY, best_model_pred_trainY)
    print("The_best_model_train_score:", R2_train, MSE_train, MAE_train, EV_train, file=GBR_model_score_filename)
    print("The_best_model_test_score:", R2_test, MSE_test, MAE_test, EV_test, file=GBR_model_score_filename)

    print("The_best_model_train_score R2_train: {:.3f}, MSE_train: {:.3f}, MAE_train: {:.3f}, EV_train: {:.3f}".format(R2_train
                                                                                       , MSE_train, MAE_train, EV_train))
    print("The_best_model_test_score R2_test: {:.3f}, MSE_test: {:.3f}, MAE_test: {:.3f}, EV_test: {:.3f}".format(R2_test
                                                                                                                  , MSE_test
                                                                                                                  , MAE_test
                                                                                                                  , EV_test))
    
    
    # Save the best model and visualization
    best_model.save_model(result_path + "best_model_" + str(Loop_Step) + "_" + str(screen_step) + ".model")
    digraph = xgboost.to_graphviz(best_model)
    digraph_name = result_path + "plot_best_tree_" + str(Loop_Step) + "_" + str(screen_step) + ".gv"
    digraph.save(filename=digraph_name)

    # Write predicted values to files
    for y in best_model_pred_testY:
        print(y, file=predicted_testY_file)
    for y in best_model_pred_trainY:
        print(y, file=predicted_trainY_file)

    # Close the files
    GBR_model_score_filename.close()
    predicted_testY_file.close()
    predicted_trainY_file.close()

    return best_model, best_model_pred_trainY, best_model_pred_testY, \
           R2_train, MSE_train, MAE_train, EV_train, \
           R2_test, MSE_test, MAE_test, EV_test

# ------------------------------------------------------------------------------
# --- Prediction
# ------------------------------------------------------------------------------

# Function to create directory if it doesn't exist
def path_mkdir(path, result_subfile, screen_step):
    feature_engineering_result_file = result_file + str(result_subfile) + str(screen_step)
    if not os.path.exists(result_file):
        os.mkdir(path + result_file)
    if not os.path.exists(feature_engineering_result_file):
        os.mkdir(path + feature_engineering_result_file)
    result_path = path + feature_engineering_result_file + "/"
    return result_path

def train_test(trainX, testX, trainY, testY, screen_step):
    pred_testArr = []              # List to store predicted test values
    mean_pred_testArr = []         # List to store mean of predicted test values
    std_pred_testArr = []          # List to store standard deviation of predicted test values
    pred_trainArr = []             # List to store predicted train values
    mean_pred_trainArr = []        # List to store mean of predicted train values
    std_pred_trainArr = []         # List to store standard deviation of predicted train values
    features_importanceArr = []    # List to store feature importances
    mean_features_importanceArr = []   # List to store mean of feature importances
    std_features_importanceArr = []    # List to store standard deviation of feature importances

    for n in range(LoopStepMin, LoopStepMax):
        # Perform GBRegression and store the results
        print('Model index: {}/{}'.format(n+1,LoopStepMax))
        best_model, best_model_pred_trainY, best_model_pred_testY, R2_train, MSE_train, MAE_train, EV_train, R2_test, MSE_test, MAE_test, EV_test = \
            GBRegression(trainX, testX, trainY, testY, HyperParameter_Step, n, screen_step)
        pred_trainArr.append(best_model_pred_trainY)
        pred_testArr.append(best_model_pred_testY)
        features_importance = best_model.feature_importances_
        features_importanceArr.append(features_importance)
    
    # Calculate mean and standard deviation of predicted test values
    for i in range(np.shape(pred_testArr)[1]):
        sample = np.array(pred_testArr)[:, i]
        testsample_prediction_mean = np.mean(sample)
        testsample_prediction_std = np.std(sample, ddof=0)
        mean_pred_testArr.append(testsample_prediction_mean)
        std_pred_testArr.append(testsample_prediction_std)
    
    # Calculate mean and standard deviation of predicted train values
    for j in range(np.shape(pred_trainArr)[1]):
        sample = np.array(pred_trainArr)[:, j]
        trainsample_prediction_mean = np.mean(sample)
        trainsample_prediction_std = np.std(sample, ddof=0)
        mean_pred_trainArr.append(trainsample_prediction_mean)
        std_pred_trainArr.append(trainsample_prediction_std)
    
    # Calculate mean and standard deviation of feature importances
    for k in range(np.shape(features_importanceArr)[1]):
        sample = np.array(features_importanceArr)[:, k]
        features_importance_mean = np.mean(sample)
        features_importance_std = np.std(sample, ddof=0)
        mean_features_importanceArr.append(features_importance_mean)
        std_features_importanceArr.append(features_importance_std)

    # Plot regression results
    plot_regression_results(trainY, testY, np.array(mean_pred_trainArr), np.array(std_pred_trainArr), np.array(mean_pred_testArr), \
                            np.array(std_pred_testArr), 'average', screen_step)

    # Open files for writing predicted train, test, and feature importance values
    train_font = open(result_path + "predicted_train_average_" + str(screen_step) + ".dat", "a")
    test_font = open(result_path + "predicted_test_average_" + str(screen_step) + ".dat", "a")
    features_importance_font = open(result_path + "features_importance_average_" + str(screen_step) + ".dat", "a")
    
    # Write mean and standard deviation of predicted train values to file
    for i, j in zip(mean_pred_trainArr, std_pred_trainArr):
        print(i, j, file=train_font)
    
    # Write mean and standard deviation of predicted test values to file
    for i, j in zip(mean_pred_testArr, std_pred_testArr):
        print(i, j, file=test_font)
    
    # Write feature names, mean, and standard deviation of feature importances to file
    for i, j, k in zip(features_name, mean_features_importanceArr, std_features_importanceArr):
        print(i, j, k, file=features_importance_font)
    
    # Close the files
    train_font.close()
    test_font.close()
    features_importance_font.close()
    
    # Return the calculated values
    return mean_pred_trainArr, std_pred_trainArr, mean_pred_testArr, std_pred_testArr, mean_features_importanceArr, std_features_importanceArr

def predict(trainX, trainY, testX, testY, predictX, screen_step):
    # print("Predicting......")
    predict_font = open(result_path + "predicted_predict_average_" + str(screen_step) + ".dat", "a")
    train_set = pd.DataFrame(trainX, columns=features_name)
    dtrain = xgboost.DMatrix(train_set)
    test_set = pd.DataFrame(testX, columns=features_name)
    dtest = xgboost.DMatrix(test_set)
    predict_set = pd.DataFrame(predictX, columns=features_name)
    dpredict = xgboost.DMatrix(predict_set)
    pred_predict_Arr = []               # List to store predicted values for predictX
    mean_pred_predict_Arr = []          # List to store mean of predicted values for predictX
    std_pred_predict_Arr = []           # List to store standard deviation of predicted values for predictX
    
    for i in range(LoopStepMin, LoopStepMax):
        model_file = result_path + "best_model_" + str(i) + "_FF.model"
        best_model = xgboost.Booster(model_file=model_file)
        best_model_pred_testY = best_model.predict(dtest)
        best_model_pred_trainY = best_model.predict(dtrain)
        best_model_pred_predictY = best_model.predict(dpredict)
        R2_test, MSE_test, MAE_test, EV_test = ModelEvaluationRegression(testY, best_model_pred_testY)
        R2_train, MSE_train, MAE_train, EV_train = ModelEvaluationRegression(trainY, best_model_pred_trainY)
        # print("R2_train: %f, MSE_train: %f, MAE_train: %f, EV_train: %f" % (R2_train, MSE_train, MAE_train, EV_train))
        # print("R2_test: %f, MSE_test: %f, MAE_test: %f, EV_test: %f" % (R2_test, MSE_test, MAE_test, EV_test))
        pred_predict_Arr.append(best_model_pred_predictY)
    
    # Calculate mean and standard deviation of predicted values for predictX
    for i in range(np.shape(pred_predict_Arr)[1]):
        sample = np.array(pred_predict_Arr)[:, i]
        predictsample_prediction_mean = np.mean(sample)
        predictsample_prediction_std = np.std(sample, ddof=0)
        mean_pred_predict_Arr.append(predictsample_prediction_mean)
        std_pred_predict_Arr.append(predictsample_prediction_std)
    
    # Write mean and standard deviation of predicted values for predictX to file
    for i, j in zip(mean_pred_predict_Arr, std_pred_predict_Arr):
        print(i, j, file=predict_font)
    
    # Return the calculated values
    return

# -----------------------------------------------------------------------------
# --- feature engineering
# -----------------------------------------------------------------------------
def feature_importance(features_name, features, screen_step):
    # Open files to write normalized and original dataset
    normdata_filename = open(result_path + "norm_dataset_" + str(screen_step) +".dat", "a")
    data_filename = open(result_path + "dataset_" + str(screen_step) +".dat", "a")
    
    if n_fixed_features != 0:
        # Extract vector features
        vector_features = features[:, n_fixed_features: np.shape(features)[1]]
        vector_features_name = list(features_name[n_fixed_features: np.shape(features)[1]])
        
        # Concatenate fixed features with vector features
        new_features = np.concatenate((fixed_features, vector_features), axis=1)
        
        # Normalize features and labels
        norm_features, norm_label = normData(new_features, label)
        
        # Concatenate fixed feature names with vector feature names
        new_features_name = np.concatenate((initial_fixed_features_name, vector_features_name), axis=0)
        
        # Write normalized dataset to file
        print("catalyst, cat_type, SA, substrate", [name for name in new_features_name], "label", file=normdata_filename)
        for a, i, j, k, x, y in zip(catalyst, cat_type, SA, substrate, norm_features, label):
            print(a, i, j, k, [item for item in x], y, file=normdata_filename)
        
        # Write original dataset to file
        print("catalyst, cat_type, SA, substrate", [name for name in new_features_name], "label", file=data_filename)
        for a, i, j, k, x, y in zip(catalyst, cat_type, SA, substrate, features, label):
            print(a, i, j, k, [item for item in x], y, file=data_filename)
    else:
        # Use all features as vector features
        vector_features = features
        vector_features_name = list(features_name)
        
        # Normalize vector features and labels
        norm_vector_features, norm_label = normData(vector_features, label)
        norm_features = norm_vector_features
        
        # Write normalized dataset to file
        print("catalyst, cat_type, SA, substrate", [name for name in vector_features_name], "label", file=normdata_filename)
        for a, i, j, k, x, y in zip(catalyst, cat_type, SA, substrate, norm_features, label):
            print(i, j, k, [item for item in x], y, file=normdata_filename)
        
        # Write original dataset to file
        print("catalyst, cat_type, SA, substrate", [name for name in vector_features_name], "label", file=data_filename)
        for a, i, j, k, x, y in zip(catalyst, cat_type, SA, substrate, vector_features, label):
            print(i, j, k, [item for item in x], y, file=data_filename)

    # Split data into train-test sets
    train_test_data_Arr, train_test_label_Arr, predict_data_Arr, predict_label_Arr = split_train_predict(norm_features,
                                                                                                         norm_label,
                                                                                                         number_sample)
    trainX, testX, trainY, testY = splitDataHO(train_test_data_Arr, train_test_label_Arr, TestSetRatio, RandomSeed)

    # Train and test the model
    mean_pred_trainArr, std_pred_trainArr, mean_pred_testArr, std_pred_testArr, \
    mean_features_importanceArr, std_features_importanceArr = train_test(trainX, testX, trainY, testY, screen_step)

    normdata_filename.close()
    data_filename.close()
    
    # Return the results
    return mean_pred_trainArr, std_pred_trainArr, mean_pred_testArr, std_pred_testArr, \
           mean_features_importanceArr, std_features_importanceArr, \
           fixed_features_name, fixed_features, vector_features_name, vector_features


def fixed_feature_engineering(mean_features_importanceArr, std_features_importanceArr, fixed_features_name,
                              fixed_features, vector_features_name, vector_features, screen_step):
    sorted_features_importance = []
    sorted_features_importance_std = []
    
    shape_fixed_features = np.shape(fixed_features)
    
    # Calculate importance of fixed features
    fixed_features_importance = np.float(np.sum(mean_features_importanceArr[0: shape_fixed_features[1]]))
    fixed_features_importance_std = np.sum(std_features_importanceArr[0: shape_fixed_features[1]]) / shape_fixed_features[1]
    
    # Calculate importance of vector features
    vector_features_importance = mean_features_importanceArr[shape_fixed_features[1]: len(mean_features_importanceArr)]
    vector_features_importance_std = std_features_importanceArr[shape_fixed_features[1]: len(mean_features_importanceArr)]
    
    # Sort vector features based on importance
    sorted_idx = np.argsort(vector_features_importance)[::-1]
    sorted_vector_features_name = np.array(vector_features_name)[sorted_idx]
    sorted_vector_features_importance = np.array(vector_features_importance)[sorted_idx]
    sorted_vector_features_importance_std = np.array(vector_features_importance_std)[sorted_idx]
    
    row_number = np.shape(vector_features)[0]
    vfeature_column_number = np.shape(vector_features)[1]
    
    # Sort vector features based on importance
    sorted_vector_features = np.zeros((row_number, vfeature_column_number))
    for i, j in zip(sorted_idx, np.arange(0, vfeature_column_number)):
        sorted_vector_features[:, j] = vector_features[:, i]
    
    # Concatenate fixed features with sorted vector features
    sorted_features = np.concatenate((fixed_features, sorted_vector_features), axis=1)
    
    # Remove last feature name and feature column from sorted vector features
    selected_sorted_vector_features_name = np.delete(sorted_vector_features_name, -1)
    selected_sorted_features = np.delete(sorted_features, -1, axis=1)
    
    # Concatenate fixed feature names with sorted vector feature names
    sorted_features_name = np.concatenate((fixed_features_name, selected_sorted_vector_features_name), axis=0)
    
    # Append fixed features importance to the list
    sorted_features_importance.append(fixed_features_importance)
    
    # Append vector features importance to the list
    for item in sorted_vector_features_importance:
        sorted_features_importance.append(item)
    
    # Append fixed features importance standard deviation to the list
    sorted_features_importance_std.append(fixed_features_importance_std)
    
    # Append vector features importance standard deviation to the list
    for item in sorted_vector_features_importance_std:
        sorted_features_importance_std.append(item)
    
    # Write sorted relative feature importance to file
    font = open(result_path + "sorted_relative_feature_importance_" + str(screen_step) + ".dat", 'a')
    for i, j, k in zip(sorted_features_name, sorted_features_importance, sorted_features_importance_std):
        print(i, j, k, file=font)
    font.close()
    
    # Return the sorted features and their names
    return sorted_features_name, selected_sorted_features


def vector_feature_engineering(mean_features_importanceArr, std_features_importanceArr,
                               vector_features_name, vector_features, screen_step):
    sorted_features_importance = []
    sorted_features_importance_std = []
    
    # Calculate importance of vector features
    vector_features_importance = mean_features_importanceArr
    vector_features_importance_std = std_features_importanceArr
    
    # Sort vector features based on importance
    sorted_idx = np.argsort(vector_features_importance)[::-1]
    sorted_vector_features_name = np.array(vector_features_name)[sorted_idx]
    sorted_vector_features_importance = np.array(vector_features_importance)[sorted_idx]
    sorted_vector_features_importance_std = np.array(vector_features_importance_std)[sorted_idx]
    
    row_number = np.shape(vector_features)[0]
    vfeature_column_number = np.shape(vector_features)[1]
    
    # Sort vector features based on importance
    sorted_vector_features = np.zeros((row_number, vfeature_column_number))
    for i, j in zip(sorted_idx[1:vfeature_column_number], np.arange(0, vfeature_column_number)):
        sorted_vector_features[:, j] = vector_features[:, i]
    
    # The sorted features only include vector features
    sorted_features = sorted_vector_features
    
    # Remove last feature name and feature column from sorted vector features
    selected_sorted_vector_features_name = np.delete(sorted_vector_features_name, -1)
    selected_sorted_features = np.delete(sorted_features, -1, axis=1)
    
    # The sorted feature names only include vector feature names
    sorted_features_name = selected_sorted_vector_features_name
    
    # Append vector features importance to the list
    for item in sorted_vector_features_importance:
        sorted_features_importance.append(item)
    
    # Append vector features importance standard deviation to the list
    for item in sorted_vector_features_importance_std:
        sorted_features_importance_std.append(item)
    
    # Write sorted relative feature importance to file
    font = open(result_path + "sorted_relative_feature_importance_" + str(screen_step) + ".dat", 'a')
    for i, j, k in zip(sorted_features_name, sorted_features_importance, sorted_features_importance_std):
        print(i, j, k, file=font)
    font.close()
    
    # Return the sorted features and their names
    return sorted_features_name, selected_sorted_features

filename1 = 'Source_1_8635_Oads_270_3439_Ea_10_200_6l_21d'
X1 = np.load(f'{filename1}/target_features.npy')
Y1 = np.load(f'{filename1}/target_labels.npy')
print(X1.shape)
print(Y1)

filename2 = 'Source_2_13113_H_in_O_ads_270_3439_Ea_10_200_4l_14d'
X2 = np.load(f'{filename2}/target_features.npy')
Y2 = np.load(f'{filename2}/target_labels.npy')
print(X2.shape)
print(Y2)

# filename3 = 'Source_3_8646_H_in_M_ads_270_3439_Ea_10_200_4l_15d'
# X3 = np.load(f'{filename3}/target_features.npy')
# Y3 = np.load(f'{filename3}/target_labels.npy')
# print(X3.shape)
# print(Y3)

# filename4 = 'Source_4_6609_OHads_270_3439_Ea_10_200_7l_18d'
# X4 = np.load(f'{filename4}/target_features.npy')
# Y4 = np.load(f'{filename4}/target_labels.npy')
# print(X4.shape)
# print(Y4)

filename5 = 'Source_5_2079_CH3ads_270_3439_Ea_10_200_5l_16d'
X5 = np.load(f'{filename5}/target_features.npy')
Y5 = np.load(f'{filename5}/target_labels.npy')
print(X5.shape)
print(Y5)

X_combined = np.concatenate([X1, X2, X5], axis=1)  

try:
    assert np.array_equal(~np.isnan(Y1)
                          , ~np.isnan(Y2)) and np.array_equal(~np.isnan(Y1)
                          , ~np.isnan(Y5))
    Y_final = Y1.copy()
    print("Labels are identical. Using Y1.")
except AssertionError:
    Y_final = np.mean([Y1, Y2, Y5], axis=0)
    print("Labels differ. Using averaged values.")

print("Combined X shape:", X_combined.shape)
print("Final Y shape:", Y_final.shape)

label_t_non_null_count = np.sum(~np.isnan(Y_final))
label_t_null_count = np.sum(np.isnan(Y_final))

print(f"Target value number: {label_t_non_null_count}")
print(f"Target nan number: {label_t_null_count}")
print(f"Target total number: {len(Y_final)}")

# ------------------------------------------------------------------------------
# --- 运行
# ------------------------------------------------------------------------------
path = os.getcwd() + "/"

result_file = f'3_Source' + \
              f'_{len(Y_final)}_{label_t_non_null_count}' + \
              f'_{X_combined.shape[1]}_Ea-merged-2000-20-0.1'  # Output results folders
result_subfile = "/Feature_Engineering_Result_"  # Output results subdirectory

n_fixed_features = X_combined.shape[1]  # the number of fixed features
HyperParameter_Step = 2000  # Hyperparameter search steps
LoopStepMin = 0  # Minimum number of iteration loop
LoopStepMax = 20  # Maximum  number of iteration loop
RandomSeed = 30  # Random seed for data split
TestSetRatio = 0.1  # Test set ratio
number_sample = 270 # Number of train&test data

features = X_combined
label = Y_final

features_name = [f'features_{i}' for i in range(n_fixed_features)]

fixed_features = features[:, 0: n_fixed_features]
initial_fixed_features_name = features_name[0:n_fixed_features]
ScreenStep = np.shape(features[:, n_fixed_features: np.shape(features)[1]])[1]

if np.shape(features)[1] > n_fixed_features:
    if n_fixed_features != 0:
        for i in range(0, ScreenStep):
            screen_step = i
            result_path = path_mkdir(path, result_subfile, screen_step)
            mean_pred_trainArr, std_pred_trainArr, mean_pred_testArr, std_pred_testArr, \
            mean_features_importanceArr, std_features_importanceArr, \
            fixed_features_name, fixed_features, vector_features_name, vector_features = feature_importance(
                features_name,
                features,
                screen_step)
            selected_sorted_features_name, selected_sorted_features = \
                fixed_feature_engineering(mean_features_importanceArr, std_features_importanceArr, fixed_features_name,
                                    fixed_features, vector_features_name,
                                    vector_features, screen_step)
            features_name = np.concatenate(
                (initial_fixed_features_name, selected_sorted_features_name[1: len(selected_sorted_features_name)]),
                axis=0)
            features = selected_sorted_features

    else:
        for i in range(0, ScreenStep):
            screen_step = i
            result_path = path_mkdir(path, result_subfile, screen_step)
            mean_pred_trainArr, std_pred_trainArr, mean_pred_testArr, std_pred_testArr, \
            mean_features_importanceArr, std_features_importanceArr, \
            fixed_features_name, fixed_features, vector_features_name, vector_features = feature_importance(
                features_name,
                features,
                screen_step)
            selected_sorted_features_name, selected_sorted_features = \
                vector_feature_engineering(mean_features_importanceArr, std_features_importanceArr,
                                           vector_features_name, vector_features, screen_step)
            features_name = selected_sorted_features_name

            features = selected_sorted_features


else:
    screen_step = "FF"
    result_path = path_mkdir(path, result_subfile, screen_step)
    norm_features, norm_label = normData(features, label)
    train_test_data_Arr, train_test_label_Arr, predict_data_Arr, predict_label_Arr = split_train_predict(
        norm_features,
        norm_label,
        number_sample)
    trainX, testX, trainY, testY = splitDataHO(train_test_data_Arr, train_test_label_Arr, TestSetRatio, RandomSeed)
    predictX = predict_data_Arr; predictY = predict_label_Arr
    mean_pred_trainArr, std_pred_trainArr, \
    mean_pred_testArr, std_pred_testArr, \
    mean_features_importanceArr, std_features_importanceArr = train_test(trainX, testX, trainY, testY, screen_step)
    if number_sample != len(label):
        predict(trainX, trainY, testX, testY, predictX, screen_step)
