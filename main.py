import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import statistics 
import time
from sklearn.metrics import classification_report
import seaborn as sns
from scipy import stats
from model_train import train, plot_history
from model_train import rnn_model
from model_train import cnn_model
sns.set_theme(style="whitegrid")


VALIDATION_DATA_JSON_PATH1 = "data_validation1.json"
TRAINING_DATA_JSON_PATH1 = "data_training1.json"
VALIDATION_DATA_JSON_PATH2 = "data_validation2.json"
TRAINING_DATA_JSON_PATH2 = "data_training2.json"
VALIDATION_DATA_JSON_PATH3 = "data_validation3.json"
TRAINING_DATA_JSON_PATH3 = "data_training3.json"
RNN_MODEL_NAME = "RNN-128x3-{}".format(int(time.time()))
CNN_MODEL_NAME = "CNN-128x3-{}".format(int(time.time()))
EPOCHS = 30
BATCH_SIZE = 16
LEARNING_RATE = 0.001

# function to load the data from the .json files
def load_data(data_path):

    data = json.load(open(data_path, "r")) # load the data dicitonary from the json files 

    X = np.array(data["MFCCs"])      # X = MFCC coefficients 
    y = np.array(data["labels"])     # y = label
    return X, y

# Function to prepare the data sets for later training and validation
def prepare_dataset(dataset_training_path, dataset_validation_path):   
    
    # load of the respective training data set
    X_train, y_train = load_data(dataset_training_path) 
    
    # load of the respective validation data set
    X_validation,  y_validation = load_data(dataset_validation_path)

    return X_train, y_train, X_validation, y_validation
   
  
def box_plot(evaluation_dict):

    df_evaluation_dict = pd.DataFrame(evaluation_dict) # Conversion to pandas dataframe

    sns.set(style='whitegrid')  # set style whitegird 

    fig1, ax1 = plt.subplots(figsize=(8,6))

    # Boxplot of the test accuracy results of the CNN and RNN model 
    g = sns.boxplot(x="Model", y="Test_Accuracy",data=df_evaluation_dict, width=0.7)

    # titel 
    plt.title("Comparison Test Accuracy", fontsize=16)

    # Removal of the edges except the lowest edge
    sns.despine(top=True, right=True, left=True,bottom=False)

    # Additional specification of the average acccuracy and standard deviation in the boxplot for the RNN model
    mean = round(df_evaluation_dict['Test_Accuracy'][df_evaluation_dict['Model']=='RNN-Model'].mean(),3)
    sd = round(df_evaluation_dict['Test_Accuracy'][df_evaluation_dict['Model']=='RNN-Model'].std(),3)
    textstr = "$\overline {x}$" + f" = {mean} \ns = {sd}"
    props = {'boxstyle': 'round', 'facecolor' :'b', 'alpha' :0.2}
    g.text(-0.30, 0.835, textstr, fontsize=12, bbox=props)

    # Additional specification of the average acccuracy and standard deviation in the boxplot for the CNN model
    mean = round(df_evaluation_dict['Test_Accuracy'][df_evaluation_dict['Model']=='CNN-Model'].mean(),3)
    sd = round(df_evaluation_dict['Test_Accuracy'][df_evaluation_dict['Model']=='CNN-Model'].std(),3)
    textstr = "$\overline {x}$" + f" = {mean} \ns = {sd}"
    props = {'boxstyle': 'round', 'facecolor' :'orange', 'alpha' :0.2}
    g.text(.83, 0.835, textstr, fontsize=12, bbox=props)

    fig1.savefig('Vergleich_Accuracy_CNN_RNN.pdf')   # save figure as .pdf file
    plt.tight_layout()
    plt.show()    



    fig2, ax2 = plt.subplots(figsize=(8,6)) # Definition of a second figure
    
    # Boxplot of the test loss results of the CNN and RNN model
    g = sns.boxplot(x="Model", y="Test_Loss", data=df_evaluation_dict, width=0.7)

    # title
    plt.title("Comparison Test Loss", fontsize=16)


    # Removal of the edges except the lowest edge
    sns.despine(top=True, right=True, left=True,bottom=False)

    # Additional specification of the average loss and standard deviation in the boxplot for the RNN model
    mean = round(df_evaluation_dict['Test_Loss'][df_evaluation_dict['Model']=='RNN-Model'].mean(),3)
    sd = round(df_evaluation_dict['Test_Loss'][df_evaluation_dict['Model']=='RNN-Model'].std(),3)
    textstr = "$\overline {x}$" + f" = {mean} \ns = {sd}"
    props = {'boxstyle': 'round', 'facecolor' :'b', 'alpha' :0.2}
    g.text(-0.30, 0.265, textstr, fontsize=12, bbox=props)

    # Additional specification of the average loss and standard deviation in the boxplot for the CNN model
    mean = round(df_evaluation_dict['Test_Loss'][df_evaluation_dict['Model']=='CNN-Model'].mean(),3)
    sd = round(df_evaluation_dict['Test_Loss'][df_evaluation_dict['Model']=='CNN-Model'].std(),3)
    textstr = "$\overline {x}$" + f" = {mean} \ns = {sd}"
    props = {'boxstyle': 'round', 'facecolor' :'orange', 'alpha' :0.2}
    g.text(.83, 0.265, textstr, fontsize=12, bbox=props)

    fig2.savefig('Vergleich_Loss_CNN_RNN.pdf')   # save figure as .pdf file
    plt.tight_layout()
    plt.show()    


# Function to summarize the training process of the speaker independent 3-fold cross validation.
# Average values and standard deviation +-1 for the 3 runs per model are displayed
def plot_history_zusammenfassung(history1, history2, history3, model_name, epochs):

    metrics1 = history1.history
    metrics2 = history2.history
    metrics3 = history3.history
    
    # Definition of the epochs in metrics
    epoch_lst = []
    for i in range(epochs):
        epoch_lst.append(i)
    
    metrics1['Epoch'] = epoch_lst
    metrics2['Epoch'] = epoch_lst
    metrics3['Epoch'] = epoch_lst
    
    # Combining the metrics into one structure
    for i in range(epochs) : 
        metrics1['Epoch'].append(metrics2['Epoch'][i])
        metrics1['loss'].append(metrics2['loss'][i])
        metrics1['accuracy'].append(metrics2['accuracy'][i])
        metrics1['val_loss'].append(metrics2['val_loss'][i])
        metrics1['val_accuracy'].append(metrics2['val_accuracy'][i])
    
    for i in range(epochs) : 
        metrics1['Epoch'].append(metrics3['Epoch'][i])
        metrics1['loss'].append(metrics3['loss'][i])
        metrics1['accuracy'].append(metrics3['accuracy'][i])
        metrics1['val_loss'].append(metrics3['val_loss'][i])
        metrics1['val_accuracy'].append(metrics3['val_accuracy'][i])
    
    
    df = pd.DataFrame(metrics1) # Conversion to pandas dataframe

    # Summarized representation of the training and validation loss of the 3-fold cross validation.
    # Average values and standard deviation +-1 of the loss for the 3 runs per model are shown.
    plt.figure(figsize=(16, 16))
    plt.subplot(2, 1, 1)
    sns.lineplot(x="Epoch", y="val_loss",ci="sd",  data=df, color="r",label='val loss') 
    sns.lineplot(x="Epoch", y="loss",ci="sd",  data=df, color="b", label='train loss')
    plt.xlabel("Epoche" , fontsize=20)
    plt.ylabel("Loss ", fontsize=20)
    plt.title('Loss evaluation', fontsize=25)
    plt.legend(fontsize = 25)
    
    # Summarized representation of the training and validation accuracy of the 3-fold cross validation.
    # Average values and standard deviation +-1 of the accuracy for the 3 runs per model are shown.
    plt.subplot(2, 1, 2)
    sns.lineplot(x="Epoch", y="val_accuracy",ci="sd",  data=df, color="r",label='val accuracy') 
    sns.lineplot(x="Epoch", y="accuracy",ci="sd",  data=df, color="b",label='train accuracy') 
    plt.title('Accuracy evaluation', fontsize=25)
    plt.xlabel("Epoche" , fontsize=20)
    plt.ylabel("Accuracy (Genauigkeit) ", fontsize=20)
    plt.legend(fontsize = 25)

    plt.tight_layout()
    plt.show()
    plt.savefig('Modell_Accuracy_and_Loss{}.pdf'.format(model_name))   #  Abspeichern der figure als .pdf Datei

# Function to calculate the average classification report of the 3-fold cross validation for both models.
def classification_report_models_average(classification_report_RNN1, classification_report_RNN2, classification_report_RNN3, 
                                         classification_report_CNN1, classification_report_CNN2, classification_report_CNN3):
    
    classification_report_average_RNN = classification_report_RNN1

    # Calculation of average of precision, recall and F1 score and standard deviation of F1 score for each command value (RNN model)
    for i in ['links', 'rechts','vor','zurück','start','stop','schneller','langsamer','Drehung links','Drehung rechts']:
        classification_report_average_RNN[i]['precision'] = statistics.mean( [classification_report_RNN1[i]['precision'], classification_report_RNN2[i]['precision'] , classification_report_RNN3[i]['precision']])
        classification_report_average_RNN[i]['recall'] = statistics.mean( [classification_report_RNN1[i]['recall'], classification_report_RNN2[i]['recall'] , classification_report_RNN3[i]['recall']])
        classification_report_average_RNN[i]['f1-score'] = statistics.mean( [classification_report_RNN1[i]['f1-score'], classification_report_RNN2[i]['f1-score'] , classification_report_RNN3[i]['f1-score']])
        standardabweichung_f1_RNN = statistics.stdev( [classification_report_RNN1[i]['f1-score'], classification_report_RNN2[i]['f1-score'] , classification_report_RNN3[i]['f1-score']])
        print('Standard deviation of the F1 score for the RNN model for the command word {} is {}'.format(i, standardabweichung_f1_RNN))
    
    print("\nAVERAGE CLASSIFICATION REPORT of RNN-LSTM-MODEL\n")
    df_classification_report_average = pd.DataFrame(classification_report_average_RNN)  
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # Display of the average classification report in the console
        print(df_classification_report_average)
    
    classification_report_average_CNN = classification_report_CNN1 
    
    # Calculation of average of precision, recall and F1 score and standard deviation of F1 score for each command value (CNN model)
    for i in ['links', 'rechts','vor','zurück','start','stop','schneller','langsamer','Drehung links','Drehung rechts']:
        classification_report_average_CNN[i]['precision'] = statistics.mean( [classification_report_CNN1[i]['precision'], classification_report_CNN2[i]['precision'] , classification_report_CNN3[i]['precision']])
        classification_report_average_CNN[i]['recall'] = statistics.mean( [classification_report_CNN1[i]['recall'], classification_report_CNN2[i]['recall'] , classification_report_CNN3[i]['recall']])
        classification_report_average_CNN[i]['f1-score'] = statistics.mean( [classification_report_CNN1[i]['f1-score'], classification_report_CNN2[i]['f1-score'] , classification_report_CNN3[i]['f1-score']])
        standardabweichung_f1_CNN = statistics.stdev( [classification_report_CNN1[i]['f1-score'], classification_report_CNN2[i]['f1-score'] , classification_report_CNN3[i]['f1-score']])
        print('Standard deviation of the F1 score for the CNN model for the command word {} is {}'.format(i, standardabweichung_f1_CNN))
    
    print("\nAVERAGE CLASSIFICATION REPORT of CNN-MODEL\n")
    df_classification_report_average_CNN = pd.DataFrame(classification_report_average_CNN)  
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # Display of the average classification report in the console
        print(df_classification_report_average_CNN)
    
# Function to calculate and display the average confusion matrix of the 3-fold cross-validated models.
def average_confusionmatrix(y_validation, y_validation2, y_validation3, y_pred_1, y_pred_2, y_pred_3, model_name):

    commands = ['links', 'rechts','vor','zurück','start','stop','schneller','langsamer','Drehung links','Drehung rechts']

    # Calculation of the three individual confusion matrices
    confusion_mtx_1 = tf.math.confusion_matrix(y_validation, y_pred_1) 
    confusion_mtx_2 = tf.math.confusion_matrix(y_validation2, y_pred_2) 
    confusion_mtx_3 = tf.math.confusion_matrix(y_validation3, y_pred_3) 
    
    # Calculation of the average confusion matrix
    average_confusion_mtx = (confusion_mtx_1 + confusion_mtx_2 + confusion_mtx_3)/3
    average_confusion_mtx = np.around(average_confusion_mtx, 1)
    
    # Representation of the average confusion matrix
    fig2 = plt.figure(figsize=(12, 10))
    sns.heatmap(average_confusion_mtx, xticklabels=commands, yticklabels=commands, 
            annot=True, fmt='g')
    plt.xlabel('Prediction')
    plt.ylabel('Label')
    fig2.savefig('Durschnittliche Verwirrungsmatrix {}.pdf'.format(model_name)) # Save the figure as a .pdf file



def main():
    
    # Load the three previously split training and validation datasets from the corresponding .json files.
    # If these are not already present, the script preprocessing.py must first be started with the unziped audio files in the folder data to generate those
    
    # Loading of the three splited training and validation datasets for 3-fold cross validation.
    X_train, y_train, X_validation, y_validation = prepare_dataset(TRAINING_DATA_JSON_PATH1, VALIDATION_DATA_JSON_PATH1)
    print("Loading training and validation data set 1 finished!")
    X_train2, y_train2, X_validation2, y_validation2 = prepare_dataset(TRAINING_DATA_JSON_PATH2, VALIDATION_DATA_JSON_PATH2)
    print("Loading training and validation data set 2 finished!")
    X_train3, y_train3, X_validation3, y_validation3 = prepare_dataset(TRAINING_DATA_JSON_PATH3, VALIDATION_DATA_JSON_PATH3)
    print("Loading training and validation data set 3 finished!")
    
    # Building the (first) RNN model
    input_shape = (X_train.shape[1], X_train.shape[2])
    model_rnn1 = rnn_model(input_shape, learning_rate=LEARNING_RATE)

    # Training/Validation of the RNN model on the first training and validation dataset
    history_RNN_1 = train(model_rnn1, EPOCHS, BATCH_SIZE, X_train, y_train, X_validation, y_validation) 
    
    # Representation of accuracy and loss for the first training and validation split/dataset of the RNN model
    plot_history(history_RNN_1)
    
    # Representation of the classification report for the first training and validation split/dataset of the RNN model
    print("\nCLASSIFICATION REPORT of RNN-MODEL ON VALIDATION DATA 1!\n")
    y_pred_RNN1 = np.argmax(model_rnn1.predict(X_validation), axis=-1)
    target_names = ['links', 'rechts','vor','zurück','start','stop','schneller','langsamer','Drehung links','Drehung rechts']
    classification_report_RNN1 = classification_report(y_validation, y_pred_RNN1, target_names=target_names, output_dict = True)
    print(classification_report(y_validation, y_pred_RNN1, target_names=target_names))
    
    
    # Building the (second) RNN model
    model_rnn2 = rnn_model(input_shape, learning_rate=LEARNING_RATE)

    # Training/Validation of the RNN model on the second training and validation dataset
    history_RNN_2 = train(model_rnn2, EPOCHS, BATCH_SIZE, X_train2, y_train2, X_validation2, y_validation2) 
    
    # Representation of accuracy and loss for the second training and validation split/dataset of the RNN model
    plot_history(history_RNN_2)
    
    # Representation of the classification report for the second training and validation split/dataset of the RNN model
    print("\nCLASSIFICATION REPORT of RNN-MODEL ON VALIDATION DATA 2!\n")
    y_pred_RNN2 = np.argmax(model_rnn2.predict(X_validation2), axis=-1)
    target_names = ['links', 'rechts','vor','zurück','start','stop','schneller','langsamer','Drehung links','Drehung rechts']
    classification_report_RNN2 = classification_report(y_validation2, y_pred_RNN2, target_names=target_names, output_dict = True)
    print(classification_report(y_validation2, y_pred_RNN2, target_names=target_names))
  
    
    # Building the (third) RNN model 
    model_rnn3 = rnn_model(input_shape, learning_rate=LEARNING_RATE)

    # Training/Validation of the RNN model on the third training and validation dataset
    history_RNN_3 = train(model_rnn3, EPOCHS, BATCH_SIZE, X_train3, y_train3, X_validation3, y_validation3) 
    
    # Representation of accuracy and loss for the third training and validation split/dataset of the RNN model
    plot_history(history_RNN_3)
    
    # Representation of the classification report for the third training and validation split/dataset of the RNN model
    print("\nCLASSIFICATION REPORT of RNN-MODEL ON VALIDATION DATA 3!\n")
    y_pred_RNN3 = np.argmax(model_rnn3.predict(X_validation3), axis=-1)
    target_names = ['links', 'rechts','vor','zurück','start','stop','schneller','langsamer','Drehung links','Drehung rechts']
    classification_report_RNN3 = classification_report(y_validation3, y_pred_RNN3, target_names=target_names, output_dict = True)
    print(classification_report(y_validation3, y_pred_RNN3, target_names=target_names))
    
    
    # Summary plot of the results of the training and validation loss and accuracy with standard deviation of +-1.
    # Averaged over the 3 split training and validation data sets (K = 3) according to the 3-fold cross-validation (RNN model).
    plot_history_zusammenfassung(history_RNN_1, history_RNN_2, history_RNN_3, RNN_MODEL_NAME, EPOCHS)
    
    
    # training process of CNN model
    epochs = 25
    
    # add a dimension, because the 2D-CNN model expects a 3-dimensional input feature vector
    X_train_CNN1 = X_train[..., np.newaxis] 
    X_validation_CNN1 = X_validation[..., np.newaxis]
    
    # Building the (first) CNN model 
    input_shape = (X_train_CNN1.shape[1], X_train_CNN1.shape[2], 1)  # (# segments from samples, #  13 MfCC´s coefficients extracted, depth audidata or MFCC like greyscale cause of mono)
    model_cnn1 = cnn_model(input_shape, learning_rate=LEARNING_RATE)

    # Training/Validation of the RNN model on the first training and validation dataset 
    history_CNN_1 = train(model_cnn1, epochs, BATCH_SIZE, X_train_CNN1, y_train, X_validation_CNN1, y_validation) 
    
    
    # Representation of accuracy and loss for the first training and validation split/dataset of the RNN model over the epochs
    plot_history(history_CNN_1)

    # Representation of the classification report for the first training and validation split/dataset of the CNN model
    print("\nCLASSIFICATION REPORT of CNN-MODEL ON VALIDATION DATA 1!\n")
    y_pred_CNN1 = np.argmax(model_cnn1.predict(X_validation_CNN1), axis=-1)
    target_names = ['links', 'rechts','vor','zurück','start','stop','schneller','langsamer','Drehung links','Drehung rechts']
    classification_report_CNN1 = classification_report(y_validation, y_pred_CNN1, target_names=target_names , output_dict = True)
    print(classification_report(y_validation, y_pred_CNN1, target_names=target_names))

    
    # add a dimension, because the 2D-CNN model expects a 3-dimensional input feature vector
    X_train_CNN2 = X_train2[..., np.newaxis] 
    X_validation_CNN2 = X_validation2[..., np.newaxis]
    
    # Building the (second) CNN model 
    input_shape = (X_train_CNN2.shape[1], X_train_CNN2.shape[2], 1) # (# segments from samples, #  13 MfCC´s coefficients extracted, depth audidata or MFCC like greyscale cause of mono)
    model_cnn2 = cnn_model(input_shape, learning_rate=LEARNING_RATE)

    # Training/Validation of the CNN model on the second training and validation dataset 
    history_CNN_2 = train(model_cnn2, epochs, BATCH_SIZE, X_train_CNN2, y_train2, X_validation_CNN2, y_validation2) 
    
    
    # Representation of accuracy and loss for the second training and validation split/dataset of the RNN model over the epochs
    plot_history(history_CNN_2)
    
    # Representation of the classification report for the second training and validation split/dataset of the CNN model
    print("\nCLASSIFICATION REPORT of CNN-MODEL ON VALIDATION DATA 2!\n")
    y_pred_CNN2 = np.argmax(model_cnn2.predict(X_validation_CNN2), axis=-1)
    target_names = ['links', 'rechts','vor','zurück','start','stop','schneller','langsamer','Drehung links','Drehung rechts']
    classification_report_CNN2 = classification_report(y_validation2, y_pred_CNN2, target_names=target_names , output_dict = True)
    print(classification_report(y_validation2, y_pred_CNN2, target_names=target_names))

    # add a dimension, because the 2D-CNN model expects a 3-dimensional input feature vector
    X_train_CNN3 = X_train3[..., np.newaxis] 
    X_validation_CNN3 = X_validation3[..., np.newaxis]
    
    # Building the (third) CNN model 
    input_shape = (X_train_CNN3.shape[1], X_train_CNN3.shape[2], 1)   # (# segments from samples, #  13 MfCC´s coefficients extracted, depth audidata or MFCC like greyscale cause of mono)
    model_cnn3 = cnn_model(input_shape, learning_rate=LEARNING_RATE)

    # Training/Validation of the CNN model on the third training and validation dataset 
    history_CNN_3 = train(model_cnn3, epochs, BATCH_SIZE, X_train_CNN3, y_train3, X_validation_CNN3, y_validation3) 
    
    
    # Representation of accuracy and loss for the third training and validation split/dataset of the RNN model over the epochs
    plot_history(history_CNN_3)
    
    # Representation of the classification report for the third training and validation split/dataset of the CNN model
    print("\nCLASSIFICATION REPORT of CNN-MODEL ON VALIDATION DATA 3!\n")
    y_pred_CNN3 = np.argmax(model_cnn3.predict(X_validation_CNN3), axis=-1)
    target_names = ['links', 'rechts','vor','zurück','start','stop','schneller','langsamer','Drehung links','Drehung rechts']
    classification_report_CNN3 = classification_report(y_validation3, y_pred_CNN3, target_names=target_names , output_dict = True)
    print(classification_report(y_validation3, y_pred_CNN3, target_names=target_names))

    
    
    # Summary plot of the results of the training and validation loss and accuracy with standard deviation of +-1.
    # Averaged over the 3 split training and validation data sets (K = 3) according to the 3-fold cross-validation (CNN model).
    plot_history_zusammenfassung(history_CNN_1, history_CNN_2, history_CNN_3, CNN_MODEL_NAME, epochs)
    
    
    # Calculation and display of the average classification report of both models
    classification_report_models_average(classification_report_RNN1, classification_report_RNN2, classification_report_RNN3,
                                         classification_report_CNN1, classification_report_CNN2, classification_report_CNN3)
  
    
    # Validation and evaluation of the individual models on the corresponding validation data sets.
    evaluation_RNN1    = model_rnn1.evaluate(X_validation, y_validation) 
    evaluation_RNN2    = model_rnn2.evaluate(X_validation2, y_validation2) 
    evaluation_RNN3    = model_rnn3.evaluate(X_validation3, y_validation3) 
    
    evaluation_CNN1    = model_cnn1.evaluate(X_validation_CNN1, y_validation)
    evaluation_CNN2    = model_cnn2.evaluate(X_validation_CNN2, y_validation2)
    evaluation_CNN3    = model_cnn3.evaluate(X_validation_CNN3, y_validation3)
    
    # Summary of the test results on the validation dtaen in a structure
    evaluation_dict  = {'Model' : ['RNN-Model', 'RNN-Model','RNN-Model','CNN-Model', 'CNN-Model','CNN-Model'],
                    'Test_Accuracy' : [evaluation_RNN1[1], evaluation_RNN2[1], evaluation_RNN3[1], 
                                   evaluation_CNN1[1], evaluation_CNN2[1], evaluation_CNN3[1]],
                    'Test_Loss' : [evaluation_RNN1[0], evaluation_RNN2[0], evaluation_RNN3[0], 
                                   evaluation_CNN1[0], evaluation_CNN2[0], evaluation_CNN3[0]]}
    
    # Boxplot of accuracy and loss with respect to test results on validation data comparing RNN-LSTM and CNN model.
    box_plot(evaluation_dict)
    

    
    RNN_Test_Accuracy_array = [evaluation_RNN1[1], evaluation_RNN2[1], evaluation_RNN3[1]]
    CNN_Test_Accuracy_array = [evaluation_CNN1[1], evaluation_CNN2[1], evaluation_CNN3[1]]
    
    
    
    
    # 2-sample t-test to compare two samples and their means
    # significance level of 5% i.e. alpha = 0.05 
    # hypothesis : the two average accuracy values of the models are not equal and are significantly different
    # null hypothesis : the two average accuracy values of the models are equal
    # Null hypothesis that both average accuracy values cannot be rejected here if p-value of t-test is above 0.05 
    # Only if the p-value is below 0.05, null hypothesis can be rejected
    
    # Before performing the t-test, one must decide if we assume, that the two populations have equal variances or not 
    # Rule of thumb : If ratio of larger sample variance to smaller sample variance is more than 4 then variances/standard deviations are not equal and one performs Welch test
    # In Welch test, population variances are not assumed to be equal


    varrianz_array = [np.var(np.array(RNN_Test_Accuracy_array)),np.var(CNN_Test_Accuracy_array)]
    if max(varrianz_array)/min(varrianz_array) > 4 :
        same_var = False 
    else:
        same_var = True
    
    # If same_var = False Welch test is performed
    RNN_vs_CNN_ttest_result = stats.ttest_ind(RNN_Test_Accuracy_array , CNN_Test_Accuracy_array, equal_var = same_var)
    
    print("\n p-value of the t-Test related to the accuracy value of both models: '{}'".format(RNN_vs_CNN_ttest_result[1])) 
    
    if (RNN_vs_CNN_ttest_result[1] < 0.05):
        print("\n Difference in average accuracy values of CNN and RNN-LSTM model is given from t-Test analysis!") 
    else: 
        print("\n It can't be said that difference of the average accuracy values of CNN and RNN-LSTM model is given from t-Test analysis!") 
        
        
    # Calculation and display of the average confusion marix from the RNN-LSTM Model.
    average_confusionmatrix(y_validation, y_validation2, y_validation3, y_pred_RNN1, y_pred_RNN2, y_pred_RNN3, RNN_MODEL_NAME) 
    
    # Calculation and display of the average confusion marix from the CNN Model.
    average_confusionmatrix(y_validation, y_validation2, y_validation3, y_pred_CNN1, y_pred_CNN2, y_pred_CNN3, CNN_MODEL_NAME) 
    

if __name__ == "__main__":
    main()