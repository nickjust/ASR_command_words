import librosa
import librosa.display
import os
import json
import shutil



# path definitions for 3-fold crossvalidation 
VALIDATION_DATASET_PATH1 = "dataset_validation1"
TRAINING_DATASET_PATH1 = "dataset_training1"
VALIDATION_DATASET_PATH2 = "dataset_validation2"
TRAINING_DATASET_PATH2 = "dataset_training2"
VALIDATION_DATASET_PATH3 = "dataset_validation3"
TRAINING_DATASET_PATH3 = "dataset_training3"
DATA_PATH = "data/"
DATA_JUST_PATH = "data_just"
VALIDATION_DATA_JSON_PATH1 = "data_validation1.json"
TRAINING_DATA_JSON_PATH1 = "data_training1.json"
VALIDATION_DATA_JSON_PATH2 = "data_validation2.json"
TRAINING_DATA_JSON_PATH2 = "data_training2.json"
VALIDATION_DATA_JSON_PATH3 = "data_validation3.json"
TRAINING_DATA_JSON_PATH3 = "data_training3.json"


# extraction of the MFCC coefficients of the training and validation datasets and storage in json format for later use
def data_preprocessing(dataset_path, json_path, num_mfcc=13, n_fft=2048, hop_length=512):
    
    
    # Definition of data dictionary to store MFCC coefficients with matching labels and the corresponding name of the audio file
    data = {
        "mapping": [],
        "labels": [],
        "MFCCs": [],
        "files": []
    }


    # Loop over all subfolders
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path, topdown= True)):

        # Ensure  the subfolder level
        if dirpath is not dataset_path:

            label = dirpath.split("/")[-1]  # saving the label
            data["mapping"].append(label) 
            print("\nVerarbeitung: '{}'".format(label))

            # Processing of all audio files in a subfolder via the loop and storage of the MFCC coefficients
            for f in filenames:
                file_path = os.path.join(dirpath, f)


                signal, sample_rate = librosa.load(file_path, duration=2.0) # loading of the audiofiles (2 seconds length) with librosa
                
                # audio files, that are too short, are caught here (limit to 1.95s which corresponds to 43000 samples)
                if len(signal) >= 43000:
                    
                    # length is unified to 1.95s (43000 samples) so that dimensions of later MFFC features are consistent across all audio files and can be used as input 
                    signal = signal[:43000] 
                    
                    # Computing of the MFCC coefficients
                    MFCCs = librosa.feature.mfcc(y=signal, sr= sample_rate, n_mfcc=num_mfcc)
                
                    # Visualization of the extracted MFCC coefficients for the respective audio file
                    #  import matplotlib.pyplot as plt
                    #  plt.figure(figsize=(10, 4))
                    #  librosa.display.specshow(MFCCs, x_axis='time')
                    #  plt.colorbar()
                    #  plt.title('MFCC')
                    #  plt.tight_layout()
                    #  plt.show()
    
                    # Storage of the MFCC coefficients, the labels, and the name of the respective audio file in the data dictionary
                    data["MFCCs"].append(MFCCs.T.tolist())
                    data["labels"].append(i-1)
                    data["files"].append(file_path)
                    print("{}: {}".format(file_path, i-1))

    # Saving the data dictionary in json format
    json.dump(data, open(json_path, "w"), indent=4)
    
    # Deleting the temporary validation or training folders
    try:
        shutil.rmtree(dataset_path)
    except OSError as e:
        print(e)
    else:
        print("Temporary folder '{}' successfully deleted".format(dataset_path))

# Speaker-independent partitioning of the total data set into validation and training data according to a 3-fold cross-validation procedure
def split_data(data_path, training_dataset_path, validation_dataset_path):
     

    # Creation of a (temporary) folder for the respective training data set of the distribution if this does not exist
    # as well as subfolders for the respective labels or classes 
    if not os.path.exists(training_dataset_path):
        os.makedirs(training_dataset_path)
        os.makedirs(os.path.join(training_dataset_path, "0_links"))
        os.makedirs(os.path.join(training_dataset_path, "1_rechts"))
        os.makedirs(os.path.join(training_dataset_path, "2_vor"))
        os.makedirs(os.path.join(training_dataset_path, "3_zurueck"))
        os.makedirs(os.path.join(training_dataset_path, "4_start"))
        os.makedirs(os.path.join(training_dataset_path, "5_stop"))
        os.makedirs(os.path.join(training_dataset_path, "6_schneller"))
        os.makedirs(os.path.join(training_dataset_path, "7_langsamer"))
        os.makedirs(os.path.join(training_dataset_path, "8_drehung_links"))
        os.makedirs(os.path.join(training_dataset_path, "9_drehung_rechts"))
        
    # Creation of a (temporary) folder for the respective validation data set of the distribution if this does not exist
    # as well as subfolders for the respective labels or classes  
    if not os.path.exists(validation_dataset_path):
        os.makedirs(validation_dataset_path)
        os.makedirs(os.path.join(validation_dataset_path, "0_links"))
        os.makedirs(os.path.join(validation_dataset_path, "1_rechts"))
        os.makedirs(os.path.join(validation_dataset_path, "2_vor"))
        os.makedirs(os.path.join(validation_dataset_path, "3_zurueck"))
        os.makedirs(os.path.join(validation_dataset_path, "4_start"))
        os.makedirs(os.path.join(validation_dataset_path, "5_stop"))
        os.makedirs(os.path.join(validation_dataset_path, "6_schneller"))
        os.makedirs(os.path.join(validation_dataset_path, "7_langsamer"))
        os.makedirs(os.path.join(validation_dataset_path, "8_drehung_links"))
        os.makedirs(os.path.join(validation_dataset_path, "9_drehung_rechts"))
    
    # Loop over all subfolders
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(data_path, topdown= True)):

        # Ensure  the subfolder level
        if dirpath is not data_path:

            
            # split: First 6 of the 18 spoken person data used for validation (33%) rest as training data (66%) for later speaker-independent 3-fold cross-validation

            split_data_val_train = 6
            
            
            # First 6 of the 18 data sets for validation (33%), rest as training data for the first split
            if training_dataset_path == "dataset_training1":
                if i > split_data_val_train:
                    links_path = os.path.join(training_dataset_path, "0_links")
                    rechts_path = os.path.join(training_dataset_path, "1_rechts")
                    vor_path = os.path.join(training_dataset_path, "2_vor")
                    zurueck_path = os.path.join(training_dataset_path, "3_zurueck")
                    start_path = os.path.join(training_dataset_path, "4_start")
                    stop_path = os.path.join(training_dataset_path, "5_stop")
                    schneller_path = os.path.join(training_dataset_path, "6_schneller")
                    langsamer_path = os.path.join(training_dataset_path, "7_langsamer")
                    drehung_links_path = os.path.join(training_dataset_path, "8_drehung_links")
                    drehung_rechts = os.path.join(training_dataset_path, "9_drehung_rechts")
                else:
                    links_path = os.path.join(validation_dataset_path, "0_links")
                    rechts_path = os.path.join(validation_dataset_path, "1_rechts")
                    vor_path = os.path.join(validation_dataset_path, "2_vor")
                    zurueck_path = os.path.join(validation_dataset_path, "3_zurueck")
                    start_path = os.path.join(validation_dataset_path, "4_start")
                    stop_path = os.path.join(validation_dataset_path, "5_stop")
                    schneller_path = os.path.join(validation_dataset_path, "6_schneller")
                    langsamer_path = os.path.join(validation_dataset_path, "7_langsamer")
                    drehung_links_path = os.path.join(validation_dataset_path, "8_drehung_links")
                    drehung_rechts = os.path.join(validation_dataset_path, "9_drehung_rechts")
            
            # First 6 and last 6 persons as training data, remaining persons as validation data for the second split
            elif  training_dataset_path == "dataset_training2": 
                if i <= split_data_val_train or i > 18-split_data_val_train :
                    links_path = os.path.join(training_dataset_path, "0_links")
                    rechts_path = os.path.join(training_dataset_path, "1_rechts")
                    vor_path = os.path.join(training_dataset_path, "2_vor")
                    zurueck_path = os.path.join(training_dataset_path, "3_zurueck")
                    start_path = os.path.join(training_dataset_path, "4_start")
                    stop_path = os.path.join(training_dataset_path, "5_stop")
                    schneller_path = os.path.join(training_dataset_path, "6_schneller")
                    langsamer_path = os.path.join(training_dataset_path, "7_langsamer")
                    drehung_links_path = os.path.join(training_dataset_path, "8_drehung_links")
                    drehung_rechts = os.path.join(training_dataset_path, "9_drehung_rechts")
                else:
                    links_path = os.path.join(validation_dataset_path, "0_links")
                    rechts_path = os.path.join(validation_dataset_path, "1_rechts")
                    vor_path = os.path.join(validation_dataset_path, "2_vor")
                    zurueck_path = os.path.join(validation_dataset_path, "3_zurueck")
                    start_path = os.path.join(validation_dataset_path, "4_start")
                    stop_path = os.path.join(validation_dataset_path, "5_stop")
                    schneller_path = os.path.join(validation_dataset_path, "6_schneller")
                    langsamer_path = os.path.join(validation_dataset_path, "7_langsamer")
                    drehung_links_path = os.path.join(validation_dataset_path, "8_drehung_links")
                    drehung_rechts = os.path.join(validation_dataset_path, "9_drehung_rechts")
                    
            # Last 6 persons of the 18 records for validation (33%), rest as training data for the third split      
            elif  training_dataset_path == "dataset_training3":  
                if i <= 18-split_data_val_train:
                    links_path = os.path.join(training_dataset_path, "0_links")
                    rechts_path = os.path.join(training_dataset_path, "1_rechts")
                    vor_path = os.path.join(training_dataset_path, "2_vor")
                    zurueck_path = os.path.join(training_dataset_path, "3_zurueck")
                    start_path = os.path.join(training_dataset_path, "4_start")
                    stop_path = os.path.join(training_dataset_path, "5_stop")
                    schneller_path = os.path.join(training_dataset_path, "6_schneller")
                    langsamer_path = os.path.join(training_dataset_path, "7_langsamer")
                    drehung_links_path = os.path.join(training_dataset_path, "8_drehung_links")
                    drehung_rechts = os.path.join(training_dataset_path, "9_drehung_rechts")
                else:
                    links_path = os.path.join(validation_dataset_path, "0_links")
                    rechts_path = os.path.join(validation_dataset_path, "1_rechts")
                    vor_path = os.path.join(validation_dataset_path, "2_vor")
                    zurueck_path = os.path.join(validation_dataset_path, "3_zurueck")
                    start_path = os.path.join(validation_dataset_path, "4_start")
                    stop_path = os.path.join(validation_dataset_path, "5_stop")
                    schneller_path = os.path.join(validation_dataset_path, "6_schneller")
                    langsamer_path = os.path.join(validation_dataset_path, "7_langsamer")
                    drehung_links_path = os.path.join(validation_dataset_path, "8_drehung_links")
                    drehung_rechts = os.path.join(validation_dataset_path, "9_drehung_rechts")
            else:
                print("Fehler in Datenbezeichnung")   
                
            # Loop over all audio data in a person subfolder
            for f in filenames:
                file_path = os.path.join(dirpath, f)
                
                if 'b1' in f and 'b10' not in f:

                    # Build the path and copy the files according to the distribution in the training or validation data folder and the class subfolders (command word 1).
                    
                    kopie_path = os.path.join(links_path, f)
                    shutil.copy(file_path, kopie_path)
                
                elif 'b2' in f:
                    
                    # Build the path and copy the files according to the distribution in the training or validation data folder and the class subfolders (command word 2).
                    
                    kopie_path = os.path.join(rechts_path, f)
                    shutil.copy(file_path, kopie_path)
                    
                elif 'b3' in f:

                    # Build the path and copy the files according to the distribution in the training or validation data folder and the class subfolders (command word 3).
                    
                    kopie_path = os.path.join(vor_path, f)
                    shutil.copy(file_path, kopie_path)
                        
                elif 'b4' in f:
                    
                    # Build the path and copy the files according to the distribution in the training or validation data folder and the class subfolders (command word 4).
                    
                    kopie_path = os.path.join(zurueck_path, f)
                    shutil.copy(file_path, kopie_path)
                    
                elif 'b5' in f:
                    
                    # Build the path and copy the files according to the distribution in the training or validation data folder and the class subfolders (command word 5).
                    
                    kopie_path = os.path.join(start_path, f)
                    shutil.copy(file_path, kopie_path)
                    
                elif 'b6' in f:
                    
                    # Build the path and copy the files according to the distribution in the training or validation data folder and the class subfolders (command word 6).
                    
                    kopie_path = os.path.join(stop_path, f)
                    shutil.copy(file_path, kopie_path)
                    
                elif 'b7' in f:
                    
                    # Build the path and copy the files according to the distribution in the training or validation data folder and the class subfolders (command word 7).
                    
                    kopie_path = os.path.join(schneller_path, f)
                    shutil.copy(file_path, kopie_path)
                    
                elif 'b8' in f:
                    
                    # Build the path and copy the files according to the distribution in the training or validation data folder and the class subfolders (command word 8).
                    
                    kopie_path = os.path.join(langsamer_path, f)
                    shutil.copy(file_path, kopie_path)
                    
                elif 'b9' in f:
                    
                    # Build the path and copy the files according to the distribution in the training or validation data folder and the class subfolders (command word 9).
                    
                    kopie_path = os.path.join(drehung_links_path, f)
                    shutil.copy(file_path, kopie_path)
                    
                elif 'b10' in f:

                    # Build the path and copy the files according to the distribution in the training or validation data folder and the class subfolders (command word 10).
                    kopie_path = os.path.join(drehung_rechts, f)
                    shutil.copy(file_path, kopie_path)
                    
                else:
                   print("Fehler in Datenbezeichnung")    
        
def sort_data(data_path):
   
    # creation of temorary folder daten_just (if not existing) to split data into folders according to person identifier
    if not os.path.exists(DATA_JUST_PATH):
        os.makedirs(DATA_JUST_PATH)
    
    # loop over all existing files in the data folder 
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(data_path, topdown= True)):

            for f in filenames:
                print("\nSortierung des files in temporären Ordner daten_just: '{}'".format(os.path.join(data_path, f)))
                new_path = os.path.join(DATA_JUST_PATH, f[:f.find("_")])
                
                # Creation of a temporary subfolder in daten_just according to the person identifier.
                if not os.path.exists(new_path):
                    os.makedirs(new_path)
                    
                # Copy the audio to the rft subfolder in data_just
                file_path = os.path.join(data_path, f)
                shutil.copy(file_path, new_path)
                
                
                
# If there are no split training and validation datasets in .json format, run this script first
if __name__ == "__main__":
    
    

     # First sorting of audio recordings in the data folder  into a temoporary subfolder data_just
    sort_data(DATA_PATH)
    
    # Sprecherunabhängige Aufteilung der Datensätze für die spätere 3-fache Kreuzvalidierung
    split_data(DATA_JUST_PATH, TRAINING_DATASET_PATH1, VALIDATION_DATASET_PATH1)
    split_data(DATA_JUST_PATH ,TRAINING_DATASET_PATH2, VALIDATION_DATASET_PATH2)
    split_data(DATA_JUST_PATH ,TRAINING_DATASET_PATH3, VALIDATION_DATASET_PATH3)

    # Extraction and saving of MFCC coefficients, label, and audio file name as .json file for later usage
    data_preprocessing(TRAINING_DATASET_PATH1, TRAINING_DATA_JSON_PATH1)
    data_preprocessing(VALIDATION_DATASET_PATH1, VALIDATION_DATA_JSON_PATH1)
    data_preprocessing(TRAINING_DATASET_PATH2, TRAINING_DATA_JSON_PATH2)
    data_preprocessing(VALIDATION_DATASET_PATH2, VALIDATION_DATA_JSON_PATH2)
    data_preprocessing(TRAINING_DATASET_PATH3, TRAINING_DATA_JSON_PATH3)
    data_preprocessing(VALIDATION_DATASET_PATH3, VALIDATION_DATA_JSON_PATH3)
    
    # Delete the folder daten_just (only temporarily necessary)
    try:
        shutil.rmtree(DATA_JUST_PATH)
    except OSError as e:
        print(e)
    else:
        print("Temporary folder '{}' successfully deleted".format(DATA_JUST_PATH))
    
    
    
