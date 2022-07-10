## Automatic Speech Recognition of command words with RNN-LSTM and CNN 

### Project overview

* Realized a Automatic Speech Recognition system (ASR/NLP) to classificate command words with custom modeled RNN-LSTM and CNN
* Implementation in Python using Tensorflow, Keras and Librosa 
* Quantitative comparison and evaluation of both neural networks by cross validation, average evaluation metric scores  
  (Precision, Recall, F1-Score, Accuracy)  confusion matrix and statistical t-test analysis
* Achieved an accuracy of over 91.9% on average

### Objective:
Speech recognition and classification of 10 command words ((1) links (2) rechts (3) vor (4) zurück (5) start (6) stop (7) schneller (8) langsamer
(9) drehung links (10) drehung rechts) using Long Short Term Memory(LSTM) and Convolutional Neural Network (CNN). The system could 
be used on a higher level for the control a robot via voice. 

### Part 1: Data acquisition
- Created a custom audio dataset by recording the 10 words from a total of 9 male and 9 female participants, with each command word recorded 30 times by the participants. (5400 audio files in total)
- Each audio of command word is saved in a separate WAV file (mono, 16-bit resolution, duration: 2 seconds)
- Script for automated recording:[AudioRecorder.py](https://github.com/nickjust/ASR_command_words/blob/main/AudioRecorder.py) 

### Part 2: Preprocessing 
- For the evaluation and validation of the two neural networks, a 3-fold speaker-independent cross-validation was chosen. For this purpose, the data set had to be divided into 3 equal parts. In order to develop a speaker-independent system, it was ensured that a speaker did not occur simultaneously in the training and validation data.

<p align="center">
  <img src="images_readme/cross_validation_fig.PNG" width="550"/>
</p>

- Extraction of MFCC features (conversion of the audio signals into a time-frequency plane) for later classification using the library librosa, since the raw audio waveform is general not directly used as an input vector for the machine/deep learning algorithms in the speech recognition domain. Illustration of the MFCC coefficients for an audio file of the used dataset (word: vor):

<p align="center">
  <img src="images_readme/MFCC_visualization.png" width="550"/>
</p>




- see script for detailed information 
