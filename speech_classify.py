import numpy as np
import pandas as pd
import os
import librosa
from tqdm import tqdm
import keras
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import itertools
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.utils import np_utils
from sklearn import metrics 
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split 
from keras.callbacks import ModelCheckpoint 
from datetime import datetime 
from pathlib import Path
import nlpaug.augmenter.audio as naa
#esp 8266


container_path = './Local/'
dataset ='./Ravdess/' 
metadata = pd.read_csv('./ravdess_dataset.csv')

max_pad_len = num_rows = 0
num_columns = 100
num_channels = 1
filter_size = 2
num_epochs = 500
num_batch_size = 32

model = Sequential()
class_names = []


def extract_features(file_name):
   
    try:
        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=num_columns)
        
        original_shape = mfccs.shape[1]
        
        pad_width = max_pad_len - mfccs.shape[1]
        
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        
    except Exception as e:
        print("Error: ", e)
        # print("Error encountered while parsing file: ", file_name)
        return None 
     
    return mfccs, original_shape


def extract_features_from_noisy_file(file_name):
   
    try:
        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=num_columns)
        pad_width = max_pad_len - mfccs.shape[1]
        
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        
        aug = naa.NoiseAug(noise_factor=0.01)         
        augmented_data = aug.augment(audio)
        
        mfccs2 = librosa.feature.mfcc(y=augmented_data, sr=sample_rate, n_mfcc=num_columns)
        pad_width = max_pad_len - mfccs2.shape[1]
        
        mfccs2 = np.pad(mfccs2, pad_width=((0, 0), (0, pad_width)), mode='constant')
        
    except Exception as e:
        print("Error: ", e)
        # print("Error encountered while parsing file: ", file_name)
        return None 
     
    return mfccs, mfccs2



# Iterate through each sound file and extract the features 
def load_files_from_folders():
    features = []
    global max_pad_len, num_rows
    max_pad_len = num_rows = 3200
    
    file_dir = Path(container_path)
    folders = [directory for directory in file_dir.iterdir() if directory.is_dir()]
    categories = [fo.name for fo in folders]
    
    for i, direc in (enumerate(folders)):
        for file in tqdm(direc.iterdir()):  
            
            data1, data2 = extract_features_from_noisy_file(file)
            features.append([data1, categories[i]])
            features.append([data2, categories[i]])
    
    return features


def load_files_from_CSV():
    features = []
    global max_pad_len, num_rows
    max_pad_len = num_rows = 196
    
    for index, row in tqdm(metadata.iterrows()):
        
        file_name = os.path.join(os.path.abspath(dataset),str(row["fname"]))
        
        class_label = row["label"]
        
        data1, data2 = extract_features_from_noisy_file(file_name)
        features.append([data1, class_label])
        features.append([data2, class_label])
    
    return features



def construct_model():
    # Construct model     
#    model = Sequential()
#    
    model.add(Conv2D(filters=16, kernel_size=3, padding='same', input_shape= (num_columns, num_rows, num_channels), activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.2))
    
    model.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.2))
    
    model.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.2))
    
    model.add(Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.2))
    
    model.add(GlobalAveragePooling2D())
    model.add(Dense(num_labels, activation='softmax'))
    

def compile_model():
    # Compile the model
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
    

def model_summary():
    # Display model architecture summary 
    model.summary()
    
    # Calculate pre-training accuracy 
    score = model.evaluate(x_test, y_test, verbose=1)
    accuracy = 100*score[1]
    
    print("Pre-training accuracy: %.4f%%" % accuracy)


def train_model():
    # train the model
    start = datetime.now()
    checkpointer = ModelCheckpoint(filepath='./model/speech_recognition_ravdess_audio_song_aug_2.h5', 
                                   verbose=1, save_best_only=True)
    
    hist = model.fit(x_train, y_train, batch_size=num_batch_size, epochs=num_epochs, validation_data=(x_test, y_test), callbacks=[checkpointer], verbose=1)
    
    duration = datetime.now() - start
    print("Training completed in time: ", duration)
    
    return hist
    

def evaluate_model():
    # Evaluating the model on the training and testing set
    score = model.evaluate(x_train, y_train, verbose=0)
    print("Training Accuracy: ", score[1]*100)
    
    score = model.evaluate(x_test, y_test, verbose=0)
    print("Validation Accuracy: ", score[1]*100)


def comparison_plot(hist):
    # comparison plot
    train_loss = hist.history['loss']
    val_loss = hist.history['val_loss']
    train_acc = hist.history['accuracy']
    val_acc = hist.history['val_accuracy']
    xc = range(num_epochs)
    
    plt.figure(1, figsize = (7, 5))
    plt.plot(xc, train_acc)
    plt.plot(xc, val_acc)
    plt.xlabel('Number of epochs')
    plt.ylabel('accuracy')
    plt.title("train acc vs val acc")
    plt.grid(True)
    plt.legend(['train', 'val'], loc=4)
    plt.style.use(['classic'])
    
    plt.figure(2, figsize = (7, 5))
    plt.plot(xc, train_loss)
    plt.plot(xc, val_loss)
    plt.xlabel('Number of epochs')
    plt.ylabel('loss')
    plt.title("train loss vs val loss")
    plt.grid(True)
    plt.legend(['train', 'val'])
    plt.style.use(['classic'])


def trained_model_evaluation():
    # model evaluation
    score = model.evaluate(x_test, y_test, verbose=0)
    print("Accuracy: ", score[1]*100)
    
#    print("Test score: ", score[0])
#    print(model.predict_classes(x_test[1:5]))
#    print(y_test[1:5])
        
    
def confussion_matrix():
    # confussion matrix
#    predictions = model.predict(x_test, batch_size=num_batch_size, verbose=0)
    class_predictions = model.predict_classes(x_test, batch_size=num_batch_size, verbose=0)
    
    print(classification_report(np.argmax(y_test, axis=1), class_predictions, target_names = class_names))
    
    cm = confusion_matrix(np.argmax(y_test, axis=1), class_predictions)
#    for i in range(0, len(cm)):
#      print(cm[i])
      
    
    plt.figure(figsize=(6,6))
    plt.imshow(cm, interpolation='nearest', cmap = plt.cm.Blues)
    plt.title("Confussion Matrix")
    # plt.colorbar()
    
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation = 45)
    # plt.yticks(tick_marks, class_names)
    
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
      plt.text(j, i, cm[i, j], 
               horizontalalignment = 'center',
               color = 'white' if i==j else 'black')
      
    
    plt.tight_layout()
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    
    


if __name__ == '__main__':
    
    # train RAVDESS dataset
    features = load_files_from_CSV()
    
    # train local dataset
#    features = load_files_from_folders()
    
    # Convert into a Panda dataframe 
    featuresdf = pd.DataFrame(features, columns=['feature','class_label'])
    
    
    print('Finished feature extraction from ', len(featuresdf), ' files')
        
    
    # Convert features and corresponding classification labels into numpy arrays
    X = np.array(featuresdf.feature.tolist())
    y = np.array(featuresdf.class_label.tolist())
    
    
    # Encode the classification labels
    le = LabelEncoder()
    yy = to_categorical(le.fit_transform(y)) 
    num_labels = yy.shape[1]
    
    
#    # split the dataset 
    train_data_set, x_test, train_label_set, y_test = train_test_split(X, yy, test_size=0.1, random_state = 42)
    x_train, x_val, y_train, y_val = train_test_split(train_data_set, train_label_set, test_size=0.2225, random_state = 42)
    
    x_train = x_train.reshape(x_train.shape[0], num_columns, num_rows, num_channels)
    x_test = x_test.reshape(x_test.shape[0], num_columns, num_rows, num_channels)
    class_names = np.unique(y) 
    
    
    construct_model()
    compile_model()
    model_summary()
    hist = train_model()
    evaluate_model()
    comparison_plot(hist)
    confussion_matrix()
    trained_model_evaluation()
#    
    




    