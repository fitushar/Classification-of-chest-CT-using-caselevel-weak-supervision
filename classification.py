#######################Loading the libararies ####################################
import numpy as np
import os
import cv2
import tensorflow as tf
########GPU Configuration#############
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from keras import backend as K
config=tf.ConfigProto()
config.gpu_options.allow_growth=True
config.gpu_options.per_process_gpu_memory_fraction=0.98
K.tensorflow_backend.set_session(tf.Session(config=config))
######Keras Utilities #################
from keras.models import Sequential, Model
from keras.layers import Conv2D, ZeroPadding2D, MaxPooling2D, GlobalMaxPooling2D, Input
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras import applications
from keras.utils import np_utils
from time import time



from sklearn.metrics import roc_auc_score,roc_curve
from sklearn.metrics import roc_curve
#import scikitplot as skplt
import matplotlib.pyplot as plt
import math

###Keras
import keras
from keras.utils import np_utils
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from keras.layers import Dense, Activation, Flatten
from keras.layers import merge, Input
from keras.layers import MaxPooling2D
from keras.models import Model
from sklearn.utils import shuffle
#from sklearn.cross_validation import train_test_split
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard

###Sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.externals import joblib
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.externals import joblib

import pandas as pd
from sklearn import metrics
from scipy.stats import zscore
from sklearn.model_selection import KFold
from keras.layers import GlobalMaxPooling2D
from keras import optimizers

from keras.utils import np_utils
from keras.callbacks import Callback, EarlyStopping
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from keras.metrics import categorical_accuracy

############################### User Input #################################
Number_of_folds=4
batch_size=32
epochs=50
Disease_name='Resnet50_Edena_Binary_CL200withAgumentation2'
num_classes=2
training_fold_1_name='/Local/cl2019/mayClassification/5_16_2019_BinaryEdemaclassclassification_hu200/excels/PNGLung_CV-Training-Fold-1.csv'
validation_fold_1_name='/Local/cl2019/mayClassification/5_16_2019_BinaryEdemaclassclassification_hu200/excels/PNGLung_CV-Validation-Fold-1.csv'
training_fold_2_name='/Local/cl2019/mayClassification/5_16_2019_BinaryEdemaclassclassification_hu200/excels/PNGLung_CV-Training-Fold-2.csv'
validation_fold_2_name='/Local/cl2019/mayClassification/5_16_2019_BinaryEdemaclassclassification_hu200/excels/PNGLung_CV-Validation-Fold-2.csv'
training_fold_3_name='/Local/cl2019/mayClassification/5_16_2019_BinaryEdemaclassclassification_hu200/excels/PNGLung_CV-Training-Fold-3.csv'
validation_fold_3_name='/Local/cl2019/mayClassification/5_16_2019_BinaryEdemaclassclassification_hu200/excels/PNGLung_CV-Validation-Fold-3.csv'
training_fold_4_name='/Local/cl2019/mayClassification/5_16_2019_BinaryEdemaclassclassification_hu200/excels/PNGLung_CV-Training-Fold-4.csv'
validation_fold_4_name='/Local/cl2019/mayClassification/5_16_2019_BinaryEdemaclassclassification_hu200/excels/PNGLung_CV-Validation-Fold-4.csv'
########################################################################
####################### Coder Info######################################
print('Code is written by @Fakrul Islam Tushar,4/10/2019')
print('Dept. of Radiology, RAI LABS, Duke University')
print('For any confusion Mail: ft42@duke.edu\n')
print('Data is loaded from the Folder')
########################################################################

###############Functions######################

#This function Compute Run time
def auc_roc(y_true, y_pred):
    # any tensorflow metric
    value, update_op = tf.contrib.metrics.streaming_auc(y_pred, y_true)

    # find all variables created for this metric
    metric_vars = [i for i in tf.local_variables() if 'auc_roc' in i.name.split('/')[1]]

    # Add metric variables to GLOBAL_VARIABLES collection.
    # They will be initialized for new session.
    for v in metric_vars:
        tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, v)

    # force to update metric values
    with tf.control_dependencies([update_op]):
        value = tf.identity(value)
        return value

def f1_score(y_true, y_pred):
    # any tensorflow metric
    value, update_op = tf.contrib.metrics.f1_score(y_true,y_pred)

    # find all variables created for this metric
    metric_vars = [i for i in tf.local_variables() if 'f1_score' in i.name.split('/')[1]]

    # Add metric variables to GLOBAL_VARIABLES collection.
    # They will be initialized for new session.
    for v in metric_vars:
        tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, v)

    # force to update metric values
    with tf.control_dependencies([update_op]):
        value = tf.identity(value)
        return value



def get_model(net,num_classes=2):

    if net=='Resnet50':
            image_input=Input(shape=(224,224,3))
            model=keras.applications.resnet50.ResNet50(input_tensor=image_input,include_top=False,weights='imagenet')
            last_layer = model.output
            z= keras.layers.GlobalMaxPooling2D()(last_layer)
            out = Dense(num_classes, activation='softmax', name='output_layer')(z)
            custom_model = Model(inputs=image_input,outputs= out)

            #Freezing the upper layers
            #for layer in custom_model.layers[:-2]:
                #layer.trainable=False

            #sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
            custom_model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy',auc_roc,f1_score])
            return custom_model

    elif net=='Vgg16':
            image_input=Input(shape=(224,224,3))
            model=keras.applications.vgg16.VGG16(input_tensor=image_input,include_top=False,weights='imagenet')
            last_layer = model.output
            z= keras.layers.GlobalMaxPooling2D()(last_layer)
            out = Dense(num_classes, activation='softmax', name='output_layer')(z)
            custom_model = Model(inputs=image_input,outputs= out)

            custom_model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy',auc_roc])
            return custom_model

    elif net=='InceptionV3':
            image_input=Input(shape=(224,224,3))
            model=keras.applications.inception_v3.InceptionV3(input_tensor=image_input,include_top=False,weights='imagenet')
            last_layer = model.output
            z= keras.layers.GlobalMaxPooling2D()(last_layer)
            out = Dense(num_classes, activation='softmax', name='output_layer')(z)
            custom_model = Model(inputs=image_input,outputs= out)

            #Freezing the upper layers
            #for layer in custom_model.layers[:-1]:
                #layer.trainable=False

            custom_model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['categorical_accuracy',auc_roc])
            return custom_model

    elif net=='DenseNet':
            image_input=Input(shape=(224,224,3))
            model=keras.applications.densenet.DenseNet121(input_tensor=image_input,include_top=False,weights='imagenet')
            last_layer = model.output
            z= keras.layers.GlobalMaxPooling2D()(last_layer)
            out = Dense(num_classes, activation='sigmoid', name='output_layer')(z)
            custom_model = Model(inputs=image_input,outputs= out)

            #Freezing the upper layers
            for layer in custom_model.layers[:-2]:
                layer.trainable=False

            custom_model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['categorical_accuracy','acc'])
            return custom_model



##########################CrossValidation############################
#This part will make the cross-validation splits and
#automatically save the Cross-validated Datas name,
#labels and Path to csv file

img_data_list=[]
Sample_Count_list=[] #How many samples in each folder
Image_name=[] # name of the images
Image_Path=[] # Path of the images

Save_Training_folds_x=[]
Save_Training_folds_y=[]
Save_Validation_folds_x=[]
Save_Validation_folds_y=[]


f1_train=pd.read_csv(training_fold_1_name,dtype=object,keep_default_na=False,na_values=[])
f2_train=pd.read_csv(training_fold_2_name,dtype=object,keep_default_na=False,na_values=[])
f3_train=pd.read_csv(training_fold_3_name,dtype=object,keep_default_na=False,na_values=[])
f4_train=pd.read_csv(training_fold_4_name,dtype=object,keep_default_na=False,na_values=[])

f1_val=pd.read_csv(validation_fold_1_name,dtype=object,keep_default_na=False,na_values=[])
f2_val=pd.read_csv(validation_fold_2_name,dtype=object,keep_default_na=False,na_values=[])
f3_val=pd.read_csv(validation_fold_3_name,dtype=object,keep_default_na=False,na_values=[])
f4_val=pd.read_csv(validation_fold_4_name,dtype=object,keep_default_na=False,na_values=[])


################Fold-1#####################
f1_train_x=f1_train['imgs']
f1_train_y=f1_train['lbls']
f1_val_x=f1_val['imgs']
f1_val_y=f1_val['lbls']

Save_Training_folds_x.append(f1_train_x)
Save_Training_folds_y.append(f1_train_y)
Save_Validation_folds_x.append(f1_val_x)
Save_Validation_folds_y.append(f1_val_y)
##################Fold-2##################
f2_train_x=f2_train['imgs']
f2_train_y=f2_train['lbls']
f2_val_x=f2_val['imgs']
f2_val_y=f2_val['lbls']

Save_Training_folds_x.append(f2_train_x)
Save_Training_folds_y.append(f2_train_y)
Save_Validation_folds_x.append(f2_val_x)
Save_Validation_folds_y.append(f2_val_y)
##################Fold-3##################
f3_train_x=f3_train['imgs']
f3_train_y=f3_train['lbls']
f3_val_x=f3_val['imgs']
f3_val_y=f3_val['lbls']

Save_Training_folds_x.append(f3_train_x)
Save_Training_folds_y.append(f3_train_y)
Save_Validation_folds_x.append(f3_val_x)
Save_Validation_folds_y.append(f3_val_y)

##################Fold-4##################
f4_train_x=f4_train['imgs']
f4_train_y=f4_train['lbls']
f4_val_x=f4_val['imgs']
f4_val_y=f4_val['lbls']

Save_Training_folds_x.append(f4_train_x)
Save_Training_folds_y.append(f4_train_y)
Save_Validation_folds_x.append(f4_val_x)
Save_Validation_folds_y.append(f4_val_y)




################################Excel is Made########################################

###########Now Reading the image and Labels From the Folds######################
for f in range(0,Number_of_folds):

    cc=f+1
    print('----------- Start Loading Fold-{} Data----------------'.format(cc))

    loading_fold_training_images_path=Save_Training_folds_x[f]
    loading_fold_training_label=Save_Training_folds_y[f]

    loading_fold_Validation_images_path=Save_Validation_folds_x[f]
    loading_fold_Validation_label=Save_Validation_folds_y[f]


    #print(loading_fold_training_images_path,loading_fold_training_label)

    TRAINING_X=[]
    VALIDATION_X=[]
    ###fold Counter #########

    #########################

    for CT_TRAIN in loading_fold_training_images_path:

        #print(CT_TRAIN)
        img1 = cv2.imread(CT_TRAIN)
        img1 = cv2.resize(img1,(224,224))
        img1 = np.tile(img1,(1,1,1))
        x=image.img_to_array(img1) #converting the image to array
        x=np.expand_dims(x,axis=0) #putting them in the row axis
        x=preprocess_input(x) #preprocessing
        TRAINING_X.append(x)

    traing_CT_img = np.array(TRAINING_X)
    traing_CT_img =np.rollaxis(traing_CT_img,1,0)
    traing_CT_img=traing_CT_img[0]
    print(traing_CT_img.shape)
    Number_of_Training_sample_in_this_fold=traing_CT_img.shape[0]
    print('Total Number of Training Sample In Fold-{} is {}'.format(cc,Number_of_Training_sample_in_this_fold))

    print('<----------Validation---------->"')
    for CT_VAL in loading_fold_Validation_images_path:

        #print(CT_VAL)
        img1 = cv2.imread(CT_VAL)
        img1 = cv2.resize(img1,(224,224))
        img1 = np.tile(img1,(1,1,1))
        x=image.img_to_array(img1) #converting the image to array
        x=np.expand_dims(x,axis=0) #putting them in the row axis
        x=preprocess_input(x) #preprocessing
        VALIDATION_X.append(x)

    Validation_CT_img = np.array(VALIDATION_X)
    Validation_CT_img =np.rollaxis(Validation_CT_img,1,0)
    Validation_CT_img=Validation_CT_img[0]
    print(Validation_CT_img.shape)
    Number_of_Validation_sample_in_this_fold=Validation_CT_img.shape[0]
    print('Total Number of Validation Sample In Fold-{} is {}'.format(cc,Number_of_Validation_sample_in_this_fold))



    print('#-----------Done Loading Fold-{} Data----------------#'.format(cc))

    #########################Data Pre-Procesing ############################
    #############################CNN Part Starts here ######################

    #Image Generator
    datagen = ImageDataGenerator(
              rescale=1./255.0,
              featurewise_center=True,
              featurewise_std_normalization=True,
              rotation_range=20,
              width_shift_range=0.2,
              height_shift_range=0.2,
              horizontal_flip=True)
    test_datagen = ImageDataGenerator(rescale=1./255.0)

    ##This is required when  featurewise_center/featurewise_std_normalization is True
    datagen.fit(traing_CT_img)
    test_datagen.fit(Validation_CT_img)

    #### Batch Normalization
    traing_CT_img= datagen.standardize(traing_CT_img)
    Validation_CT_img= test_datagen.standardize(Validation_CT_img)


    loading_fold_training_label_categorical= np_utils.to_categorical(loading_fold_training_label,num_classes)
    loading_fold_Validation_label_categorical= np_utils.to_categorical(loading_fold_Validation_label,num_classes)
    ###Generating batches
    train_datagen=datagen.flow(traing_CT_img,loading_fold_training_label_categorical,batch_size=batch_size)
    validation_datagen=test_datagen.flow(Validation_CT_img,loading_fold_Validation_label_categorical,batch_size=batch_size)

    ###Check-point
    tensorboard = TensorBoard(log_dir="logs{}_5_16_2019_HU200/Fold-{}_{}".format(Disease_name,cc,time()))
    filepath= '{}_weights-best_model-Fold-{}'.format(Disease_name,cc)+'.hdf5'
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint,EarlyStopping(monitor='val_acc', patience=300, verbose=1, mode='max'),tensorboard]

    custom_resnet_model=get_model('Resnet50')

    print('----Starting training.......>.>.>..')

    his=custom_resnet_model.fit_generator(train_datagen,
                                          validation_data=validation_datagen,
                                          epochs=epochs,
                                          steps_per_epoch=math.ceil(traing_CT_img.shape[0]/batch_size),
                                          verbose=1,
                                          validation_steps=math.ceil(Validation_CT_img.shape[0]/batch_size),
                                          callbacks=callbacks_list)


    ##### Saving The History
    History_name = '{}-History-Fold-{}'.format(Disease_name,cc)+'.csv'
    pd.DataFrame(his.history).to_csv(History_name)


    ####Prediction
    pred = custom_resnet_model.predict(Validation_CT_img)
    score= custom_resnet_model.evaluate(Validation_CT_img, loading_fold_Validation_label_categorical)

    probability_for_normal_class=pred[:,0]
    probability_for_edema_class=pred[:,1]


    prediction=np.argmax(pred, axis=-1)

    roc_name = '{}_roc-Fold-{}'.format(Disease_name,cc)+'.csv'
    #fpr_keras, tpr_keras, _ = roc_curve(loading_fold_Validation_label,probability_for_atlectasis_class)
    Inf0_data=pd.DataFrame(list(zip(pred,probability_for_normal_class,probability_for_edema_class,prediction,
                                    loading_fold_Validation_label)),
    columns=['pred','pred0','pred1','prediction','y_true'])
    Inf0_data.to_csv(roc_name, encoding='utf-8', index=False)

    ###calculating AUC
    #auc_keras = auc(fpr_keras, tpr_keras)

    ####Saving Validation Data
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

    PredictionAndTrueLabel_name = '{}_PredictionAndTruelabel-Fold-{}'.format(Disease_name,cc)+'.csv'
    Inf0_data=pd.DataFrame(list(zip(prediction,loading_fold_Validation_label)),
    columns=['pre','lbls'])
    Inf0_data.to_csv(PredictionAndTrueLabel_name, encoding='utf-8', index=False)



    ####Model and Normalizer
    model_name = '{}_Model-Fold-{}'.format(Disease_name,cc)+'.json'
    normalizer_name = '{}_Normalizer-Fold-{}'.format(Disease_name,cc)+'.pkl'
    #####Saving t he model##########
    model_json = custom_resnet_model.to_json()
    with open(model_name, 'w') as json_file:
        json_file.write(model_json)

    joblib.dump(datagen,normalizer_name)

    print("model and weights have been saved\n")
    print('#-----------Done Training Fold-{} Data----------------#\n'.format(cc))

print('----------------------Training Ends------------------------------ ')
