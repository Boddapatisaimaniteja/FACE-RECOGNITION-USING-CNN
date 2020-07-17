# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 13:11:21 2020

@author: bodda
"""

import matplotlib.pyplot as plt
import keras
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D,Flatten,Activation,Dropout,BatchNormalization,MaxPooling2D,Dense
from keras.models import Sequential

img_rows,img_col=74,74
Batch_size=16
no_of_classes=3

train_dir=r'E:\face rec\train'
validation_dir=r'E:\face rec\validation'

train_datagen=ImageDataGenerator(rescale=1./255,zoom_range=0.3,shear_range=0.4,horizontal_flip=True,vertical_flip=True)
validation_datagen=ImageDataGenerator(rescale=1./255)

training_data=train_datagen.flow_from_directory(train_dir,target_size=(img_rows,img_col),color_mode='rgb',class_mode='categorical',shuffle=True,batch_size=Batch_size)
validation_data=validation_datagen.flow_from_directory(validation_dir,target_size=(img_rows,img_col),color_mode='rgb',class_mode='categorical',shuffle=True,batch_size=Batch_size)

model=Sequential()

#Block1
model.add(Conv2D(32,(3,3),kernel_initializer='he_normal',padding='same',input_shape=(img_rows,img_col,3)))
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(Conv2D(32,(3,3),kernel_initializer='he_normal',padding='same',input_shape=(img_rows,img_col,3)))
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

#block2
model.add(Conv2D(64,(3,3),kernel_initializer='he_normal',padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(Conv2D(64,(3,3),kernel_initializer='he_normal',padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))
#block3
model.add(Conv2D(128,(3,3),kernel_initializer='he_normal',padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(Conv2D(128,(3,3),kernel_initializer='he_normal',padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

#block4
model.add(Conv2D(256,(3,3),kernel_initializer='he_normal',padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(Conv2D(256,(3,3),kernel_initializer='he_normal',padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

#block5
model.add(Flatten())
model.add(Dense(128,kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
#block6
model.add(Dense(128,kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
#block7
model.add(Dense(3,kernel_initializer='he_normal'))
model.add(Activation('softmax'))
#checkpoints and earlystopping
# earlystopping
from keras.optimizers import RMSprop,Adam,SGD
from keras.callbacks import ModelCheckpoint,EarlyStopping,ReduceLROnPlateau

checkpoint=ModelCheckpoint(r'E:\face rec\fe.h5',monitor='val_loss',mode='min',save_best_only=True,verbose=1)

earlystop=EarlyStopping(monitor='val_loss',min_delta=0,patience=3,verbose=1,restore_best_weights=True)

reduce_lr=ReduceLROnPlateau(monitor='val_loss',factor=0.2,patience=3,verbose=1,min_delta=0.0001)

callbacks=[earlystop,checkpoint,reduce_lr]



#commpile
model.compile(optimizer=Adam(lr=0.001),loss='categorical_crossentropy',metrics=['accuracy'])

no_training_s=7500
no_valid_s=3000
epochs=10
#fitting
history=model.fit_generator(training_data,steps_per_epoch=no_training_s//Batch_size,epochs=epochs,validation_data=validation_data,validation_steps=no_valid_s//Batch_size,callbacks=callbacks)
plt.plot(history.history['accuracy'],c='b',label='training_acc')
plt.plot(history.history['val_accuracy'],c='r',label='val_acc')
plt.legend()