import glob
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import keras
import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import cv2


Label_list = ["Black-grass", "Charlock", "Cleavers", "Common Chickweed", "Common wheat", "Fat Hen",
                "Loose Silky-bent", "Maize", "Scentless Mayweed", "Shepherds Purse", "Small-flowered Cranesbill",
                "Sugar beet"]
PATH_TRAIN = "/content/output/train/"
PATH_VAL = "/content/output/val"
PATH_TEST = "/content/output/test"
IMAGE_SIZE = 224
BATCH_SIZE = 32

# Số ảnh mỗi nhãn
file_num = []
for i in range(12) :
    imfile = glob.glob(PATH_TRAIN +Label_list[i]+'/*.png')
    file_num += [len(imfile)]

print(file_num)
# vẽ biểu đồ dữ liệu tập train
fig = plt.figure(figsize = (8,5))
ax = fig.add_subplot(111)
ax.bar(Label_list,file_num)
plt.xticks(rotation = 90)
plt.savefig('train_chart')
plt.show()

input_shape = (IMAGE_SIZE,IMAGE_SIZE,3)
n_classes = 12

model = keras.Sequential(
    [
    layers.InputLayer(input_shape=input_shape),
    layers.Conv2D(32, kernel_size = (3,3), activation='relu',strides = (1,1), padding = 'same'),
    layers.Conv2D(64, kernel_size = (3,3), activation='relu', strides = (1,1), padding = 'same'),
    layers.MaxPooling2D((2, 2)),

     
    layers.Conv2D(64,  kernel_size = (3,3), activation='relu',strides = (1,1), padding = 'same'),
    layers.Conv2D(64,  kernel_size = (3,3), activation='relu',strides = (1,1), padding = 'same'),
    layers.MaxPooling2D((2, 2)),
     
    layers.Conv2D(64, (3, 3), activation='relu',strides = (1,1), padding = 'same'),
    layers.Conv2D(64, (3, 3), activation='relu',strides = (1,1), padding = 'same'),
    layers.MaxPooling2D((2, 2)),
     
    layers.Conv2D(64, kernel_size = (3,3), activation='relu',strides = (1,1),padding = 'same'),
    layers.Conv2D(64, (3, 3), activation='relu',strides = (1,1), padding = 'same'),
    layers.MaxPooling2D((2, 2)),
     
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(n_classes, activation='softmax'),
    ]
)
model.summary()
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)
train_generator, validation_generator , test_generator = define_generator()
EPOCHS = 20
history = model.fit(
    train_generator,
    steps_per_epoch= train_generator.samples // BATCH_SIZE,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE,
    verbose=1,
    epochs=EPOCHS,
    #callbacks=[save_callback] dung khi luu gia tri tot nhat cua val_acc.
)
#loss: 0.5239 - accuracy: 0.8239 - val_loss: 0.5010 - val_accuracy: 0.8362
accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(range(EPOCHS), accuracy, label='Training Accuracy')
plt.plot(range(EPOCHS), val_accuracy, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(range(EPOCHS), loss, label='Training Loss')
plt.plot(range(EPOCHS), val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
plt.savefig('Not_use_BatchNorm_and_Dropout.png')

# Kiểm tra với một ảnh bất kỳ
path_img = 'PATH_IMG'
im = np.array(Image.open(path_img))
im = cv2.resize(im,(224,224))
im = numpy.expand_dims(im,0)
# print(im.shape)
# plt.imshow(im[0])
y_predict = model.predict(im)
print("Gia tri du doan", Label_list[np.argmax(y_predict)])
