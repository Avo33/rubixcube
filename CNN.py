import tensorflow as tf
import os
import random
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import clear_output
from tensorflow.keras import datasets, layers, models
import cv2
import matplotlib.image as mpimg

IMG_WIDTH=54
IMG_HEIGHT=1
img_folder=r'rgb_image\1'
test_folder=r'test_rbg\1'
plt.figure(figsize=(20,20))

for i in range(5):
    file = random.choice(os.listdir(img_folder))
    image_path= os.path.join(img_folder, file)
    img = mpimg.imread(image_path)
    ax = plt.subplot(1,5,i+1)
    ax.title.set_text(file)
    plt.imshow(img)


for i in range(5):
    test_file = random.choice(os.listdir(test_folder))
    test_image_path= os.path.join(test_folder, test_file)
    test_img = mpimg.imread(test_image_path)
    test_ax = plt.subplot(1,5,i+1)
    test_ax.title.set_text(test_file)
    plt.imshow(test_img)


def create_dataset(img_folder):
    img_data_array = []
    class_name = []

    for dir1 in os.listdir(img_folder):
        for file in os.listdir(os.path.join(img_folder, dir1)):
            image_path = os.path.join(img_folder, dir1, file)
            image = cv2.imread(image_path, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH), interpolation=cv2.INTER_AREA)
            image = np.array(image)
            image = image.astype('float32')
            image /= 255
            img_data_array.append(image)
            class_name.append(dir1)
    return img_data_array, class_name


# extract the image array and class name
img_data, class_name = create_dataset(r'rgb_image')
test_img_data, test_class_name = create_dataset(r'test_rbg')


target_dict={k: v for v, k in enumerate(np.unique(class_name))}
test_target_dict={k: v for v, k in enumerate(np.unique(test_class_name))}
print(target_dict)

target_val=[target_dict[class_name[i]] for i in range(len(class_name))]
print(target_val)


# CSV_COLUMN_NAMES = ['1','2','3','4','5','6','7',
#               '8','9','10','11','12','13',
#               '14','15','16','17','18','19',
#               '20','21','22','23','24','25',
#               '26','27','28','29','30','31',
#               '32','33','34','35','36','37',
#               '38','39','40','41','42','43',
#               '44','45','46','47','48','49',
#               '50','51','52','53','54', 'heuristic']
# HEURISTIC = ['1','2','3','4','5','6','7',
#               '8','9','10','11','12','13',
#               '14','15','16','17','18','19',
#               '20']
#
#
# """load data"""
img_data=np.array(img_data)
img_data=np.reshape(img_data,(100,162))
dftrain = pd.DataFrame(img_data)
# dfeval = pd.read_csv('evaluation.csv')
# print(dftrain.head())
y_train = np.array(target_val)
# y_eval = dfeval.pop('heuristic')
print(type(img_data))
print(type(y_train))

test_img_data=np.array(test_img_data)
test_img_data=np.reshape(test_img_data,(100,162))
dftrain = pd.DataFrame(test_img_data)
# dfeval = pd.read_csv('evaluation.csv')
# print(dftrain.head())
y_train = np.array(target_val)
y_eval = y_train



#
#
# """convert categorical data into numeric data"""
# NUMERIC_COLUMNS = ['1','2','3','4','5','6','7',
#               '8','9','10','11','12','13',
#               '14','15','16','17','18','19',
#               '20','21','22','23','24','25',
#               '26','27','28','29','30','31',
#               '32','33','34','35','36','37',
#               '38','39','40','41','42','43',
#               '44','45','46','47','48','49',
#               '50','51','52','53','54']
# feature_columns = []
# for feature_name in NUMERIC_COLUMNS:
#     feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))
# print(feature_columns)
#
#
"""Building CNN Base"""
model = models.Sequential()
model.add(layers.Conv2D(162, (3, 1), activation='relu', input_shape=(162, 1, 3)))
model.add(layers.MaxPooling2D((2, 1)))
model.add(layers.Conv2D(108, (3, 1), activation='relu'))
model.add(layers.MaxPooling2D((2, 1)))
model.add(layers.Conv2D(108, (3, 1 ), activation='relu'))

""""Adding Dense Layers"""
model.add(layers.Flatten())
model.add(layers.Dense(162, activation='relu'))
model.add(layers.Dense(20))


"""Training"""
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])



history = model.fit(img_model = tf.expand_dims(img_data, axis=-1),y_model=tf.expand_dims(y_train, axis=-1),epochs=4)

# test_loss, test_acc = model.evaluate(img_data, y_train, verbose=2)
#
# print(test_acc)

