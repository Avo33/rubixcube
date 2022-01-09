import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import clear_output
from tensorflow.keras import datasets, layers, models

CSV_COLUMN_NAMES = ['1','2','3','4','5','6','7',
              '8','9','10','11','12','13',
              '14','15','16','17','18','19',
              '20','21','22','23','24','25',
              '26','27','28','29','30','31',
              '32','33','34','35','36','37',
              '38','39','40','41','42','43',
              '44','45','46','47','48','49',
              '50','51','52','53','54', 'heuristic']
HEURISTIC = ['1','2','3','4','5','6','7',
              '8','9','10','11','12','13',
              '14','15','16','17','18','19',
              '20']


"""load data"""
dftrain = pd.read_csv('train.csv')
dfeval = pd.read_csv('evaluation.csv')
print(dftrain.head())
y_train = dftrain.pop('heuristic')
y_eval = dfeval.pop('heuristic')


"""convert categorical data into numeric data"""
NUMERIC_COLUMNS = ['1','2','3','4','5','6','7',
              '8','9','10','11','12','13',
              '14','15','16','17','18','19',
              '20','21','22','23','24','25',
              '26','27','28','29','30','31',
              '32','33','34','35','36','37',
              '38','39','40','41','42','43',
              '44','45','46','47','48','49',
              '50','51','52','53','54']
feature_columns = []
for feature_name in NUMERIC_COLUMNS:
    feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))
print(feature_columns)


"""Building CNN Base"""
model = models.Sequential()
model.add(layers.Conv2D(54, (3, 1), activation='relu', input_shape=(54, 1, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(108, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(108, (3, 3), activation='relu'))

""""Adding Dense Layers"""
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(20))


"""Training"""
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(dftrain, y_train, epochs=4,
                    validation_data=(dfeval, y_eval))

test_loss, test_acc = model.evaluate(dfeval, y_eval, verbose=2)

print(test_acc)