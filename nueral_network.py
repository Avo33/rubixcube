import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import clear_output

CSV_COLUMN_NAMES = ['1','2','3','4','5','6','7',
              '8','9','10','11','12','13',
              '14','15','16','17','18','19',
              '20','21','22','23','24','25',
              '26','27','28','29','30','31',
              '32','33','34','35','36','37',
              '38','39','40','41','42','43',
              '44','45','46','47','48','49',
              '50','51','52','53','54', 'heuristic']
HEURISTIC = ['0','1','2','3','4','5','6','7',
              '8','9','10','11','12','13',
              '14','15','16','17','18','19']
numClasses = 20

"""load data"""
dftrain = pd.read_csv('train.csv', names=CSV_COLUMN_NAMES, header=0)
dfeval = pd.read_csv('evaluation.csv', names=CSV_COLUMN_NAMES, header=0)
y_train = dftrain.pop('heuristic')
y_eval = dfeval.pop('heuristic')


my_feature_columns = []
NUMERIC_COLUMNS = ['1','2','3','4','5','6','7',
              '8','9','10','11','12','13',
              '14','15','16','17','18','19',
              '20','21','22','23','24','25',
              '26','27','28','29','30','31',
              '32','33','34','35','36','37',
              '38','39','40','41','42','43',
              '44','45','46','47','48','49',
              '50','51','52','53','54']
for feature_name in NUMERIC_COLUMNS:
    my_feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))



'''Input Function'''
def input_fn(features, labels, training=True, batch_size=256):
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    # Shuffle and repeat if you are in training mode.
    if training:
        dataset = dataset.shuffle(1000).repeat()
    return dataset.batch(batch_size)

"""Build a DNN with 3 hidden layers with 10, 20 and 10 hidden nodes each."""
classifier = tf.estimator.DNNClassifier(
    feature_columns=my_feature_columns,
    hidden_units=[20,20,20],
    # The model must choose between 20 classes.
    n_classes=numClasses)

classifier.train(
    input_fn=lambda: input_fn(dftrain, y_train, training=True),
    steps=500)


eval_result = classifier.evaluate(
    input_fn=lambda: input_fn(dfeval, y_eval, training=False))

print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))








