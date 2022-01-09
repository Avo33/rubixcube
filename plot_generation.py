import csv
import numpy as np
from data_generation import num_pair
import matplotlib.pyplot as plt

file = open('train.csv')
csvreader = csv.reader(file)
header = []
header = next(csvreader)

num_row = num_pair*20
rows = []
for row in csvreader:
    for i in range(54):
        rows.append(row[i])


file.close

arr = np.array(rows)
arr_2d = np.resize(arr,(num_pair*20,54))


"""generate graph"""
for i in range(len(arr_2d)):
    rbg_arr = arr_2d[i,:]
    print(rbg_arr[0][1])