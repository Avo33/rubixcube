from ScrambleRubixcube import xInitial, make_move
import csv
import random
import numpy as np
import matplotlib.pyplot as plt

num_pic = 0
header = ['1','2','3','4','5','6','7',
              '8','9','10','11','12','13',
              '14','15','16','17','18','19',
              '20','21','22','23','24','25',
              '26','27','28','29','30','31',
              '32','33','34','35','36','37',
              '38','39','40','41','42','43',
              '44','45','46','47','48','49',
              '50','51','52','53','54', 'heuristic']

with open('train_data.csv','w',encoding='UTF8',newline='') as f:
    writer= csv.writer(f)
    writer.writerow(header)

    num_move=21
    for k in range(1,num_move):
        num_pair = 5
        move = np.zeros((num_pair,k))
        csv_data = []
        reverse = 0
        move = move.tolist()
        for i in range(num_pair):
            x = xInitial
            for j in range(k):
                move[i][j] = random.randint(1,12)
                m = move[i][j]
                make_move(x, m, reverse)
                y = x.reshape((-1,54))
                y = y[0].tolist()
                for e in range(len(y)):
                    if y[e] == 'B':
                        y[e] = [0,0,255]
                    if y[e] == 'W':
                        y[e] = [255,255,255]
                    if y[e] == 'R':
                        y[e] = [255,0,0]
                    if y[e] == 'Y':
                        y[e] = [255,255,0]
                    if y[e] == 'O':
                        y[e] = [255,160,122]
                    if y[e] == 'G':
                        y[e] = [0,128,0]
            print(type(y))
            rgb_array = y
            img = np.array(rgb_array, dtype=int).reshape((1, len(rgb_array), 3))
            plt.imshow(img, extent=[0, 54, 0, 1], aspect='auto')
            plt.savefig("image{}.jpg".format(num_pic))
            num_pic += 1
            y += [k - 1]
            csv_data.append(y)
        for data in csv_data:
            writer.writerow(data)


