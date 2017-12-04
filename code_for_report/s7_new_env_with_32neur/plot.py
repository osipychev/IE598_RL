import sys
import numpy as np
import csv
import matplotlib.pyplot as plt

if len(sys.argv)<4: 
   print('use format: {python plot.py filename columnx columny title xlabel ylabel}')
   sys.exit()

filename = sys.argv[1]
x_col = int(sys.argv[2])
y_col = int(sys.argv[3])

if len(sys.argv)<5:
   title = []
   xlabel = []
   ylabel = []
else:
   title = sys.argv[4]
   xlabel = sys.argv[5]
   ylabel = sys.argv[6]

x_collection = []
y_collection = []

with open(filename) as f:
    data = f.read()

data = data.split('\n')

for row in data:
    if '[' in row:
       row = row.replace('[',' ')
       row = row.replace(']',' ')
       row_spl = row.split()
       if (len(row_spl) > x_col and len(row_spl) > y_col):
          x = row_spl[x_col]
          y = row_spl[y_col]
          print(x,y)
          x_collection.append(x)
          y_collection.append(y)

plt.title(title)    
plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.plot(x_collection,y_collection, c='r', label='the data')
plt.grid(True)
plt.savefig('plot.pdf')
plt.show()


