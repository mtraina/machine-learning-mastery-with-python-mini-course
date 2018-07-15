# Scatter Plot Matrix
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from pandas import read_csv
from pandas.plotting import scatter_matrix

url = 'https://goo.gl/bDdBiA'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(url, names=names)
scatter_matrix(data)
plt.show()