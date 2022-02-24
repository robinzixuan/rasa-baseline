import pandas as pd
import numpy as np

data = pd.read_csv('Store_Clothes_Detail.csv')

name = list(data['skuName'])

name_str = ''
for i in name:
   name_str +=  i + '\n'


with open('name.txt', 'w') as f:
    f.write(name_str)