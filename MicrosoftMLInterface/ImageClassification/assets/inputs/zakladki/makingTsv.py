import os
from os import walk
import csv

filenames1 = []
filenames2 = []
for (dirpath, dirnames, filenames) in walk("./test/nezakl"):
    filenames1.extend(filenames)
    break
for (dirpath, dirnames, filenames) in walk("./test/zakl"):
    filenames2.extend(filenames)
    break  
dictionary = {}

for file in filenames1:
    dictionary.update({file:"nezakl"})
for file in filenames2:
    dictionary.update({file:"zakl"})    

w = csv.writer(open("image_list.tsv", "w"), delimiter='\t')
for key, val in dictionary.items():
    w.writerow([key, val])

def remove_empty_lines(filename):
    if not os.path.isfile(filename):
        print("{} does not exist ".format(filename))
        return
    with open(filename) as filehandle:
        lines = filehandle.readlines()

    with open(filename, 'w') as filehandle:
        lines = filter(lambda x: x.strip(), lines)
        filehandle.writelines(lines)   
#remove_empty_lines("image_list.tsv")    
