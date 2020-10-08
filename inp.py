import random
import json

f = open('Input' , 'w')

A= []

for i in range(1):
    LIST = []
    for j in range(7):
        x,y = random.randint(0,100),random.randint(0,100)
        x1,y1 = random.randint(0,100),random.randint(0,100)
        LIST.append([(x,y) , (x1,y1)])
    A.append(LIST)

json.dump(A, f)

f.close() 
