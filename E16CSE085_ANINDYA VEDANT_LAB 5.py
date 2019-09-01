import random
import pandas as pd
import numpy
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('wbc-dataset.csv',header = None)
dataset  = dataset.drop(0,axis = 1)
dataset = dataset.replace('?', numpy.NaN)
dataset.dropna(inplace = True)
dataTrain,dataTest = train_test_split(dataset,test_size = 0.5)

class Node:
    def __init__(self, data):
        self.data = data
        self.left = None
        self.right = None

    def insert(isterminal,height):
        if(height==6):
            data = random.choice([i for i in range(len(dataTrain.values[0])-1)])
            fnum.append(data)
            root = Node(data)
            root.left = None
            root.right = None
        else:
            if(isterminal == False):
                data = random.choice(['+','-','*','/'])
                root = Node(data)
                isterminal = random.choice([True,False])
                root.left = insert(isterminal,height+1)
                isterminal = random.choice([True,False])
                root.right = insert(isterminal,height+1)
            else:
                data = random.choice([i for i in range(len(dataTrain.values[0])-1)])
                fnum.append(data)
                root= Node(data)
                root.left = None
                root.right = None
        return root

    def evaluateExpressionTree(root):
        if root is None:
            return 0
        if root.left is None and root.right is None:
            return int(root.data)
        left_sum = evaluateExpressionTree(root.left)
        right_sum = evaluateExpressionTree(root.right)
        if root.data == '+':
            return left_sum + right_sum
        elif root.data == '-':
            return left_sum - right_sum
        elif root.data == '*':
            return left_sum * right_sum
        else:
            try:
                return left_sum / right_sum
            except:
                return 0
    def getfnum():
        global c
        c+=1
        return fnum[c]`

    def fit_tree(Root,instance_num,data):
        if Root is None:
            return
        if(type(Root.data)==int):
            feature_num = getfnum()
            Root.data = data.values[instance_num,feature_num]
        fit_tree(Root.left,instance_num,data)
        fit_tree(Root.right,instance_num,data)

    def getaccuracy(data):
        accuracy = 0
        for i in range (len(data)):
            global c
            c = -1
            fit_tree(Root,i,data)
            if(evaluateExpressionTree(Root)>=0):
                Class = 2
            else:
                Class = 4
            if(data.values[i,9]==Class):
                accuracy+=1
        accuracy = (accuracy/len(data))*100
        return accuracy

    def tree(Root):
        if(Root is None):
            return
        print(Root.data)
        tree(Root.left)
        tree(Root.right)

fnum = []
Root = insert(False,0)
print(getaccuracy(dataTrain))
print(getaccuracy(dataTest))
tree(Root)

def changefun(Root):
    if Root is None:
        return
    global evolved
    choice = random.choice([True,False])
    if(type(Root.data)==str and evolved == False and choice == True):
        Root.data = random.choice(['+','-','/','*'])
        evolved = True
    changefun(Root.left)
    changefun(Root.right)

def changeterm(Root):
    if Root is None:
        return
    global evolved
    if(type(Root.data)==int and evolved == False):
        feature_num = random.choice([i for i in range(len(dataTrain[0])-1)])
        fnum[0] = feature_num
        evolved = True
    changefun(Root.left)
    changefun(Root.right)

global evolved
for i in range(50):
    ch = random.choice(['f','t'])
    evolved = False
    changefun(Root)
    if(getaccuracy(dataTest)==100):
        break
print(getaccuracy(dataTest))
