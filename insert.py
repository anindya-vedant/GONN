import warnings
warnings.filterwarnings('ignore')
import random
import pandas as pd
import numpy
from sklearn.model_selection import train_test_split
import matplotlib as plt

class Node:
    def __init__(self,data):
        self.data=data
        self.left=None
        self.right=None

def insert_1(isterminal, height):
    if (height == 6):
        node_data = random.choice([i for i in range(len(dataTrain.values[0]) - 1)])
        root = Node(node_data)
        root.left = None
        root.right = None
    else:
        if (isterminal == False):
            node_data = random.choice(["+", "-", "*", "/"])
            root = Node(node_data)
            isterminal = random.choice([True, False])
            root.left = insert_1(isterminal, height + 1)
            isterminal = random.choice([True, False])
            root.right = insert_1(isterminal, height + 1)
        else:
            node_data = random.choice([i for i in range(len(dataTrain.values[0]) - 1)])
            root = Node(node_data)
            root.right = None
            root.left = None
    return root


def insert(isterminal,height,data=None):
    if(height==6 or isterminal==True):
        choices=[]
        choices.append(random.uniform(0.0,10.0))
        choices.append(i for i in range(len(dataTrain.values[0])-1))
        node_data=random.choice(choices)
        root=Node(node_data)
        root.left=None
        root.right=None
    else:
        if(height==0):
            root=Node("P")
            isterminal=False
            root.left=insert(isterminal,height+1)
            root.right=insert(isterminal,height+1)
        elif(height==1 or data=="P"):
            root=Node("W")
            isterminal=random.choice([True,False])
            root.left=insert_1(isterminal,height+1)
            isterminal=False
            root.right=insert(isterminal,height+1)
        else:
            choices=["P"]
            choices.append(random.uniform(0.0,10.0))
            choices.append(i for i in range(len(dataTrain.values[0])-1))
            choices.append("+")
            choices.append("-")
            choices.append("*")
            choices.append("/")
            node_data=random.choice(choices)
            root = Node(node_data)
            if(node_data=="P"):
                isterminal=False
                root.left=insert(isterminal,height,node_data)
                root.right=insert(isterminal,height,node_data)
            elif(type(node_data)==str):
                isterminal=random.choice([True,False])
                root.left=insert_1(isterminal,height+1)
                isterminal = random.choice([True, False])
                root.right=insert_1(isterminal,height+1)
    return root

def preorder_expression_array(root):
    if root:
        tree_expression.append(root.data)
        preorder_expression_array(root.left)
        preorder_expression_array(root.right)
def expression_array(root):
    if root:
        expression_array(root.left)
        expression.append(root.data)
        expression_array(root.right)

dataset=pd.read_csv('wbc-dataset.csv', header=None)
dataset=dataset.drop(0,axis=1)
dataset=dataset.replace("?",numpy.NaN)
dataset.dropna(inplace=True)
dataTrain,dataTest = train_test_split(dataset,test_size = 0.5)
features=dataTrain.iloc[:,:-1]
classes=dataTrain.iloc[:,-1]
tree_population=[]
accuracy_list=[]
preorder_trees=[]
fitness_values=[]
root_array=[]
for i in range(100):
    expression=[]
    tree_expression = []
    Root=insert(False,0)
    root_array.append(Root)
    expression_array(Root)
    preorder_expression_array(Root)
    tree_population.append(expression)
    preorder_trees.append(tree_expression)
for i in preorder_trees:
    for j in i:
        if(type(j)!= int and type (j)!= str and type(j)!=float):
            print(j)
            print(i)