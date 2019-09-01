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
        choices.append(random.uniform(0.1,10.0))
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
            isterminal=False
            # isterminal=True
            # isterminal=random.choice([True,False])
            root.left=insert_1(isterminal,height+1)
            # isterminal=random.choice([True,False])
            isterminal=False
            # isterminal=True
            root.right=insert(isterminal,height+1)
        else:
            choices=["P"]
            choices.append(random.uniform(0.1,10.0))
            for i in range(len(dataTrain.values[0])-1):
                choices.append(i)
            choices.append("+")
            choices.append("-")
            choices.append("*")
            choices.append("/")
            # if (height>4):
            #     print("Height is", height)
            #     choices.remove("P")
            node_data=random.choice(choices)
            print(choices)
            root = Node(node_data)
            if(node_data=="P"):
                isterminal=False
                root.left=insert(isterminal,height+1,node_data)
                root.right=insert(isterminal,height+1,node_data)
            elif(type(node_data)==str):
                isterminal=random.choice([True,False])
                root.left=insert_1(isterminal,height+1)
                isterminal = random.choice([True, False])
                root.right=insert_1(isterminal,height+1)
    return root

def sig(x):
    return 1/(1+numpy.exp(-x))

def evaluateExpressionTree(root):
    if root is None:
        return 0
    if root.left is None and root.right is None:
        return (root.data)
    left_sum = evaluateExpressionTree(root.left)
    right_sum = evaluateExpressionTree(root.right)
    if root.data == '+':
        return left_sum + right_sum
    elif root.data == '-':
        return left_sum - right_sum
    elif root.data == '*':
        return left_sum * right_sum
    elif root.data == "W":
        return left_sum*right_sum
    elif root.data=="P":
        x= left_sum+right_sum
        return 1 / (1 + numpy.exp(-x))
    else:
        try:
            return left_sum / right_sum
        except:
            return 0
def getstringexpression(expression):
    exp = ""
    for i in expression:
        exp += str(i)
    return exp
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
def printInorder(root):
    if root:
        # First recur on left child


        # then print the data of node
        print(root.data),
        printInorder(root.left)

        # now recur on right child
        printInorder(root.right)

def evaluatestack(stack):
    elem = stack.pop(0)

    if (elem == '-'):
        return evaluatestack(stack) - evaluatestack(stack)
    elif (elem == '+'):
        return evaluatestack(stack) + evaluatestack(stack)
    elif (elem == '/'):
        try:
            return evaluatestack(stack) / evaluatestack(stack)
        except:
            return 0
    elif (elem == '*'):
        return evaluatestack(stack) * evaluatestack(stack)
    elif(elem=='P'):
         return sig(evaluatestack(stack)+evaluatestack(stack))
    elif(elem=="W"):
        return evaluatestack(stack)*evaluatestack(stack)
    else:
        # print(type(elem))
        return int(elem)

def dsceval(stou):
    # print(stin)

    tp = 0
    tn = 0
    fp = 0
    fn = 0
    c1 = 0
    c2 = 0
    l = []
    for i in range(len(dataTrain)):
        stin = stou.copy()
        for k in range(len(stin)):
            if stin[k] not in ['+', '-', '*', '/'] and type(stin[k]) == int:
                stin[k] = (dataTrain.values[i, stin[k]])
        # print(i,k,stin)
        val = evaluatestack(stin)
        classifier_values.append(val)
        # print(val)

def putvalue(root,row,row_no):
    root_1=root
    printInorder(root)
    print("Root 1 yeh h")
    printInorder(root_1)
    if root_1.left:
        putvalue(root_1.left,row,row_no)
    if root_1.right:
        putvalue(root_1.right,row,row_no)
    if (type(root_1.data)==int):
        # print(root_1.data)
        # print(row_no)
        root_1.data=dataTrain.values[row_no,root.data]
    #for i in range(len(exp_copy)):
    #     if type(exp_copy[i])!=int:
    #         continue
    #     else:
    #         exp_copy[i]=dataTrain.values[row_no,exp_copy[i]]
    row.append(root)


def trainaccuracy(root,row_values):
    for i in range(len(dataTrain)-1):
        putvalue(root,row_values,i)

def evalexpression(exp):
    return eval(exp)

def bubbleSort(arr,trees,pre,fit):
    n = len(arr)

    # Traverse through all array elements
    for i in range(n):

        # Last i elements are already in place
        for j in range(0, n - i - 1):

            # traverse the array from 0 to n-i-1
            # Swap if the element found is greater
            # than the next element
            if arr[j] < arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                trees[j], trees[j + 1] = trees[j + 1], trees[j]
                pre[j], pre[j + 1] = pre[j + 1], pre[j]
                fit[j], fit[j + 1] = fit[j + 1], fit[j]
def mutation(array):
    index=random.randrange(len(array))
    if type(array[index])==int:
        array[index]=random.choice([i for i in range(len(dataTrain.values[0])-1)])
    elif (array[index] in ["+","-","*","/"]):
        array[index]=random.choice(["+","-","*","/"])


def standard_crossover(tree_1, tree_2):
    index = random.randrange(1, len(tree_1) - 1)
    to_be_copied = []
    rad = tree_1[index::]
    #print(index)
    to_be_copied.append(rad[0])
    i = -1
    indi = [0]
    for j in to_be_copied:
        i = i + 1
        if j in ["+", "-", "*", "/","W","P"]:
            counter = 0
            for k in range(len(tree_1)):

                if i + k not in indi and counter < 2:
                    to_be_copied.append(rad[i + k])
                    indi.append(i + k)
                    counter = counter + 1
                if counter == 2:
                    break

    # print(to_be_copied)
    # print(tree_2)

    index1 = random.randrange(1, len(tree_2) - 1)
    to_be_copied1 = []
    rad = tree_2[index1::]
    # print(rad)
    # print(index1)
    to_be_copied1.append(rad[0])
    i = -1
    indi = [0]
    for j in to_be_copied1:
        i = i + 1
        if j in ["+", "-", "*", "/","W","P"]:
            counter = 0
            for k in range(len(tree_2)):

                if i + k not in indi and counter < 2:
                    to_be_copied1.append(rad[i + k])
                    indi.append(i + k)
                    counter = counter + 1
                if counter == 2:
                    break

    newtree1 = tree_1[0:index] + to_be_copied1 + tree_1[index + len(to_be_copied)::]
    newtree2 = tree_2[0:index1] + to_be_copied + tree_2[index1 + len(to_be_copied1)::]
    next_gen_cross.append(newtree1)
    next_gen_cross.append(newtree2)

    return to_be_copied

def hill_climb(tree_1,tree_2):
    one_index=preorder_trees.index(tree_1)
    tree_1_accuracy=accuracy_list[one_index]
    two_index=preorder_trees.index(tree_2)
    tree_2_accuracy=accuracy_list[two_index]
    index = random.randrange(1, len(tree_1) - 1)
    to_be_copied = []
    rad = tree_1[index::]
    #print(index)
    to_be_copied.append(rad[0])
    i = -1
    indi = [0]
    for j in to_be_copied:
        i = i + 1
        if j in ["+", "-", "*", "/","W","P"]:
            counter = 0
            for k in range(len(tree_1)):

                if i + k not in indi and counter < 2:
                    to_be_copied.append(rad[i + k])
                    indi.append(i + k)
                    counter = counter + 1
                if counter == 2:
                    break

    # print(to_be_copied)
    # print(tree_2)

    index1 = random.randrange(1, len(tree_2) - 1)
    to_be_copied1 = []
    rad = tree_2[index1::]
    # print(rad)
    # print(index1)
    to_be_copied1.append(rad[0])
    i = -1
    indi = [0]
    for j in to_be_copied1:
        i = i + 1
        if j in ["+", "-", "*", "/","W","P"]:
            counter = 0
            for k in range(len(tree_2)):

                if i + k not in indi and counter < 2:
                    to_be_copied1.append(rad[i + k])
                    indi.append(i + k)
                    counter = counter + 1
                if counter == 2:
                    break

    newtree1 = tree_1[0:index] + to_be_copied1 + tree_1[index + len(to_be_copied)::]
    newtree2 = tree_2[0:index1] + to_be_copied + tree_2[index1 + len(to_be_copied1)::]
    tries=0
    classifier_values=[]
    dsceval(newtree1)
    predicted_class = []
    for i in classifier_values:
        if i > 0.5:
            predicted_class.append(2)
        else:
            predicted_class.append(4)
    accuracy_1= 0
    for i in range(len(predicted_class) - 1):
        if (predicted_class[i] == dataTrain.values[i, -1]):
            accuracy_1 += 1
    dsceval(newtree2)
    predicted_class = []
    for i in classifier_values:
        if i > 0.5:
            predicted_class.append(2)
        else:
            predicted_class.append(4)
    accuracy_2 = 0
    for i in range(len(predicted_class) - 1):
        if (predicted_class[i] == dataTrain.values[i, -1]):
            accuracy_2 += 1

    print(accuracy_1)
    print(accuracy_2)
    print(tree_1_accuracy)
    print(tree_2_accuracy)

    if (accuracy_1>tree_1_accuracy and accuracy_1>tree_2_accuracy):
        if(accuracy_2>tree_1_accuracy and accuracy_2>tree_2_accuracy):
            next_gen_hill_climb.append(newtree1)
            next_gen_hill_climb.append(newtree2)
    elif(accuracy_1>tree_1_accuracy and accuracy_1>tree_2_accuracy):
        if(accuracy_2<tree_1_accuracy and accuracy_2<tree_2_accuracy):
            next_gen_hill_climb.append(newtree1)
            if (tree_1_accuracy> tree_2_accuracy):
                next_gen_hill_climb.append(tree_1)
            else:
                next_gen_hill_climb.append(tree_2)
    elif(accuracy_1<tree_1_accuracy and accuracy_1<tree_2_accuracy):
        if(accuracy_2>tree_1_accuracy and accuracy_2>tree_2_accuracy):
            next_gen_hill_climb.append(newtree2)
            if(tree_1_accuracy>tree_2_accuracy):
                next_gen_hill_climb.append(tree_1)
            else:
                next_gen_hill_climb.append(tree_2)
    elif(tries==10):
        next_gen_hill_climb.append(tree_1)
        next_gen_hill_climb.append(tree_2)
    else:
        tries+=1
        hill_climb(tree_1,tree_2)

    return to_be_copied


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
#Root=insert(False,0)
# printInorder(Root)
for i in range(100):
    expression=[]
    tree_expression = []
    Root=insert(False,0)
    root_array.append(Root)
    expression_array(Root)
    preorder_expression_array(Root)
    tree_population.append(expression)
    preorder_trees.append(tree_expression)
    # print(evaluateExpressionTree(Root))
# printInorder(Root)
print(preorder_trees)
classifier_values=[]
for i in preorder_trees:
    for j in i:
        if(type(j)!= int and type (j)!= str and type(j)!=float):
            print(j)
            print(i)
for tree in preorder_trees:
    classifier_values = []
    dsceval(tree)
    fitness=0
    # row_values=[]
    # # expressions_array=[]
    # # #print("The expression for tree is : ",exp)
    # trainaccuracy(tree,row_values)
    # #     expressions_array.append(express)
    # # #print("The trees are :")
    # # for i in expressions_array:
    # #     #print(i)
    # #print("The value for classifier are :")
    # #print(classifier_values)
    predicted_class=[]
    for i in classifier_values:
        if i > 0.5:
            predicted_class.append(2)
            fitness+=1
        else:
            predicted_class.append(4)
#print(predicted_class)
    accuracy=0
    for i in range(len(predicted_class)-1):
        if (predicted_class[i]==dataTrain.values[i,-1]):
            accuracy+=1
#print(accuracy)
    accuracy_list.append(accuracy)
    fitness_values.append(fitness)
    print("The accuracy percent is",((accuracy/len(dataTrain))*100))
#print(accuracy_list)
#print(tree_population)
# print(preorder_trees)
bubbleSort(accuracy_list,tree_population,preorder_trees,fitness_values)
next_gen=[]
for i in range(10):
    next_gen.append(tree_population[i])
#print(next_gen)
for_mutation=tree_population[-10:]
tree_population=tree_population[10:-10]
#print(for_mutation)
for i in range(len(for_mutation)):
    mutation(for_mutation[i])
#print(for_mutation)
preorder_trees=preorder_trees[10:-10]
#print(tree_population)
# print(preorder_trees)
half=len(preorder_trees)//2
std_crossover=preorder_trees[half:]
hill_climb_crossover=preorder_trees[:half]
next_gen_hill_climb=[]
next_gen_cross=[]
print(std_crossover)
while(len(std_crossover)!=0):
    tree_1=random.choice(std_crossover)
    std_crossover.remove(tree_1)
    tree_2=random.choice(std_crossover)
    std_crossover.remove(tree_2)
    standard_crossover(tree_1,tree_2)
print(next_gen_cross)
print("Hill climb is")
print(hill_climb_crossover)
while(len(hill_climb_crossover)!=0):
    tree_1=random.choice(hill_climb_crossover)
    hill_climb_crossover.remove(tree_1)
    tree_2=random.choice(hill_climb_crossover)
    hill_climb_crossover.remove(tree_2)
    hill_climb(tree_1,tree_2)
print(next_gen_hill_climb)
