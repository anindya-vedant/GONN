import warnings
warnings.filterwarnings('ignore')
import random
import pandas as pd
import numpy
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
# import sys

class Node:
    def __init__(self,data):
        self.data=data
        self.left=None
        self.right=None

def insert_1(isterminal, height): #features and arithmetic operations
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

def insert(isterminal,height,data=None): # fesures, arithematic operation, P functions...
    if(height==6 or isterminal==True):
        choices=[]
        choices.append(random.uniform(0.1,10.0))
        for i in range(len(dataTrain.values[0])-1):
            choices.append(i)
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
            isterminal=random.choice([True, False])
            root.left=insert_1(isterminal,height+1)
            isterminal=random.choice([True, False])
            root.right=insert(isterminal,height+1)
        else:
            choices=[]
            choices.append("P")
            choices.append(random.uniform(0.1,10.0))
            choices.append("+")
            choices.append("-")
            choices.append("*")
            choices.append("/")
            for i in range(len(dataTrain.values[0])-1):
                choices.append(i)
            if(height==5):
                choices.pop(0)
            node_data=random.choice(choices)
            root=Node(node_data)
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


def preorder_expression_array(root):
    if root:
        tree_expression.append(root.data)
        preorder_expression_array(root.left)
        preorder_expression_array(root.right)

def sig(x):
    return 1/(1+numpy.exp(-x))

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
    classifier_values=[]
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    c1 = 0
    c2 = 0
    l = []
    for i in range(len(dataTrain)):
        stin = stou[::]
        for k in range(len(stin)):
            if stin[k] not in ['+', '-', '*', '/'] and type(stin[k]) == int:
                stin[k] = (dataTrain.values[i, stin[k]])
        # print(i,k,stin)
        val = evaluatestack(stin)
        classifier_values.append(val)
    return classifier_values

def accuracy_cal(classifier_values):
    predicted_class=[]
    for i in classifier_values:
        if i > 0.5:
            predicted_class.append(2)
        else:
            predicted_class.append(4)
    accuracy = 0
    for i in range(len(predicted_class) - 1):
        if (predicted_class[i] == dataTrain.values[i, -1]):
            accuracy += 1
    ## print("The accuracy percent is",((accuracy/len(dataTrain))*100))

    return accuracy

def confusion_matrix_fn(classifier_values):
    predicted_class=[]
    for i in classifier_values:
        if i > 0.5:
            predicted_class.append(2)
        else:
            predicted_class.append(4)
    accuracy = 0
    for i in range(len(predicted_class) - 1):
        if (predicted_class[i] == dataTrain.values[i, -1]):
            accuracy += 1
    ## print("The accuracy percent is",((accuracy/len(dataTrain))*100))

    return predicted_class

def bubbleSort(arr,pre):
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
                pre[j], pre[j + 1] = pre[j + 1], pre[j]

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
    # if(tree_1[index]=="W"):
    #     index1=tree_2.index("W")
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

def hill_climb(tree_1,tree_2,tries,child_count):
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
    # if(tree_1[index]=="W"):
    #     index1=tree_2.index("W")
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
    classifier_values=dsceval(newtree1)
    accuracy_1=accuracy_cal(classifier_values)
    classifier_values=dsceval(newtree2)
    accuracy_2=accuracy_cal(classifier_values)
    # print(newtree1)
    # print(newtree2)
    # print(tree_1)
    # print(tree_2)
    # print("Child 1 accuracy is",accuracy_1)
    # print("Child 2 accuracy is",accuracy_2)
    # print("Parent 1 accuracy is",tree_1_accuracy)
    # print("Parent 2 accuracy is",tree_2_accuracy)

    if (accuracy_1>tree_1_accuracy and accuracy_1>tree_2_accuracy and child_count<2):
            ##print("Trial Number", tries)
            next_gen_hill_climb.append(newtree1)
            child_count=child_count+1
            ##print("Child 1 appended")
    if(child_count>0 and child_count<2):
        if(accuracy_2>tree_1_accuracy and accuracy_2>tree_2_accuracy):
            ##print("Trial Number", tries)
            next_gen_hill_climb.append(newtree2)
            child_count+=1
            ##print("Child 2 appended")
    # elif(accuracy_1<tree_1_accuracy and accuracy_1<tree_2_accuracy):
    #     if(accuracy_2>tree_1_accuracy and accuracy_2>tree_2_accuracy):
    #         print("Trial Number", tries)
    #         next_gen_hill_climb.append(newtree2)
    #         if(tree_1_accuracy>tree_2_accuracy):
    #             next_gen_hill_climb.append(tree_1)
    #             print("Child 2 and Parent 1 appended")
    #         else:
    #             next_gen_hill_climb.append(tree_2)
    #             print("Child 2 and Parent 1 appended")
    if(tries==10):
        ##print("Trial Number", tries)
        if (child_count==0):
            next_gen_hill_climb.append(tree_1)
            next_gen_hill_climb.append(tree_2)
        elif(child_count==1):
            if(tree_1_accuracy>tree_2_accuracy):
                next_gen_hill_climb.append(tree_1)
            else:
                next_gen_hill_climb.append(tree_2)
        # print("Parents appended")
    else:
        ##print("Trial Number", tries)
        tries=tries+1
        hill_climb(tree_1,tree_2,tries,child_count)

    return to_be_copied

# sys.setrecursionlimit(1500)

def test(stou):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    c1=0
    c2=0
    l = []

    for i in range(len(dataTest)):
        stin = stou[::]
        for k in range(len(stin)):
            if stin[k] not in ['+', '-', '*', '/'] and type(stin[k]) == int:
                stin[k] = dataTest.values[i, stin[k]]
        # print(i,k,stin)
        val = evaluatestack(stin)
        # print(val)
        l.append(sig(val))
        if val >= 0.5:
            if dataTest.values[i, -1] == 2:
                tp = tp + 1
                c1 = c1 + (sig(val) ** 2)
            else:
                fp = fp + 1
        else:
            if dataTest.values[i, -1] == 4:
                fn = fn + 1
            else:
                tn = tn + 1
                c2 = c2 + (sig(val) ** 2)

    c1 = c1 / (tp + fn + 1)
    c2 = c2 / (tn + fp + 1)
    print("Confusion Matrix", tp, fp, fn, tn)
    print("Accuracy: ", ((tp+tn)/(tp+tn+fp+fn))*100 )
    # print(tn)
    # print(fp)
    # print(fn)

    return (((2 * c1 * c2) / (c1 + c2)))

#lading the data set
dataset=pd.read_csv('wbc-dataset.csv', header=None)
dataset=dataset.drop(0,axis=1)
dataset=dataset.replace("?",numpy.NaN)
dataset.dropna(inplace=True)
# dataTrain,dataTest = train_test_split(dataset,test_size = 0.3) # 50-50, 60-40, 70-30
# dataTrain,dataTest=KFold(n_splits=10)
kf = KFold(n_splits=10,shuffle=False)
for train_index, test_index in kf.split(dataset):
    dataTrain, dataTest = dataset.iloc[train_index], dataset.iloc[test_index]
    features=dataTrain.iloc[:,:-1]
    classes=dataTrain.iloc[:,-1]

    actual_class = classes.tolist()  # The actual values with which compare our predicted work.

    accuracy_list=[]
    preorder_trees=[]
    # Root=insert(False,0)
    # inorder(Root)
    for i in range(100):
        expression=[]
        tree_expression = []
        Root=insert(False,0)
        preorder_expression_array(Root)
        preorder_trees.append(tree_expression)
    for tree in preorder_trees:
        classifier_values = dsceval(tree)
        accuracy_list.append(accuracy_cal(classifier_values))

    generation=0

    while(generation!=50):
        bubbleSort(accuracy_list,preorder_trees)
        next_gen=[]
        next_gen_preorder=[]
        for i in range(10):
            next_gen.append(preorder_trees[i])
        #print(next_gen)
        for_mutation=preorder_trees[-10:]
        # print(len(preorder_trees))
        preorder_trees=preorder_trees[10:-10]
        #print(for_mutation)
        for i in range(len(for_mutation)):
            mutation(for_mutation[i])
        #print(for_mutation)
        # print(len(preorder_trees))
        #print(tree_population)
        # print(preorder_trees)
        half=len(preorder_trees)//2
        std_crossover=preorder_trees[half:]
        hill_climb_crossover=preorder_trees[:half]
        next_gen_hill_climb=[]
        next_gen_cross=[]
        # print(std_crossover)
        while(len(std_crossover)!=0):
            tree_1=random.choice(std_crossover)
            std_crossover.remove(tree_1)
            tree_2=random.choice(std_crossover)
            std_crossover.remove(tree_2)
            standard_crossover(tree_1,tree_2)
        # print(hill_climb_crossover)
        # print(len(hill_climb_crossover))
        while(len(hill_climb_crossover)!=0):
            tries = 0
            child_count=0
            t1=random.choice(hill_climb_crossover)
            hill_climb_crossover.remove(t1)
            t2=random.choice(hill_climb_crossover)
            hill_climb_crossover.remove(t2)
            # print(hill_climb_crossover)
            # print(len(hill_climb_crossover))
            hill_climb(t1,t2,tries,child_count)
            # print("Len of hill climb,", len(next_gen_hill_climb))
        # print(next_gen_hill_climb)
        # print(len(next_gen))
        for i in next_gen_hill_climb:
            next_gen.append(i)
        # print(len(next_gen))
        for i in next_gen_cross:
            next_gen.append(i)
        # print(len(next_gen))
        for i in for_mutation:
            next_gen.append(i)
        # print(len(next_gen))
        accuracy_list=[]
        preorder_trees=next_gen[::]
        for i in preorder_trees:
            classifier_values=dsceval(i)
            accuracy_list.append(accuracy_cal(classifier_values))
        print(generation)
        generation+=1

    bubbleSort(accuracy_list, preorder_trees)
    best_tree_order = preorder_trees[0]
    test(best_tree_order)

    # Confusion Matrix

    acc_list_copy = accuracy_list[::]
    highest_acc = acc_list_copy.index(max(acc_list_copy))       # Index of the best fitness tree

    best_tree = preorder_trees[highest_acc]                     # Best Tree
    classifier=dsceval(best_tree)                                #Classifier value for best tree for 341 rows

    predicted_max = confusion_matrix_fn(classifier)                 # List of predicted values of the highest tree



    confusion_matrix = [0, 0, 0, 0]             # TP, FP, FN, TN

    for i in range(len(actual_class)):
        if actual_class[i] == predicted_max[i]:
            if actual_class[i] == 4:
                confusion_matrix[3] += 1
            if actual_class[i] == 2:
                confusion_matrix[0] += 1

        else:
            if actual_class[i] == 2 and predicted_max[i] == 4:
                confusion_matrix[2] += 1
            else:
                confusion_matrix[1] += 1

    print("classifier", len(classifier))
    print("actual_class", len(actual_class))
    print("predicted_max", len(predicted_max))
    print("data train", len(dataTrain))

    print("Confusion Matrix: ", confusion_matrix)
    print("Data Train: ", sum(confusion_matrix))
    print("The best accuracy", (max(accuracy_list)/len(dataTrain))*100)
    # print("Average accuracy", (sum(accuracy_list)/(len(dataTrain)**2))*100)
