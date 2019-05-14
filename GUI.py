from tkinter import *
import getData
import logistic_regression
import pandas as pd
import numpy as np 

root = Tk()
root.resizable(False, False)
root.title("")
result = []
W = 0
b = 0
train_costs = 0
first = True

def go_back():
    begin()

def do_train(lr, ages):
    global W, b, train_costs
    W, b, train_costs = logistic_regression.training(xtrain, ytrain, float(lr), int(ages))
    go_back()

def _train():
    global root
    root.destroy()
    root = Tk()
    root.title("")
    Frame(root, height=300, width=200).pack()
    Label(root, text="Learning rate").place(x=60, y=25)
    e1 = Entry(root, width=10)
    e1.place(x=60, y=55)
    Label(root, text="Epochs").place(x=60, y=105)
    e2 = Entry(root, width=10)
    e2.place(x=60, y=135)
    train = lambda: do_train(e1.get(), e2.get())
    Button(root, text="Train", width=10, command=train).place(x=60, y=215)
    

def _test():
    pYtest, cs = logistic_regression.logistic_regression(xtest,ytest,W,b)
    root2 = Toplevel()
    root2.title("")
    Frame(root2, height=100, width=200).pack()
    Label(root2, text="Classification rate").place(x=50, y=25)
    Label(root2, text=cs).place(x=50, y=65)

def _try():
    global root, result
    root.destroy()
    root = Tk()
    root.title("")
    Frame(root, height=300, width=200)
    heads = list(data)
    result = []
    for i in range(len(heads)-1):
        result.append(IntVar())
        C = Checkbutton(root,text=heads[i],variable=result[i],onvalue = 1, offvalue=0, height=1)
        C.grid(column = i%3, row = i%11)
    Button(root, text="SET", width=10, command=predict).grid(column = 1, row = 11)

def predict():
    global result
    res = []
    for n in result:
        res.append(n.get())
    res = np.array(res)
    #print(res)
    predict, cs = logistic_regression.logistic_regression(res,ytest,W,b)
    root2 = Toplevel()
    root2.title("")
    Frame(root2, height=100, width=200).pack()
    ans = ""
    if(predict == 1):
        ans = "Eres atractivo"
    else:
        ans = "No eres atractivo"
    Label(root2, text=ans).place(x=50, y=25)  
    root2.protocol("WM_DELETE_WINDOW", go_back)





def begin():
    global first
    global root
    
    if first:
        first = False
        b_frame(root)
        root.mainloop()
    else:
        root.destroy()
        root = Tk()
        root.title("")
        b_frame(root)

def b_frame(root):    
    Frame(root, height=300, width=200).pack()

    Button(root, text="Train", width=10, command=_train).place(x=60, y=55)
    Button(root, text="Test", width=10, command=_test).place(x=60, y=135)
    Button(root, text="Try", width=10, command=_try).place(x=60, y=215)



print("Fetching Data...")
data = getData.prepareData()
print("Preparing Data...")
train,test = getData.separateTrainTest(data, 75)
print("Distributing...")
xtrain,ytrain,xtest,ytest = getData.separateDataXY(train,test)

print("\nReady")
begin()
