from tkinter import *
import getData
import logistic_regression

root = Tk()
root.resizable(False, False)
root.title("")
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
    frame = Frame(root, height=300, width=200).pack()
    Label(frame, text="Learning rate").place(x=60, y=25)
    e1 = Entry(frame, width=10)
    e1.place(x=60, y=55)
    Label(frame, text="Epochs").place(x=60, y=105)
    e2 = Entry(frame, width=10)
    e2.place(x=60, y=135)
    train = lambda: do_train(e1.get(), e2.get())
    Button(frame, text="Train", width=10, command=train).place(x=60, y=215)
    

def _test():
    pYtest, cs = logistic_regression.logistic_regression(xtest,ytest,W,b)
    root2 = Toplevel()
    root2.title("")
    frame = Frame(root2, height=100, width=200).pack()
    Label(root2, text="Classification rate").place(x=50, y=25)
    Label(root2, text=cs).place(x=50, y=65)

def _try():
    global root
    root.destroy()

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
    frame = Frame(root, height=300, width=200).pack()

    Button(frame, text="Train", width=10, command=_train).place(x=60, y=55)
    Button(frame, text="Test", width=10, command=_test).place(x=60, y=135)
    Button(frame, text="Try", width=10, command=_try).place(x=60, y=215)
    return frame


print("Fetching Data...")
data = getData.prepareData()
print("Preparing Data...")
train,test = getData.separateTrainTest(data, 75)
print("Distributing...")
xtrain,ytrain,xtest,ytest = getData.separateDataXY(train,test)

print("\nReady")
begin()
