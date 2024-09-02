from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
import pandas as pd
import numpy as np
from tkinter import *
from tkinter import messagebox

df = pd.read_csv("c:\\Users\\T3DCOMPUTER\\Downloads\\archive (2)\\Traffic_Dataset_ml.csv")

X = np.array(
    df[["Date", "CarCount", "BikeCount", "BusCount", "TruckCount", "Total"]].values
)
y = np.array(df["Traffic Situation"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# SVM
svm_model = svm.SVC()  # Thay đổi các tham số phù hợp
svm_model.fit(X_train, y_train)

# CART
cart_model = DecisionTreeClassifier(
    criterion="gini", max_depth=4, random_state=42
)  # Thay đổi các tham số phù hợp
cart_model.fit(X_train, y_train)

# Neural Network
nn_model = MLPClassifier()  # Thay đổi các tham số phù hợp
nn_model.fit(X_train, y_train)


# Sử dụng phương pháp học kết hợp Bagging
bagging = BaggingClassifier(base_estimator=svm.SVC(), n_estimators=10, random_state=0)
bagging.fit(X_train, y_train)

# form
form = Tk()
form.title("Dự đoán :")
form.geometry("11500x1000")


lable_ten = Label(form, text="Nhập thông tin:", font=("Arial Bold", 16), fg="Purple")
lable_ten.grid(row=1, column=1, padx=40, pady=10)

date = Label(form, text=" Date")
date.grid(row=2, column=1, padx=40, pady=10)
date1 = Entry(form)
date1.grid(row=2, column=2)

car = Label(form, text="Car count:")
car.grid(row=3, column=1, pady=10)
car1 = Entry(form)
car1.grid(row=3, column=2)

bike = Label(form, text="Bike count:")
bike.grid(row=4, column=1, pady=10)
bike1 = Entry(form)
bike1.grid(row=4, column=2)

bus = Label(form, text="Bus count:")
bus.grid(row=5, column=1, pady=10)
bus1 = Entry(form)
bus1.grid(row=5, column=2)

truck = Label(form, text="Truck count:")
truck.grid(row=6, column=1, pady=10)
truck1 = Entry(form)
truck1.grid(row=6, column=2)

total = Label(form, text="Total :")
total.grid(row=7, column=1, pady=10)
total1 = Entry(form)
total1.grid(row=7, column=2)


# dự đoán Cart theo test
y_pred_cart = cart_model.predict(X_test)
lbl1 = Label(form)
lbl1.grid(column=1, row=8)
lbl1.configure(
    text="Tỉ lệ dự đoán đúng của CART: "
    + "\n"
    + "Precision: "
    + str(precision_score(y_test, y_pred_cart, average="micro") * 100)
    + "%"
    + "\n"
    + "Recall: "
    + str(recall_score(y_test, y_pred_cart, average="micro") * 100)
    + "%"
    + "\n"
    + "F1-score: "
    + str(f1_score(y_test, y_pred_cart, average="micro") * 100)
    + "%"
    + "\n"
)


def dudoancart():
    date = date1.get()
    car = car1.get()
    bike = bike1.get()
    bus = bus1.get()
    truck = truck1.get()
    total = total1.get()
    if (
        (date == "")
        or (car == "")
        or (bike == "")
        or (bus == "")
        or (truck == "")
        or (total == "")
    ):
        messagebox.showinfo("Thông báo", "Bạn cần nhập đầy đủ thông tin!")
    else:
        X_dudoan = np.array([date, car, bike, bus, truck, total]).reshape(1, -1)
        y_kqua = cart_model.predict(X_dudoan)
        lbl5.configure(text=y_kqua)


button_cart = Button(form, text="Kết quả dự đoán theo CART", command=dudoancart)
button_cart.grid(row=9, column=1, pady=20)
lbl5 = Label(form, text="...")
lbl5.grid(column=2, row=9)


def khanangcart():
    y_cart = cart_model.predict(X_test)
    dem = 0
    for i in range(len(y_cart)):
        if y_cart[i] == y_test[i]:
            dem = dem + 1
    count = (dem / len(y_cart)) * 100
    lbl1.configure(text=count)


button_cart1 = Button(form, text="Khả năng dự đoán đúng ", command=khanangcart)
button_cart1.grid(row=10, column=1, padx=30)
lbl1 = Label(form, text="...")
lbl1.grid(column=2, row=10)


# dự đoán SVM theo test
y_pred_svm = svm_model.predict(X_test)
lbl2 = Label(form)
lbl2.grid(column=3, row=8)
lbl2.configure(
    text="Tỉ lệ dự đoán đúng của SVM: "
    + "\n"
    + "Precision: "
    + str(precision_score(y_test, y_pred_svm, average="micro") * 100)
    + "%"
    + "\n"
    + "Recall: "
    + str(recall_score(y_test, y_pred_svm, average="micro") * 100)
    + "%"
    + "\n"
    + "F1-score: "
    + str(f1_score(y_test, y_pred_svm, average="micro") * 100)
    + "%"
    + "\n"
)


def dudoansvm():
    date = date1.get()
    car = car1.get()
    bike = bike1.get()
    bus = bus1.get()
    truck = truck1.get()
    total = total1.get()
    if (
        (date == "")
        or (car == "")
        or (bike == "")
        or (bus == "")
        or (truck == "")
        or (total == "")
    ):
        messagebox.showinfo("Thông báo", "Bạn cần nhập đầy đủ thông tin!")
    else:
        X_dudoan = np.array([date, car, bike, bus, truck, total]).reshape(1, -1)
        y_kqua = svm_model.predict(X_dudoan)
        lbl6.configure(text=y_kqua)


button_svm = Button(form, text="Kết quả dự đoán theo SVM", command=dudoansvm)
button_svm.grid(row=9, column=3, pady=20)
lbl6 = Label(form, text="...")
lbl6.grid(column=4, row=9)


def khanangsvm():
    y_svm = svm_model.predict(X_test)
    dem = 0
    for i in range(len(y_svm)):
        if y_svm[i] == y_test[i]:
            dem = dem + 1
    count = (dem / len(y_svm)) * 100
    lbl2.configure(text=count)


button_svm1 = Button(form, text="Khả năng dự đoán đúng ", command=khanangsvm)
button_svm1.grid(row=10, column=3, padx=30)
lbl2 = Label(form, text="...")
lbl2.grid(column=4, row=10)

# dự đoán MLP theo test
y_pred_nn = nn_model.predict(X_test)
lbl3 = Label(form)
lbl3.grid(column=1, row=11)
lbl3.configure(
    text="Tỉ lệ dự đoán đúng của MLP: "
    + "\n"
    + "Precision: "
    + str(precision_score(y_test, y_pred_nn, average="micro") * 100)
    + "%"
    + "\n"
    + "Recall: "
    + str(recall_score(y_test, y_pred_nn, average="micro") * 100)
    + "%"
    + "\n"
    + "F1-score: "
    + str(f1_score(y_test, y_pred_nn, average="micro") * 100)
    + "%"
    + "\n"
)


def dudoannn():
    date = date1.get()
    car = car1.get()
    bike = bike1.get()
    bus = bus1.get()
    truck = truck1.get()
    total = total1.get()
    if (
        (date == "")
        or (car == "")
        or (bike == "")
        or (bus == "")
        or (truck == "")
        or (total == "")
    ):
        messagebox.showinfo("Thông báo", "Bạn cần nhập đầy đủ thông tin!")
    else:
        X_dudoan = np.array([date, car, bike, bus, truck, total]).reshape(1, -1)
        y_kqua = nn_model.predict(X_dudoan)
        lbl7.configure(text=y_kqua)


button_nn = Button(form, text="Kết quả dự đoán theo MLP", command=dudoannn)
button_nn.grid(row=12, column=1, pady=20)
lbl7 = Label(form, text="...")
lbl7.grid(column=2, row=12)


def khanangnn():
    y_nn = nn_model.predict(X_test)
    dem = 0
    for i in range(len(y_nn)):
        if y_nn[i] == y_test[i]:
            dem = dem + 1
    count = (dem / len(y_nn)) * 100
    lbl3.configure(text=count)


button_nn1 = Button(form, text="Khả năng dự đoán đúng ", command=khanangnn)
button_nn1.grid(row=13, column=1, padx=30)
lbl3 = Label(form, text="...")
lbl3.grid(column=2, row=13)


# dự đoán bagging theo test
y_pred_bagging = bagging.predict(X_test)
lbl4 = Label(form)
lbl4.grid(column=3, row=11)
lbl4.configure(
    text="Tỉ lệ dự đoán đúng của Bagging: "
    + "\n"
    + "Precision: "
    + str(precision_score(y_test, y_pred_bagging, average="micro") * 100)
    + "%"
    + "\n"
    + "Recall: "
    + str(recall_score(y_test, y_pred_bagging, average="micro") * 100)
    + "%"
    + "\n"
    + "F1-score: "
    + str(f1_score(y_test, y_pred_bagging, average="micro") * 100)
    + "%"
    + "\n"
)


def dudoanbagging():
    date = date1.get()
    car = car1.get()
    bike = bike1.get()
    bus = bus1.get()
    truck = truck1.get()
    total = total1.get()
    if (
        (date == "")
        or (car == "")
        or (bike == "")
        or (bus == "")
        or (truck == "")
        or (total == "")
    ):
        messagebox.showinfo("Thông báo", "Bạn cần nhập đầy đủ thông tin!")
    else:
        X_dudoan = np.array([date, car, bike, bus, truck, total]).reshape(1, -1)
        y_kqua = bagging.predict(X_dudoan)
        lbl8.configure(text=y_kqua)


button_bagging = Button(
    form, text="Kết quả dự đoán theo Bagging", command=dudoanbagging
)
button_bagging.grid(row=12, column=3, pady=20)
lbl8 = Label(form, text="...")
lbl8.grid(column=4, row=12)


def khanangbagging():
    y_bagging = bagging.predict(X_test)
    dem = 0
    for i in range(len(y_bagging)):
        if y_bagging[i] == y_test[i]:
            dem = dem + 1
    count = (dem / len(y_bagging)) * 100
    lbl4.configure(text=count)


button_bagging1 = Button(form, text="Khả năng dự đoán đúng ", command=khanangbagging)
button_bagging1.grid(row=13, column=3, padx=30)
lbl4 = Label(form, text="...")
lbl4.grid(column=4, row=13)

form.mainloop()
