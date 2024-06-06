# ai-codes
pgm1
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

ds = pd.read_excel("Book1.xlsx")
print(ds.shape)
print(ds)

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
ds['Age'] = imputer.fit_transform(ds[['Age']])
ds['Salary'] = imputer.fit_transform(ds[['Salary']])
print(ds)

imputer = SimpleImputer(strategy='most_frequent')
ds['Purchased'] = imputer.fit_transform(ds[['Purchased']]).ravel()
print(ds)

le = LabelEncoder()
ds['Purchased'] = le.fit_transform(ds[['Purchased']])
ds['Country'] = le.fit_transform(ds[['Country']])
print(ds)

xtrain, xtest = train_test_split(ds, test_size=0.3, random_state=0)
print('training data')
print(xtrain)
print('testing data')
print(xtest)

norm = MinMaxScaler()
xtrain_norm = norm.fit_transform(xtrain)
xtest_norm = norm.fit_transform(xtest)
print(xtrain_norm)
print(xtest_norm)


pgm2


import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


x = np.array([5, 15, 25, 35 , 45, 55]).reshape(-1, 1)
y = np.array([3, 13, 20, 37, 49, 51])

model = LinearRegression()
model.fit(x, y)

print(model.intercept_)
print(model.coef_)

y_pred = model.predict(x)

x_new = np.arange(5).reshape(-1, 1)
y_pred_new =   model.predict(x_new)

plt.scatter(x, y, label="Original data")
plt.scatter(x, y_pred, label="predicted data")
plt.plot(x, y_pred, label="best fit line")
plt.scatter(x_new, y_pred_new, label="New predicted data")

plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()


pgm3
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

ds = pd.read_csv("diab.csv")
print(ds)
features = ds[['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age']]
target = ds.label

train_x, test_x, train_y, test_y = train_test_split(features, target, test_size=0.25, random_state=1)
model =LogisticRegression()
model.fit(train_x,train_y)
y_pred = model.predict(test_x)

cnf = confusion_matrix(test_y, y_pred)
print(cnf)

c_r = classification_report(test_y, y_pred, target_names=['with diab', 'without diab'])
print(c_r)


pgm4

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
# sthwpcpv
df = pd.read_csv('pgm4_c.csv')
print(df.head())
feature_cols = ['slno', 'temperature', 'humidity', 'wind', 'precipitation', 'cloud', 'visibility']
features = df[feature_cols]
target = df.pressure

x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=1)
model = DecisionTreeClassifier(criterion="entropy", random_state=0)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
print(y_pred)
acc = accuracy_score(y_test, y_pred)
print(acc)

plt.subplots(figsize=(20, 20))
plot_tree(model, fontsize=20)
plt.show()


prog 5
weather = ['sunny', 'overcast', 'rainy', 'overcast', 'rainy', 'overcast', 'rainy']
temp = ['hot', 'mild', 'cool', 'mild', 'cool', 'mild', 'cool']
play = ['yes', 'yes', 'no', 'yes', 'no', 'yes', 'no']

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
weather_encoded = le.fit_transform(weather)
print(weather_encoded)
temp_encoded = le.fit_transform(temp)
print(temp_encoded)
decieder = le.fit_transform(play)
print(decieder)
features = list(zip(weather_encoded, temp_encoded))
print(features)

from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=3)
model.fit(features, decieder)

predicted1 = model.predict([[0, 2]])
predicted2 = model.predict([[2, 1]])
print(predicted1)
print(predicted2)



pgm7
from nltk import word_tokenize

sentence = "text Text teXt yo Yo oY oyo y"
words = word_tokenize(sentence.lower())
print(words)
hashMap = {}

for word in words:
    if word in hashMap:
        hashMap[word] += 1
    else:
        hashMap[word] = 1

for word, count in hashMap.items():
    print(word, count)




    pgm8
#ttime series date and passengers

import pmdarima
import pandas as pd
from matplotlib import pyplot as plt


df = pd.read_csv("airp.csv")
print(df.head())
print(df.tail())

df['Month'] = pd.to_datetime(df['Month'], format="%Y-%m")
df.index = df['Month']
print(df.head())
del df['Month']
print(df.head())

df.plot()
plt.show()

df['Month'] = df.index

train = df[df['Month'] < pd.to_datetime('1951-08', format='%Y-%m')]
del train['Month']
print(train)

test = df[df['Month'] >= pd.to_datetime('1951-08', format='%Y-%m')]
del test['Month']
print(test)

plt.plot(train, color='black')
plt.plot(test, color='red')
plt.title("Train/test split for passenger data")
plt.xlabel("Date")
plt.ylabel("Passengers")
plt.show()

model = pmdarima.auto_arima(train)
prediction = model.predict(n_periods = 5)
prediction_df = pd.DataFrame(prediction, index=test.index, columns=['test'])
print(prediction_df)




pgm10

import speech_recognition as sr
r = sr.Recognizer()
with sr.Microphone() as src:
	print('listening...')
	audio = r.listen(src)
	text = r.recognize_google(audio)
	print(text)
