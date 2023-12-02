import time
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense

X_train = pd.read_csv("https://raw.githubusercontent.com/ereason/neural-network-iis-lab3/main/xtrain.csv", header=None)
Y_train = pd.read_csv("https://raw.githubusercontent.com/ereason/neural-network-iis-lab3/main/ytrain.csv", header=None)
X_test = pd.read_csv("https://raw.githubusercontent.com/ereason/neural-network-iis-lab3/main/xtest.csv", header=None)
Y_test = pd.read_csv("https://raw.githubusercontent.com/ereason/neural-network-iis-lab3/main/ytest.csv", header=None)



classifier = Sequential() # Инициализация НС
classifier.add(Dense(units = 16, activation = 'relu', input_dim = 30))
classifier.add(Dense(units = 8, activation = 'relu'))
classifier.add(Dense(units = 6, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

classifier.compile(optimizer = 'ftrl', loss = 'binary_crossentropy')
classifier.summary()

trainStart = time.time()
classifier.fit(X_train, Y_train, batch_size = 1, epochs = 25)
trainEnd = time.time()

Y_pred = classifier.predict(X_test) # подаём на вход обученной НС тестовый набор данных
Y_pred = [ 1 if y>=0.5 else 0 for y in Y_pred ]

total = 0
correct = 0
wrong = 0
for i in range(len(Y_pred)):
  total=total+1
  if(Y_test.at[i,0] == Y_pred[i]):
    correct=correct+1
  else:
    wrong=wrong+1

print("Total " + str(total))
print("Correct " + str(correct))
print("Wrong " + str(wrong))
print("Train time " + str(trainEnd - trainStart))