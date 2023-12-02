import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from keras import layers

## Набор данных Auto MPG
np.set_printoptions(
    precision=3,
    suppress=True
)
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
column_names = [
    'MPG',
    'Cylinders',
    'Displacement',
    'Horsepower',
    'Weight',
    'Acceleration',
    'Model Year',
    'Origin'
]
raw_dataset = pd.read_csv(
    url,
    names=column_names,
    na_values='?',
    comment='\t',
    sep=' ',
    skipinitialspace=True
)
dataset = raw_dataset.copy()
## dataset.tail()

## Очистить данные
dataset.isna().sum()
dataset = dataset.dropna()
dataset['Origin'] = dataset['Origin'].map({
    1: 'USA',
    2: 'Europe',
    3: 'Japan',
})
dataset = pd.get_dummies(
    dataset,
    columns=['Origin'],
    prefix='',
    prefix_sep=''
)
## dataset.tail()

## Разделите данные на обучающие и тестовые наборы
train_dataset = dataset.sample(
    frac=0.8,
    random_state=0
)

## Проверить данные
test_dataset = dataset.drop(train_dataset.index)

sns.pairplot(
    train_dataset[['MPG', 'Cylinders', 'Displacement', 'Weight']],
    diag_kind='kde',
)
train_dataset.describe().transpose()

## Отделить функции от меток
train_features = train_dataset.copy().astype(np.float32)
test_features = test_dataset.copy().astype(np.float32)

train_labels = train_features.pop('MPG')
test_labels = test_features.pop('MPG')

## Нормализация
train_dataset.describe().transpose()[['mean', 'std']]

## Слой нормализации
normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(np.array(train_features))
# print(normalizer.mean.numpy())

first = np.array(train_features[:1])

# with np.printoptions(precision=2, suppress=True):
#     print('First example:', first)
#     print()
#     print('Normalized:', normalizer(first).numpy())


## Линейная регрессия

## Линейная регрессия с одной переменной ########################################
horsepower = np.array(train_features['Horsepower'])

horsepower_normalizer = layers.Normalization(
    input_shape=[1, ],
    axis=None
)
horsepower_normalizer.adapt(horsepower)
horsepower_model = tf.keras.Sequential([
    horsepower_normalizer,
    layers.Dense(units=1)
])

horsepower_model.summary()
horsepower_model.predict(horsepower[:10])

horsepower_model.compile(
    optimizer=tf.optimizers.legacy.Adam(learning_rate=0.1),
    loss='mean_absolute_error'
)

timeStart = time.time()
history = horsepower_model.fit(
    train_features['Horsepower'],
    train_labels,
    epochs=70,
    # Suppress logging.
    verbose=0,
    # Calculate validation results on 20% of the training data.
    validation_split=0.2
)
timeEnd = time.time()

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()


def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.ylim([0, 10])
    plt.xlabel('Epoch')
    plt.ylabel('Error [MPG]')
    plt.legend()
    plt.grid(True)
    plt.show()


plot_loss(history)

test_results = {}
result = horsepower_model.evaluate(
    test_features['Horsepower'],
    test_labels,
    verbose=0,
)
test_results['horsepower_model'] = [result, timeEnd-timeStart]


x = tf.linspace(0.0, 250, 251)
y = horsepower_model.predict(x)

def plot_horsepower(x, y):
    plt.scatter(train_features['Horsepower'], train_labels, label='Data')
    plt.plot(x, y, color='k', label='Predictions')
    plt.xlabel('Horsepower')
    plt.ylabel('MPG')
    plt.legend()
    plt.show()


plot_horsepower(x, y)

# Линейная регрессия с несколькими входами ########################################
linear_model = tf.keras.Sequential([
    normalizer,
    layers.Dense(units=1)
])

linear_model.predict(train_features[:10])
linear_model.layers[1].kernel
linear_model.compile(
    optimizer=tf.optimizers.legacy.Adam(learning_rate=0.1),
    loss='mean_absolute_error'
)

timeStart = time.time()
history = linear_model.fit(
    train_features,
    train_labels,
    epochs=70,
    # Suppress logging.
    verbose=0,
    # Calculate validation results on 20% of the training data.
    validation_split=0.2
)
timeEnd = time.time()

plot_loss(history)

result = linear_model.evaluate(
    test_features,
    test_labels,
    verbose=0
)
test_results['linear_model'] = [result, timeEnd-timeStart]

## Регрессия с глубокой нейронной сетью (DNN) ########################################
def build_and_compile_model(norm):
    model = keras.Sequential([
        norm,
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])

    model.compile(
        loss='mean_absolute_error',
        optimizer=tf.keras.optimizers.legacy.Adam(0.001)
    )

    return model


## Регрессия с использованием DNN и одного входа ########################################
dnn_horsepower_model = build_and_compile_model(horsepower_normalizer)

dnn_horsepower_model.summary()

timeStart = time.time()
history = dnn_horsepower_model.fit(
    train_features['Horsepower'],
    train_labels,
    validation_split=0.2,
    verbose=0,
    epochs=70,
)
timeEnd = time.time()

plot_loss(history)

x = tf.linspace(0.0, 250, 251)
y = dnn_horsepower_model.predict(x)

plot_horsepower(x, y)

result = dnn_horsepower_model.evaluate(
    test_features['Horsepower'],
    test_labels,
    verbose=0
)
test_results['dnn_horsepower_model'] = [result, timeEnd-timeStart]

## Регрессия с использованием DNN и нескольких входных данных ########################################
dnn_model = build_and_compile_model(normalizer)
dnn_model.summary()

timeStart = time.time()
history = dnn_model.fit(
    train_features,
    train_labels,
    validation_split=0.2,
    verbose=0,
    epochs=100
)
timeEnd = time.time()

plot_loss(history)

result = dnn_model.evaluate(
    test_features,
    test_labels,
    verbose=0
)
test_results['dnn_model'] = [result, timeEnd-timeStart]

## Представление ########################################
print(pd.DataFrame(test_results, index=['Mean absolute error [MPG]', 'fit_time']).T)
#print(test_results)

## Делать предсказания
test_predictions = dnn_model.predict(test_features).flatten()

a = plt.axes(aspect='equal')
plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [MPG]')
plt.ylabel('Predictions [MPG]')
lims = [0, 50]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)
plt.show()

error = test_predictions - test_labels
plt.hist(error, bins=25)
plt.xlabel('Prediction Error [MPG]')
_ = plt.ylabel('Count')
plt.show()

dnn_model.save('dnn_model')

reloaded = tf.keras.models.load_model('dnn_model')

test_results['reloaded'] = reloaded.evaluate(
    test_features, test_labels, verbose=0)

print(pd.DataFrame(test_results, index=['Mean absolute error [MPG]','fit_time']).T)

pd.options.display.width = 0
print(pd.DataFrame(dataset).describe())