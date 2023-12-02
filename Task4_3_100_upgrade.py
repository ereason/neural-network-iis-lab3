import time
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPool2D
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import BatchNormalization
from keras.initializers import RandomNormal, Constant
from keras import optimizers
from keras import datasets
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import (to_categorical)



(train_images, train_labels), (test_images, test_labels) = datasets.cifar100.load_data()
train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0


train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

train_datagen = ImageDataGenerator(
    rotation_range=20,
    horizontal_flip=True,
)

X_train, X_validation, y_train, y_validation = train_test_split(
    train_images,
    train_labels,
    test_size=0.2,
    random_state=93,
)

train_datagen.fit(X_train)

############################################################################################
model = Sequential()

model.add(Conv2D(128, (3, 3), padding='same', input_shape=(32, 32, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(128, (3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(256, (3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(256, (3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(256, (3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(256, (3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(
    BatchNormalization(
        momentum=0.95,
        epsilon=0.005,
        beta_initializer=RandomNormal(mean=0.0, stddev=0.05),
        gamma_initializer=Constant(value=0.9)
    )
)

model.add(Dense(100, activation='softmax'))


model.summary()


model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.legacy.RMSprop(learning_rate=1e-4),
              metrics=['accuracy'])


model.load_weights("./savedModel100.keras")
test_loss, test_acc = model.evaluate(test_images, test_labels)

print(test_acc)
