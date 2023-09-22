#Import necessary packages
from tensorflow import keras
from pyimagesearch.models.Sudokunet import SudokuNet
from keras.optimizers import Adam
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
import argparse

#argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True, help="Path to output model after training")
args = vars(ap.parse_args())

#hyperparameters
INIT_LR = 1e-1
EPOCHS = 20
BS = 64

#loading dataset
print("[INFO] Accessing MNIST...")
((trainData, trainLabels), (testData, testLabels)) = mnist.load_data()

trainData = trainData.astype("float32") / 255.0
testData = testData.astype("float32") / 255.0

le = LabelBinarizer()
trainLabels = le.fit_transform(trainLabels)
testLabels = le.fit_transform(testLabels)

aug = ImageDataGenerator(
    rotation_range = 10,
    zoom_range = 0.05,
    width_shift = 0.1,
    height_shift = 0.1,
    shear_range = 0.15,
    horizontal_flip=False,
    fill_mode="nearest"
)

print("[INFO] Compiling model...")
opt = Adam(lr=INIT_LR)
model = SudokuNet.build(width=28, height=28, depth=1, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

print("[INFO] Training network...")
H = model.fit(aug.flow(trainData, trainLabels, batch_size=BS),
              validation_data=(testData, testLabels),
              epochs=EPOCHS,
              verbose=1)

print("[INFO] Evaluating network...")
predictions = model.predict(testData)
print(classification_report(testLabels.argmax(axis=1),
      predictions.argmax(axis=1),
      target_names=[str(x) for x in le.classes_]))

print("[INFO] Serializing digit classifying model...")
model.save(args["model"], save_format="h5")