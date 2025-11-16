# import the necessary packages
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
import config
from imutils import paths
import numpy as np
import pickle
import random
import os

# load the VGG19 network and initialize the label encoder
print("[INFO] loading network...")
model = VGG19(weights="imagenet", include_top=False)
le = None

# loop over the data splits
for split in (config.TRAIN, config.TEST, config.VAL):
    # grab all image paths in the current split
    print("[INFO] processing '{} split'...".format(split))
    p = os.path.sep.join([config.BASE_PATH, split])
    imagePaths = list(paths.list_images(p))
    
    # extract the class labels from the file paths
    labels = [p.split(os.path.sep)[-2] for p in imagePaths]
    
    # if the label encoder is None, create it
    if le is None:
        le = LabelEncoder()
        le.fit(labels)
    
    # open the output CSV file for writing
    csvPath = os.path.sep.join([config.BASE_CSV_PATH, f"{split}.csv"])
    csv = open(csvPath, "w")
    
    # loop over the images in batches
    for (b, i) in enumerate(range(0, len(imagePaths), config.BATCH_SIZE)):
        # extract the batch of images and labels
        print(f"[INFO] processing batch {b + 1}/"
              f"{int(np.ceil(len(imagePaths) / float(config.BATCH_SIZE)))}")
        batchPaths = imagePaths[i:i + config.BATCH_SIZE]
        batchLabels = le.transform(labels[i:i + config.BATCH_SIZE])
        batchImages = []
        
        # loop over the images in the current batch
        for imagePath in batchPaths:
            print(imagePath)
            image = load_img(imagePath, target_size=(224, 224))
            image = img_to_array(image)
            image = np.expand_dims(image, axis=0)
            image = preprocess_input(image)
            batchImages.append(image)
        
        # pass the images through the network and use the outputs as features
        batchImages = np.vstack(batchImages)
        features = model.predict(batchImages, batch_size=config.BATCH_SIZE)
        features = features.reshape((features.shape[0], 7 * 7 * 512))
        
        # loop over the class labels and extracted features
        for (label, vec) in zip(batchLabels, features):
            vec = ",".join([str(v) for v in vec])
            csv.write(f"{label},{vec}\n")
    
    # close the CSV file
    csv.close()

# serialize the label encoder to disk
f = open(config.LE_PATH, "wb")
f.write(pickle.dumps(le))
f.close()