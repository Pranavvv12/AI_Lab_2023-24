# Ex.No: 13 Learning – Use Supervised Learning  
### DATE:                                                                            
### REGISTER NUMBER : 
### AIM: 
To write a program to train the classifier for Leaf Detection
###  Algorithm:

### Program:
```Import Libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from skimage.io import imread
from skimage.transform import resize
import os
import cv2

data_dir = 'path/to/your/leaf_dataset'
classes = os.listdir(data_dir)
image_data = []
labels = []

for class_name in classes:
    class_path = os.path.join(data_dir, class_name)
    for img_name in os.listdir(class_path):
        try:
            img_path = os.path.join(class_path, img_name)
            img = imread(img_path)
            img_resized = resize(img, (128, 128))  # Resize to a fixed size
            image_data.append(img_resized.flatten())  # Flatten to 1D
            labels.append(class_name)
        except Exception as e:
            print(f"Error reading {img_name}: {e}")

X = np.array(image_data)
y = np.array(labels)

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

classifier = DecisionTreeClassifier(random_state=42)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=le.classes_)

print("Accuracy:", accuracy)
print("Classification Report:\n", report)```

### Output:

### Result:
Thus the system was trained successfully and the prediction was carried out.
