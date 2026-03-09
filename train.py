import os
import glob
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import cv2

# Import from your main.py
from main import preprocess_and_align, extract_rois, get_fused_vector

def load_data(folder_path, label):
    X, y = [],[]
    for img_path in glob.glob(os.path.join(folder_path, "*.jpg")):
        img = cv2.imread(img_path)
        aligned = preprocess_and_align(img)
        rois = extract_rois(aligned)
        features = get_fused_vector(rois)
        X.append(features)
        y.append(label)
    return X, y

print("Extracting features from Genuine IDs... (This may take a minute)")
X_gen, y_gen = load_data("dataset/genuine", 1)

print("Extracting features from Counterfeit IDs...")
X_fake, y_fake = load_data("dataset/counterfeit", 0)

X = np.array(X_gen + X_fake)
y = np.array(y_gen + y_fake)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nTraining Random Forest Model...")
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(f"\nModel Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
print(classification_report(y_test, y_pred, target_names=["Counterfeit", "Genuine"]))

joblib.dump(clf, "models/authenticity_model.pkl")
print("Model saved to 'models/authenticity_model.pkl'. Ready for presentation!")