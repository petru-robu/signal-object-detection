import cv2
import numpy as np
import pandas as pd
from skimage.feature import hog
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def load_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = img.astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    
    return img

def extract_hog_features(img):
    features = hog(
        img,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        transform_sqrt=True,
        feature_vector=True
    )

    return features

if __name__ == '__main__':
    train_df = pd.read_csv('./data/train.csv')
    img_ids = train_df['id'].to_numpy()
    X, y = [], []
    for img_id in img_ids:
        img_path = './data/train/' + img_id
        img = load_image(img_path)
        features = extract_hog_features(img)
        X.append(features)

    X = np.array(X)
    y = train_df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=6,
        stratify=y
    )

    model = make_pipeline(
        StandardScaler(),
        LinearSVC(C=1.0, class_weight=None, max_iter=10000)
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
