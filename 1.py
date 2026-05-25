import cv2 
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

def load_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = img.astype(np.float32)
    return img

def extract_pixel_features(img, size=(32, 32)):
    resized = cv2.resize(img, size)
    resized = resized / 255.0
    return resized.flatten()

def extract_features(img):
    features = {}

    features["mean_intensity"] = np.mean(img)
    features["std_intensity"] = np.std(img)
    features["min_intensity"] = np.min(img)
    features["max_intensity"] = np.max(img)

    percentiles = [5, 10, 25, 50, 75, 90, 95]
    perc_values = np.percentile(img, percentiles)

    for p, v in zip(percentiles, perc_values):
        features[f"percentile_{p}"] = v

    threshold = 200
    bright_mask = img > threshold

    features["num_bright_pixels"] = np.sum(bright_mask)
    features["bright_pixel_fraction"] = np.mean(bright_mask)

    labeled = label(bright_mask)
    regions = regionprops(labeled)

    component_areas = [region.area for region in regions]

    features["num_connected_components"] = len(component_areas)

    if len(component_areas) > 0:
        features["largest_component_size"] = np.max(component_areas)
        features["average_component_size"] = np.mean(component_areas)
        features["median_component_size"] = np.median(component_areas)
        features["min_component_size"] = np.min(component_areas)
        features["max_component_size"] = np.max(component_areas)
    else:
        features["largest_component_size"] = 0
        features["average_component_size"] = 0
        features["median_component_size"] = 0
        features["min_component_size"] = 0
        features["max_component_size"] = 0

    img_uint8 = img.astype(np.uint8)
    edges = cv2.Canny(img_uint8, threshold1=100, threshold2=200)
    edge_pixels = edges > 0

    features["num_edge_pixels"] = np.sum(edge_pixels)
    features["edge_density"] = np.mean(edge_pixels)

    horizontal_projection = np.sum(bright_mask, axis=1)
    vertical_projection = np.sum(bright_mask, axis=0)

    features["horizontal_projection_mean"] = np.mean(horizontal_projection)
    features["horizontal_projection_std"] = np.std(horizontal_projection)
    features["horizontal_projection_min"] = np.min(horizontal_projection)
    features["horizontal_projection_max"] = np.max(horizontal_projection)

    features["vertical_projection_mean"] = np.mean(vertical_projection)
    features["vertical_projection_std"] = np.std(vertical_projection)
    features["vertical_projection_min"] = np.min(vertical_projection)
    features["vertical_projection_max"] = np.max(vertical_projection)

    handcrafted_features = np.array(
        [float(features[feat]) for feat in features],
        dtype=np.float32
    )

    pixel_features = extract_pixel_features(img, size=(32, 32))

    combined_features = np.concatenate([
        handcrafted_features,
        pixel_features
    ])

    return combined_features

if __name__ == '__main__':
    df = pd.read_csv('./data/train.csv')
    img_ids = df['id'].to_numpy()

    print(df["label"].value_counts())
    print(df["label"].value_counts(normalize=True))

    X, y = [], []

    for img_id in img_ids:
        img_path = './data/train/' + img_id

        img = load_image(img_path)
        features = extract_features(img)

        X.append(features)

        label_value = df.loc[df["id"] == img_id, "label"].iloc[0]
        y.append(label_value)

    X = np.array(X)
    y = np.array(y)

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=6,
        stratify=y
    )

    # --- Random Forest ---
    rfc = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=6,
        class_weight="balanced"
    )
    rfc.fit(X_train, y_train)
    y_val_pred = rfc.predict(X_val)
    acc = accuracy_score(y_val, y_val_pred)
    cm = confusion_matrix(y_val, y_val_pred)
    print('Accuracy: ', acc)
    print('CM: \n', cm)
    print(classification_report(y_val, y_val_pred))

    # --- Extra Trees Classifier ---
    # etc = ExtraTreesClassifier(
    #     n_estimators=500,
    #     random_state=6,
    #     class_weight="balanced",
    #     n_jobs=-1
    # )
    # etc.fit(X_train, y_train)
    # y_val_pred = etc.predict(X_val)
    # acc = accuracy_score(y_val, y_val_pred)
    # cm = confusion_matrix(y_val, y_val_pred)
    # print('Accuracy: ', acc)
    # print('CM: \n', cm)
    # print(classification_report(y_val, y_val_pred))

    # --- SVM with scaling ---
    # svm = make_pipeline(
    #     StandardScaler(),
    #     SVC(kernel="rbf", C=10, gamma="scale", class_weight="balanced")
    # )
    # svm.fit(X_train, y_train)
    # y_val_pred = svm.predict(X_val)
    # print("Accuracy:", accuracy_score(y_val, y_val_pred))
    # print(confusion_matrix(y_val, y_val_pred))
    # print(classification_report(y_val, y_val_pred))

    # submission
    X_test = []
    df_test = pd.read_csv('./data/test.csv')
    img_ids_test = df_test['id'].to_numpy()
    for img_id in img_ids_test:
        img_path = './data/test/' + img_id

        img = load_image(img_path)
        features = extract_features(img)

        X_test.append(features)

    X_test = np.array(X_test)

    y_pred = rfc.predict(X_test)

    submission = pd.DataFrame({
        "id": img_ids_test,
        "label": y_pred
    })

    submission.to_csv("submission.csv", index=False)

    print(submission.head())
    print("Saved predictions to submission.csv")