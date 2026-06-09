import cv2
import random
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.feature import hog
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay


# ------- utility --------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)


def parse_args():
    # arg parser, because I train on kaggle notebook for GPUs and also to change params easily
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--output-dir", type=str, default="./output")
    parser.add_argument("--seed", type=int, default=42)

    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"Ignoring unknown args: {unknown}")
    return args


# ------- dataset --------
def load_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    # height=55, width=128
    img = img.astype(np.float32)

    # normalize
    img = img.astype(np.float32) / 255.0
    return img

def load_dataset(data_dir, seed):
    train_df = pd.read_csv(data_dir + "/train.csv")

    X = []
    for img_id in train_df["id"].to_numpy():
        img_path = "./data/train/" + img_id
        X.append(extract_all_features(img_path))

    X = np.array(X)
    y = train_df["label"].values
    print("Feature matrix shape:", X.shape)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )

    return X_train, X_val, y_train, y_val

# ------- feature extraction --------
def extract_hog_features(img):
    return hog(
        img,
        orientations=9,
        pixels_per_cell=(4, 8),
        cells_per_block=(2, 2),
        feature_vector=True,
    )


def extract_stat_features(img):
    feats = []
    feats.extend(
        [
            img.mean(),
            img.std(),
            img.min(),
            img.max(),
            np.median(img),
        ]
    )

    img_uint8 = (img * 255).astype(np.uint8)
    percentiles = np.percentile(img_uint8, [1, 5, 10, 25, 50, 75, 90, 95, 97, 99])
    feats.extend(percentiles)

    for q in [70, 80, 85, 90, 95, 97, 99]:
        thr = np.percentile(img_uint8, q)
        mask = img_uint8 > thr

        feats.extend(
            [
                mask.mean(),
                img_uint8[mask].mean() if mask.any() else 0.0,
                img_uint8[mask].std() if mask.any() else 0.0,
                img_uint8[mask].max() if mask.any() else 0.0,
            ]
        )

    return np.array(feats, dtype=np.float32)


def extract_component_features(img):
    feats = []
    img_uint8 = (img * 255).astype(np.uint8)

    for q in [70, 80, 85, 90, 95, 97, 99]:
        thr = np.percentile(img_uint8, q)
        mask = (img_uint8 > thr).astype(np.uint8)

        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

        areas = stats[1:, cv2.CC_STAT_AREA] if num_labels > 1 else np.array([])
        if len(areas) == 0:
            feats.extend([0, 0, 0, 0, 0])
        else:
            areas = areas.astype(np.float32)
            feats.extend(
                [len(areas), areas.sum(), areas.max(), areas.mean(), areas.std()]
            )
    return np.array(feats, dtype=np.float32)


def extract_projection_features(img):
    feats = []

    row_sum = img.sum(axis=1)
    col_sum = img.sum(axis=0)

    for arr in [row_sum, col_sum]:
        feats.extend(
            [
                arr.mean(),
                arr.std(),
                arr.min(),
                arr.max(),
                np.percentile(arr, 75),
                np.percentile(arr, 90),
                np.percentile(arr, 95),
                np.percentile(arr, 99),
            ]
        )

    return np.array(feats, dtype=np.float32)


def extract_all_features(image_path):
    img = load_image(image_path)
    hog_feats = extract_hog_features(img)
    stat_feats = extract_stat_features(img)
    comp_feats = extract_component_features(img)
    proj_feats = extract_projection_features(img)
    return np.concatenate([hog_feats, stat_feats, comp_feats, proj_feats])


# ----- models ------

def SVCClasif(X_train, X_val, y_train, y_val):
    print("--- Training Linear SVC ---")
    pipeline = make_pipeline(
        StandardScaler(), 
        LinearSVC(max_iter=30000, random_state=42)
    )
    
    # Simple param search for the C (regularization) parameter
    param_grid = {'linearsvc__C': [0.1, 0.3, 1.0]}
    
    search = GridSearchCV(pipeline, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    search.fit(X_train, y_train)
    
    best_model = search.best_estimator_
    y_pred = best_model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)

    print(f"Best Params: {search.best_params_}")
    print(f"Accuracy: {acc:.4f}\n")
    print("Classification report:")
    print(classification_report(y_val, y_pred))
    
    return best_model, acc

def ExtraTreesClasif(X_train, X_val, y_train, y_val):
    print("--- Training Extra Trees ---")
    model = ExtraTreesClassifier(random_state=42)
    
    # Simple param search for tree count and depth
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20]
    }
    
    search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    search.fit(X_train, y_train)
    
    best_model = search.best_estimator_
    y_pred = best_model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)

    print(f"Best Params: {search.best_params_}")
    print(f"Accuracy: {acc:.4f}\n")
    print("Classification report:")
    print(classification_report(y_val, y_pred))
    return best_model, acc

def RandomForestClassif(X_train, X_val, y_train, y_val):
    print("--- Training Random Forest ---")
    model = RandomForestClassifier(random_state=42)
    
    # search for tree count and depth
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20]
    }
    
    # n_jobs=-1 so it runs in parallel 
    search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy', n_jobs=-1) 
    search.fit(X_train, y_train)
    
    best_model = search.best_estimator_
    y_pred = best_model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)

    print(f"Best Params: {search.best_params_}")
    print(f"Accuracy: {acc:.4f}\n")
    print("Classification report:")
    print(classification_report(y_val, y_pred))
    return best_model, acc

def main():
    # args and settings
    args = parse_args()
    set_seed(args.seed)

    print("Loading dataset...")
    X_train, X_val, y_train, y_val = load_dataset(data_dir= args.data_dir, seed=args.seed)
    print("Loading dataset...")
    
    SVCClasif(X_train, X_val, y_train, y_val)
    ExtraTreesClasif(X_train, X_val, y_train, y_val)
    RandomForestClassif(X_train, X_val, y_train, y_val)
    
 
if __name__ == "__main__":
    main()
