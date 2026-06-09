import cv2
import numpy as np
import pandas as pd

from skimage.feature import hog

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def load_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    # height=55, width=128
    img = cv2.resize(img, (128, 55))
    img = img.astype(np.float32)

    # normalize to [0, 1]
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)

    return img


def extract_hog_features(img):
    return hog(
        img,
        orientations=9,
        pixels_per_cell=(4, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        transform_sqrt=True,
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

    percentiles = np.percentile(img, [1, 5, 10, 25, 50, 75, 90, 95, 97, 99])
    feats.extend(percentiles)

    for q in [70, 80, 85, 90, 95, 97, 99]:
        thr = np.percentile(img, q)
        mask = img > thr

        feats.extend(
            [
                mask.mean(),
                img[mask].mean() if mask.any() else 0.0,
                img[mask].std() if mask.any() else 0.0,
                img[mask].max() if mask.any() else 0.0,
            ]
        )

    return np.array(feats, dtype=np.float32)


def extract_component_features(img):
    feats = []

    img_uint8 = (img * 255).astype(np.uint8)

    for q in [70, 80, 85, 90, 95, 97, 99]:
        thr = np.percentile(img_uint8, q)
        mask = (img_uint8 > thr).astype(np.uint8)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )

        areas = stats[1:, cv2.CC_STAT_AREA] if num_labels > 1 else np.array([])

        if len(areas) == 0:
            feats.extend([0, 0, 0, 0, 0, 0, 0])
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


if __name__ == "__main__":
    train_df = pd.read_csv("./data/train.csv")

    X = []
    for img_id in train_df["id"].to_numpy():
        img_path = "./data/train/" + img_id
        X.append(extract_all_features(img_path))

    X = np.array(X)
    y = train_df["label"].values

    print("Feature matrix shape:", X.shape)

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # scale, because svc requires centered data
    model = make_pipeline(
        StandardScaler(), LinearSVC(C=0.3, max_iter=30000, random_state=42)
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_valid)

    print("Accuracy:", accuracy_score(y_valid, y_pred))

    print("Confusion matrix:")
    print(confusion_matrix(y_valid, y_pred))

    print("Classification report:")
    print(classification_report(y_valid, y_pred))
