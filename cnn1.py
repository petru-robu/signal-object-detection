# CNNS
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
import torch.nn.functional as F

import torchvision
from torchvision import transforms

from torchmetrics import Accuracy

# ------ Image manipulation stuff -------


def imshow(img):
    npimg = img.numpy()
    # convert RGB ro grayscale
    npimg = np.transpose(npimg, (1, 2, 0))
    npimg = npimg.mean(axis=2)

    plt.imshow(npimg, cmap='viridis')
    plt.show()


def load_image(image_path):
    """
        Load an image from path
    """

    # image size is 55, 128
    # tensor shape later is: 1, 128, 55
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    img = img.astype(np.float32) / 255.0
    return img

# ------ Dataset stuff -------


class ObjDetTorchDataset(Dataset):
    def __init__(self, csv_file, has_label=True, img_dir=None, transform=None):
        self.df = pd.read_csv(csv_file)
        self.has_label = has_label
        self.transform = transform
        self.img_dir = img_dir

        if img_dir is None:
            self.img_dir = './data/train' if "train" in csv_file else './data/test'

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        img_id = row["id"]

        img_path = self.img_dir + f"/{img_id}"
        image = load_image(img_path)

        # transform from np to tensor
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)

        if self.transform is not None:
            image = self.transform(image)

        if self.has_label:
            label = int(row["label"]) - 1
            label = torch.tensor(label, dtype=torch.long)
            return image, label

        return image, img_id


def load_dataset(data_dir="./data", batch_size=32, num_workers=2, val_split=0.2):
    """
        Loads the dataset but for torch required format, data and loader
    """
    # paths
    train_csv, train_img_dir = data_dir + "/train.csv", data_dir + "/train"
    test_csv, test_img_dir = data_dir + "/test.csv", data_dir + "/test"

    # torch data
    full_train_data = ObjDetTorchDataset(
        csv_file=train_csv,
        has_label=True,
        img_dir=train_img_dir
    )

    test_data = ObjDetTorchDataset(
        csv_file=test_csv,
        has_label=False,
        img_dir=test_img_dir
    )

    # split
    val_size = int(len(full_train_data) * val_split)
    train_size = len(full_train_data) - val_size

    train_data, val_data = random_split(
        full_train_data,
        [train_size, val_size]
    )

    # data loaders
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return test_data, train_data, val_data, test_loader, train_loader, val_loader


def check_dataset():
    # Loading dataset
    print("Loading dataset...")
    test_data, train_data, val_data, test_loader, train_loader, val_loader = load_dataset()
    images, labels = next(iter(train_loader))
    print("Dataset loaded succesfully!")
    print("Image batch shape:", images.shape)
    print("Label batch shape:", labels.shape)

    # show images
    dataiter = iter(train_loader)
    images, labels = next(dataiter)
    imshow(torchvision.utils.make_grid(images))


# ----- CNN stuff ------
class CNN(nn.Module):
    def __init__(self, in_channels, num_classes):
        """
            in_channels: number of channels in the input image, num_classes: no. of classes to predict (10)
            Here we define our network
        """
        super(CNN, self).__init__()

        # conv layer 1
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=16,
            kernel_size=3,
            padding=1
        )

        # conv layer 2
        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            padding=1
        )

        # conv layer 3
        self.conv3 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1
        )

        # pool layer
        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        # Input image: [B, 1, 128, 55]
        #
        # After conv1 + pool: [B, 16, 64, 27]
        # After conv2 + pool: [B, 32, 32, 13]
        # After conv3 + pool: [B, 64, 16, 6]
        #
        # flatten 1 size = 64 * 16 * 6
        self.fc1 = nn.Linear(
            64 * 16 * 6,
            128
        )

        # flatten 2
        self.fc2 = nn.Linear(
            128,
            num_classes
        )

        self.dropout = nn.Dropout(0.18)

    def forward(self, x):
        """
            Define the forward pass of the neural network
            x: input tensor, returns also a tensor
        """

        # conv -> relu -> pool
        x = F.relu(self.conv1(x))
        x = self.pool(x)

        # conv -> relu -> pool
        x = F.relu(self.conv2(x))
        x = self.pool(x)

        # conv -> relu -> pool
        x = F.relu(self.conv3(x))
        x = self.pool(x)

        x = x.reshape(x.shape[0], -1)

        # linearize -> relu
        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        # linearize
        x = self.fc2(x)

        return x


def main():
    # device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device is {device}!")
    # check_dataset()

    # Loading dataset
    print("Loading dataset...")
    test_data, train_data, val_data, test_loader, train_loader, val_loader = load_dataset()
    images, labels = next(iter(train_loader))
    print("Dataset loaded succesfully!")
    print("Image batch shape:", images.shape)
    print("Label batch shape:", labels.shape)

    # train 5 classes
    model = CNN(in_channels=1, num_classes=5).to(device)
    print("Model is: \n", model)

    # train, use cross entropy
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # one epoch = one pass through the dataset
    num_epochs = 10
    for epoch in range(num_epochs):
        print(f"Epoch [{epoch+1} / {num_epochs}]")

        # iterate over batches of data from train data
        for batch_idx, (data, targets) in enumerate(tqdm(train_loader)):
            data = data.to(device)
            targets = targets.to(device)

            # compute model prediction
            scores = model(data)

            # calculate the loss using criterion loss function
            loss = criterion(scores, targets)

            # clear last gradient
            optimizer.zero_grad()

            # gradient of loss
            loss.backward()

            # update model params based on loss
            optimizer.step()

    # Model is trained, now test accuracy on validation
    acc = Accuracy(task="multiclass", num_classes=5).to(device)
    model.eval()
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            acc.update(preds, labels)

    val_accuracy = acc.compute()
    print(f"Validation accuracy: {val_accuracy:.4f}")

    # Use model to predict test and save submission.csv
    model.eval()
    test_ids = []
    test_preds = []

    with torch.no_grad():
        for images, img_ids in tqdm(test_loader, desc="Predicting test"):
            images = images.to(device)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            # so convert predictions back to 1-based for the submission.
            preds = preds.cpu().numpy() + 1

            test_ids.extend(list(img_ids))
            test_preds.extend(preds.tolist())

    submission = pd.DataFrame({
        "id": test_ids,
        "label": test_preds
    })

    submission.to_csv("submission.csv", index=False)
    print("Saved submission.csv")


if __name__ == '__main__':
    main()
