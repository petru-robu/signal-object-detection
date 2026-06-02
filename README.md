# Signal Object Detection
Participants are asked to employ machine learning methods to detect (count) objects in noisy radio signals, where classes represent the number of objects.

Notes:
- Images are size 55x128
- Images are grayscale but colormapped with viridis (so not RGB)

## `1.py`: Approach 1 
This is the first thing I tried:

Engineer features from the images and put them into a model.

Intensity features:
- mean, std, min, max intesity
- intensity percentiles
- number of bright pixels above threshold
- fraction of bright pixels above threshold
- number of connected components above threshold
- largest, average, min, max connected component size
- edge pixel count
- edge density
- horizontal projection mean, std, min, max
- vertical projection mean, std, min, max
- resized raw pixel values (32 x 32)

Then try out different models on a training data split, I obtained:

- RandomForestClassifier: 29.8% accuracy
- ExtraTreesClassifier: 28.1% accuracy
- Scaler + SVM: 23.4% accuracy
 
## `3.py`: Approach 2
More agressive feature engineering:

Aside from the statistical features from above we can try:
- https://en.wikipedia.org/wiki/Histogram_of_oriented_gradients: HOG features tell something about directions of edges and things like this. It is useful for object detection.

- connected componenets features: looks for blocks of high intensity and analyze them. Something like finding ccs in a graph. Those blobs might be the objects we try to find. After finding components, we compute statistics on them and use them as features.

- projection features: row sum, column sum, etc. This might be useful as well because we see that our images contain vertical lines and they might be relevant for our object detection.

---
> Changing to CNNs: Standard models don't perform so well, also we are working with images, next best approach: CNNs

## `cnn1.py`: Approach 3
For CNNs I am using `pytorch` and training locally on a computer with CUDA. 

> Later I tried kaggle notebooks as well but it wasn't much better than locally somehow.

My first CNN design was the following:

```bash
Conv(kernel_size=3, padding=1)
Conv(kernel_size=3, padding=1)
Conv(kernel_size=3, padding=1)
nn.MaxPool2d(kernel_size=2, stride=2)

Flatten

Linear 
Linear
Dropout
```
```bash
Criterion: nn.CrossEntropyLoss()
Optimizer: optim.Adam(model.parameters(), lr=0.001)
```

Training this on 10-15 epochs got me at 30% accuracy.

From here I started to do some trial and error and find suitable CNN ideas.

## `cnn4.py`: Approach 4
I looked more into CNNs and decided to make the following changes / improvements:

1) Change the network to this:

```bash
4 layers of this block:
nn.Conv2d(kernel_size=3, padding=1, bias=False),
nn.BatchNorm2d(),
nn.SiLU(inplace=True),
nn.Conv2d(kernel_size=3, padding=1, bias=False),
nn.BatchNorm2d(),
nn.SiLU(inplace=True),
nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),

This was going: 1 -> 16 -> 32 -> 64 -> 128 

At the end:
nn.AdaptiveAvgPool2d((1, 1))

nn.Flatten(),
nn.Linear(128, 128),
nn.BatchNorm1d(128),
nn.SiLU(inplace=True),
nn.Dropout(dropout),
nn.Linear(128, num_classes),
```

```bash
criterion = nn.CrossEntropyLoss() # kept cross entropy loss
optimizer = optim.AdamW() # changed from Adam to AdamW
```

2) Added a scheduler:

```
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
```

3) Added data augumentation
Augument the data to make the network learn features in different circumstances.

- Gaussian noise
- Random shift: a small translation on x / y axes

4) Added class weights

- In the data there is a slight bias towards classes 1 and 2. So we need to take it into account.
- Also i tried to manually alter class weights by giving a coeff array to make the network favor a class or the other. This was based on the confusion matrix, seeing that i confused specific classes with one another. - Later I droped this idea


With this, I got a submission of 72%

### `cnn7.py`: Approach 5

In the meantime I edited the network from CNN4, but gave me no big difference.

Baiscally, here I did hyperparameter tuning with `optuna` on the CNN above.

I did 5 trials of 30 epochs each (because best acc was obtained usually around epoch 20-30).

Doing this, I got from 72% to 76%.

### `cnn8.py`: Approach 6 

The most logical thing to obtain better accuracies were pretrained networks like ResNet or AlexNet. So from here, a more accesible one to implement by hand were resnets: https://en.wikipedia.org/wiki/Residual_neural_network

Resnet takes into account earlier layers and gives better performance.

This gave me a submission of 80%.

### Next approach: Resnet + Optuna