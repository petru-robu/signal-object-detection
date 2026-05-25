# Signal Object Detection
Participants are asked to employ machine learning methods to detect (count) objects in noisy radio signals, where classes represent the number of objects. In this context, a question that arises is whether this task should be treated as a classification or regression task.

## Notes
- Images are grayscale, they are only colormapped


## Approach 1:
Engineer features from the images:

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

## Approach 2:
Standard models don't perform so well, also we are working with images, best approach: CNN.
For CNNs I am using pytorch and training locally on a computer with CUDA.

### First CNN try
My first CNN design was the following:

Conv(kernel_size=3, padding=1)
ReLU 
Pool(kernel_size=2, stride=2)

Conv(kernel_size=3, padding=1)
ReLU
Pool(kernel_size=2, stride=2)

Conv(kernel_size=3, padding=1)
ReLU
Pool(kernel_size=2, stride=2)

Flatten

Linear 
ReLU
Dropout(0.3)

Linear