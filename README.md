# Signal Object Detection
Participants are asked to employ machine learning methods to detect (count) objects in noisy radio signals, where classes represent the number of objects. In this context, a question that arises is whether this task should be treated as a classification or regression task.

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