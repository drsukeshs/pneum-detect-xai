#imports
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
import matplotlib.image as mpimg
from tensorflow import keras
import numpy as np
from tensorflow.keras.preprocessing import image

#kaggle dataset import & extraction 
from google.colab import files
files.upload()
!mkdir -p ~/.kaggle
!mv kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
!unzip chest-xray-pneumonia.zip -d ./data

#image preprocessing
IMG_SIZE=(224,224)
BATCH_SIZE=32

def preprocess(image, label):
  image = tf.image.resize(image, IMG_SIZE)
  image = tf.cast(image, tf.float32)
  image = image / 255.0
  return image, label

#set datasets for training, validation and testing
train_ds = tf.keras.utils.image_dataset_from_directory(
  os.path.join(data_dir, 'train'),
  image_size=IMG_SIZE,
  batch_size=BATCH_SIZE
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(data_dir, 'test'),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(data_dir, 'val'),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

#image mapped
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
train_ds = train_ds.map(
    preprocess, num_parallel_calls=AUTOTUNE
)
test_ds = test_ds.map(
    preprocess, num_parallel_calls=AUTOTUNE
)
val_ds = val_ds.map(
    preprocess, num_parallel_calls=AUTOTUNE
)

#NN specification
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10
)

#threshold estimation and metrics
from sklearn.metrics import f1_score, confusion_matrix, classification_report, roc_curve, auc

y_prob = model.predict(test_ds).ravel()  
y_true = np.concatenate([y for _, y in test_ds], axis=0)

plt.figure(figsize=(8,5))
plt.hist(y_prob[y_true == 0], bins=50, alpha=0.6, label='NORMAL', color='blue')
plt.hist(y_prob[y_true == 1], bins=50, alpha=0.6, label='PNEUMONIA', color='red')
plt.axvline(0.5, color='black', linestyle='--', label='Default threshold 0.5')
plt.xlabel('Predicted Probability for PNEUMONIA')
plt.ylabel('Frequency')
plt.title('Probability Distributions')
plt.legend()
plt.show()

fpr, tpr, thresholds = roc_curve(y_true, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.title('ROC Curve')
plt.legend()
plt.show()

thresholds_range = np.arange(0, 1.01, 0.01)
f1_scores = []

best_thresh = 0
best_f1 = 0

for t in thresholds_range:
    y_pred_thresh = (y_prob >= t).astype(int)
    f1 = f1_score(y_true, y_pred_thresh)
    f1_scores.append(f1)
    if f1 > best_f1:
        best_f1 = f1
        best_thresh = t

plt.figure(figsize=(8,5))
plt.plot(thresholds_range, f1_scores, marker='o')
plt.axvline(best_thresh, color='red', linestyle='--', label=f'Best threshold = {best_thresh:.2f}')
plt.xlabel('Threshold')
plt.ylabel('F1 Score')
plt.title('F1 Score vs. Threshold')
plt.legend()
plt.show()

y_pred_best = (y_prob >= best_thresh).astype(int)
print(f"Best threshold: {best_thresh:.2f}, Best F1-score: {best_f1:.4f}\n")
print("Classification Report:")
print(classification_report(y_true, y_pred_best))
print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred_best))

#xai setup
os.environ["KERAS_BACKEND"] = "tensorflow"

import keras

from IPython.display import Image, display
import matplotlib as mpl

model_builder = keras.applications.xception.Xception
img_size = (299, 299)
preprocess_input = keras.applications.xception.preprocess_input
decode_predictions = keras.applications.xception.decode_predictions

last_conv_layer_name = "block14_sepconv2_act" #last conv layer

img_path = "/content/gr1.jpg" # Define img_path here

display(Image(img_path))

def get_img_array(img_path, size):
    
    img = keras.utils.load_img(img_path, target_size=size)
    array = keras.utils.img_to_array(img)
    array = np.expand_dims(array, axis=0)
    return array

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
  
    grad_model = keras.models.Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

img_array = preprocess_input(get_img_array(img_path, size=img_size))

model = model_builder(weights="imagenet")
model.layers[-1].activation = None
preds = model.predict(img_array)
print("Predicted:", decode_predictions(preds, top=1)[0])

heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)

plt.matshow(heatmap)
plt.show()

def save_and_display_gradcam(img_path, heatmap, cam_path="cam.jpg", alpha=0.4):

    img = keras.utils.load_img(img_path)
    img = keras.utils.img_to_array(img)

    heatmap = np.uint8(255 * heatmap)

    jet = mpl.colormaps["jet"]
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    jet_heatmap = keras.utils.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.utils.img_to_array(jet_heatmap)

    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.utils.array_to_img(superimposed_img)

    superimposed_img.save(cam_path)
    display(Image(cam_path))
  
save_and_display_gradcam(img_path, heatmap)
