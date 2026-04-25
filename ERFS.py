"""
=========================================================
Speech Emotion Recognition (SER) - HIGH ACCURACY VERSION
Target: 85%–92% on RAVDESS (realistic ceiling)

UPGRADES:
✔ Log-Mel Spectrogram (BEST feature for CNN SER)
✔ Audio augmentation (noise, pitch, stretch)
✔ Deep CNN (VGG-style)
✔ Proper normalization
✔ Early stopping
✔ Stable training pipeline
=========================================================
"""

# =========================================================
# 1. IMPORTS
# =========================================================
import os
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks


# =========================================================
# 2. CONFIG
# =========================================================
DATA_PATH = r"D:\CodeAlpha_Tasks\Emotion_Recognition_from_Speech\Actors"

EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

FIXED_TIME = 128
N_MELS = 128


# =========================================================
# 3. DATASET LOADING
# =========================================================
def build_dataset(path):
    data = []

    for actor in os.listdir(path):
        actor_path = os.path.join(path, actor)

        if not os.path.isdir(actor_path):
            continue

        for file in os.listdir(actor_path):
            if file.endswith(".wav"):
                emotion = EMOTION_MAP[file.split("-")[2]]

                data.append({
                    "path": os.path.join(actor_path, file),
                    "emotion": emotion
                })

    return pd.DataFrame(data)


df = build_dataset(DATA_PATH)

print("Dataset size:", df.shape)
print(df["emotion"].value_counts())


# =========================================================
# 4. AUDIO AUGMENTATION (KEY BOOST)
# =========================================================
def augment(audio, sr):

    return [
        audio + 0.005 * np.random.randn(len(audio)),          # noise
        librosa.effects.pitch_shift(audio, sr=sr, n_steps=2), # pitch
        librosa.effects.time_stretch(audio, rate=0.9)         # speed
    ]


# =========================================================
# 5. LOG-MEL FEATURE EXTRACTION (CRITICAL UPGRADE)
# =========================================================
def extract_logmel(audio, sr):

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=N_MELS
    )

    log_mel = librosa.power_to_db(mel)

    # FIX SHAPE
    if log_mel.shape[1] < FIXED_TIME:
        pad = FIXED_TIME - log_mel.shape[1]
        log_mel = np.pad(log_mel, ((0,0),(0,pad)))
    else:
        log_mel = log_mel[:, :FIXED_TIME]

    return log_mel


# =========================================================
# 6. BUILD DATASET
# =========================================================
X, y = [], []

print("\nProcessing dataset...")

for _, row in df.iterrows():

    try:
        audio, sr = librosa.load(row["path"], sr=None)
        audio, _ = librosa.effects.trim(audio)

        # ORIGINAL
        X.append(extract_logmel(audio, sr))
        y.append(row["emotion"])

        # AUGMENTED
        for aug in augment(audio, sr):
            X.append(extract_logmel(aug, sr))
            y.append(row["emotion"])

    except Exception as e:
        print("Error:", row["path"], e)


X = np.array(X)
y = np.array(y)

print("\nRaw shape:", X.shape)


# =========================================================
# 7. NORMALIZATION (IMPORTANT)
# =========================================================
X = X / (np.max(np.abs(X), axis=(1,2), keepdims=True) + 1e-8)


# =========================================================
# 8. LABEL ENCODING
# =========================================================
le = LabelEncoder()
y_encoded = le.fit_transform(y)

print("Classes:", le.classes_)


# =========================================================
# 9. RESHAPE FOR CNN
# =========================================================
X = np.expand_dims(X, axis=-1)

print("CNN input shape:", X.shape)


# =========================================================
# 10. TRAIN TEST SPLIT
# =========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)


# =========================================================
# 11. DEEP CNN (VGG STYLE - HIGH PERFORMANCE)
# =========================================================
model = models.Sequential([

    layers.Input(shape=(128,128,1)),

    # Block 1
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2),

    # Block 2
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2),

    # Block 3
    layers.Conv2D(256, (3,3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2),

    # Block 4 (deeper learning)
    layers.Conv2D(512, (3,3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2),

    # Classifier
    layers.Flatten(),

    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),

    layers.Dense(256, activation='relu'),
    layers.Dropout(0.3),

    layers.Dense(len(le.classes_), activation='softmax')
])


# =========================================================
# 12. COMPILE
# =========================================================
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()


# =========================================================
# 13. EARLY STOPPING
# =========================================================
early_stop = callbacks.EarlyStopping(
    monitor="val_accuracy",
    patience=10,
    restore_best_weights=True
)


# =========================================================
# 14. TRAINING
# =========================================================
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=70,
    batch_size=32,
    callbacks=[early_stop]
)


# =========================================================
# 15. EVALUATION
# =========================================================
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=le.classes_))


# =========================================================
# 16. CONFUSION MATRIX
# =========================================================
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8,6))
plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix")
plt.colorbar()

plt.xticks(range(len(le.classes_)), le.classes_, rotation=45)
plt.yticks(range(len(le.classes_)), le.classes_)

plt.show()


# =========================================================
# 17. FINAL SCORE
# =========================================================
loss, acc = model.evaluate(X_test, y_test)

print("\nFINAL TEST ACCURACY:", acc)

# create models folder if not exists
os.makedirs("models", exist_ok=True)

# save trained model
model.save("models/speech_emotion_model.keras")

print("\nModel saved successfully!")
print("Path: models/speech_emotion_model.keras")

# =========================================================
# 19. SAVE LABEL ENCODER
# =========================================================


with open("models/label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

print("Label encoder saved successfully!")