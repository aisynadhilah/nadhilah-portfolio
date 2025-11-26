import os
import numpy as np
import tensorflow as tf # type: ignore
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras import Sequential, regularizers # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional # type: ignore
from tensorflow.keras.losses import BinaryCrossentropy # type: ignore
from tensorflow.keras.metrics import AUC # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint # type: ignore
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
from sklearn.utils.class_weight import compute_class_weight
import gc

# LOAD DATA FROM FOLDER
def load_data_from_folder(folder_path, allowed_labels=['Normal', 'PPOK']):
    X_list, y_list = [], []

    for class_name in os.listdir(folder_path):
        if class_name not in allowed_labels:
            continue

        class_dir = os.path.join(folder_path, class_name)
        if not os.path.isdir(class_dir):
            continue

        for file in os.listdir(class_dir):
            if file.endswith(".npy"):
                X_list.append(np.load(os.path.join(class_dir, file)))
                y_list.append(class_name)

    return X_list, y_list

# PREPARE LSTM DATA
def prepare_data_for_lstm(X_list, y_list):
    max_len = max([x.shape[0] for x in X_list])
    X_padded = pad_sequences(X_list, maxlen=max_len, dtype='float32', padding='post', truncating='post')

    le = LabelEncoder()
    y_encoded = le.fit_transform(y_list)

    return X_padded, y_encoded, le

# BILSTM MODEL ARCHITECTURE
def build_bilstm_model(input_shape):
    model = Sequential([
        Bidirectional(LSTM(64, return_sequences=False, dropout=0.2, input_shape=input_shape)),
        BatchNormalization(),
        Dense(48, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(1e-4))
    ])

    loss_fn = BinaryCrossentropy(label_smoothing=0.05)
    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)

    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=[
            'accuracy',
            AUC(curve='ROC', name='auroc'),
            AUC(curve='PR', name='auprc')
        ]
    )

    return model

# NOISE AUGMENTATION
def add_noise(X, noise_factor=0.01):
    return X + np.random.normal(0, noise_factor, X.shape)

# TRAINING + EVALUATION (5-Fold Cross Validation)
def train_and_evaluate(data_path, checkpoint_folder="checkpoints"):

    # ---- Load raw data ----
    X_list, y_list = load_data_from_folder(data_path)
    # ---- Prepare padded data ----
    X, y, encoder = prepare_data_for_lstm(X_list, y_list)
    # ---- K-Fold ----
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    val_acc_per_fold = []
    fold_no = 1

    os.makedirs(checkpoint_folder, exist_ok=True)
  
    # TRAINING
    for train_idx, val_idx in kfold.split(X, y):
        print(f"\n========== TRAINING FOLD {fold_no} ==========")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
      
        X_train_noisy = add_noise(X_train)
        class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
        class_weights = dict(enumerate(class_weights))
        model = build_bilstm_model(input_shape=(X.shape[1], X.shape[2]))
        checkpoint_path = os.path.join(checkpoint_folder, f"model_fold{fold_no}.keras")
        callbacks = [
            ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='val_loss', mode='min'),
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
        ]
        history = model.fit(
            X_train_noisy, y_train,
            epochs=150, batch_size=16,
            validation_data=(X_val, y_val),
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1
        )
        eval_results = model.evaluate(X_val, y_val, verbose=0)
        val_acc_per_fold.append(eval_results[1])

        print(f"Fold {fold_no} → Val_Acc = {eval_results[1]:.4f}")

        # Visualization
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title(f'Fold {fold_no} - Loss')
        plt.grid()

        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title(f'Fold {fold_no} - Accuracy')
        plt.grid()
        plt.show()

        # Predict & CM
        y_prob = model.predict(X_val)
        y_pred = (y_prob > 0.5).astype(int)

        print(classification_report(y_val, y_pred, target_names=encoder.classes_))

        cm = confusion_matrix(y_val, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=encoder.classes_, yticklabels=encoder.classes_)
        plt.title(f'Confusion Matrix Fold {fold_no}')
        plt.show()

        fold_no += 1
        tf.keras.backend.clear_session()
        gc.collect()

    # SELECT BEST MODEL
    best_fold = np.argmax(val_acc_per_fold) + 1
    print("\n============================================")
    print(f"  BEST MODEL → FOLD {best_fold}")
    print(f"  BEST VAL ACC: {val_acc_per_fold[best_fold-1]:.4f}")
    print("============================================")

    return best_fold
