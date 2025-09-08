# C_train_model.py
# Capstone Part III: Training a 1D CNN with participant-level split (reverted stable version)

import os
import json
import numpy as np
import tensorflow as tf
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # 0=all logs, 1=filter INFO, 2=filter INFO+WARNING, 3=only ERROR
from sklearn.model_selection import train_test_split
from tensorflow.keras import regularizers
from globals import *


def mask_from_subj(subj_array, chosen):
    """Return boolean mask for samples belonging to chosen participants."""
    return np.isin(subj_array, chosen)


def main():
    # --- Load dataset ---
    dataLocation = os.path.join("Windowed_data", "windowed_dataset.npz")
    if not os.path.exists(dataLocation):
        raise FileNotFoundError(f"Windowed dataset not found at: {dataLocation}")

    data = np.load(dataLocation)
    X = data["X"]      # shape (N, win_size, n_channels)
    y = data["y"]      # shape (N,)
    subj = data["subj"]  # shape (N,)

    print(f"Loaded dataset: X={X.shape}, y={y.shape}, subj={subj.shape}")

    # --- Get unique participants ---
    participants = np.unique(subj)
    print(f"Unique participants ({len(participants)}): {participants}")

    # --- Train/Val/Test split at participant level ---
    train_subj, test_subj = train_test_split(participants, test_size=0.2, random_state=42)
    train_subj, val_subj = train_test_split(train_subj, test_size=0.25, random_state=42)
    # → 60% train, 20% val, 20% test (all by participant)
    print(f"TRAIN participants ({len(train_subj)}): {train_subj}")
    print(f"VAL   participants ({len(val_subj)}): {val_subj}")
    print(f"TEST  participants ({len(test_subj)}): {test_subj}")

    # --- Build masks for each set ---
    train_mask = mask_from_subj(subj, train_subj)
    val_mask   = mask_from_subj(subj, val_subj)
    test_mask  = mask_from_subj(subj, test_subj)

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val     = X[val_mask], y[val_mask]
    X_test, y_test   = X[test_mask], y[test_mask]

    print(f"Data shapes -> Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # --- Encode labels (0..n_classes-1) ---
    unique_labels = np.unique(y)
    label_to_index = {int(lbl): idx for idx, lbl in enumerate(unique_labels)}
    index_to_label = {idx: int(lbl) for idx, lbl in enumerate(unique_labels)}

    # vectorize mapping
    map_fn = np.vectorize(lambda v: label_to_index[int(v)])
    y_train = map_fn(y_train)
    y_val   = map_fn(y_val)
    y_test  = map_fn(y_test)

    print(f"Classes found: {list(unique_labels)} -> encoded as 0..{len(unique_labels)-1}")

    # Save label maps (ints are JSON-friendly)
    with open("label_mappings.json", "w") as f:
        json.dump({"label_to_index": label_to_index,
                   "index_to_label": index_to_label}, f, indent=2)

    # --- Normalize per channel (using training set stats only) ---
    # mean/std computed over all windows and time steps for each channel (keepdims for broadcast)
    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std = X_train.std(axis=(0, 1), keepdims=True) + 1e-6
    X_train = (X_train - mean) / std
    X_val   = (X_val - mean) / std
    X_test  = (X_test - mean) / std

    # --- Build improved 1D CNN model (stable) ---
    n_classes = len(unique_labels)

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=X_train.shape[1:]),

        # Conv block 1
        tf.keras.layers.Conv1D(64, kernel_size=5, activation="relu",
                               kernel_regularizer=regularizers.l2(1e-4)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling1D(pool_size=2),

        # Conv block 2
        tf.keras.layers.Conv1D(128, kernel_size=5, activation="relu",
                               kernel_regularizer=regularizers.l2(1e-4)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling1D(pool_size=2),

        # Conv block 3
        tf.keras.layers.Conv1D(256, kernel_size=3, activation="relu",
                               kernel_regularizer=regularizers.l2(1e-4)),
        tf.keras.layers.BatchNormalization(),

        # Pooling
        tf.keras.layers.GlobalMaxPooling1D(),

        # Dense blocks with stronger dropout
        tf.keras.layers.Dense(256, activation="relu",
                              kernel_regularizer=regularizers.l2(1e-4)),
        tf.keras.layers.Dropout(0.6),

        tf.keras.layers.Dense(128, activation="relu",
                              kernel_regularizer=regularizers.l2(1e-4)),
        tf.keras.layers.Dropout(0.6),

        tf.keras.layers.Dense(n_classes, activation="softmax")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()

    # --- Callbacks ---
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint("best_model.keras", save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss",
                                             factor=0.5,
                                             patience=3,
                                             min_lr=1e-6,
                                             verbose=1)
    ]

    # --- Train ---
    print("Training model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=64,   # smaller batch for better generalization
        callbacks=callbacks,
        verbose=1
    )

    # --- Evaluate ---
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test accuracy (participant-level split): {test_acc:.4f}")


if __name__ == "__main__":
    main()
