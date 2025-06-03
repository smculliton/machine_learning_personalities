import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os

# Get the absolute path to the project root directory
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load data with correct path
df = pd.read_csv(os.path.join(project_root, "data", "personality_dataset.csv"))

# Map 'Yes' to 1, 'No' to 0 for those columns
df['Stage_fear'] = df['Stage_fear'].map({'Yes': 1, 'No': 0})
df['Drained_after_socializing'] = df['Drained_after_socializing'].map({'Yes': 1, 'No': 0})

print(df.head())

X = df.drop("Personality", axis=1)
y = df["Personality"]

# Debug prints for original data
print("\nOriginal data statistics:")
print("X info:")
print(X.info())
print("\nX description:")
print(X.describe())
print("\nChecking for NaN values:")
print(X.isnull().sum())

# Handle any NaN values if they exist
X = X.fillna(X.mean())

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# Encode y values
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_val_encoded = le.transform(y_val)
y_test_encoded = le.transform(y_test)

print(f"Y labels: {y_train_encoded[:5]}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Debug prints
print("\nData statistics after scaling:")
print(f"X_train_scaled mean: {np.mean(X_train_scaled):.4f}")
print(f"X_train_scaled std: {np.std(X_train_scaled):.4f}")
print(f"X_train_scaled min: {np.min(X_train_scaled):.4f}")
print(f"X_train_scaled max: {np.max(X_train_scaled):.4f}")

# Build the model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model with gradient clipping
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)
model.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train the model
history = model.fit(
    X_train_scaled,
    y_train_encoded,
    epochs=50,
    batch_size=32,
    validation_data=(X_val_scaled, y_val_encoded),
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
    ]
)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test_encoded)
print(f"\nTest accuracy: {test_accuracy:.4f}")

# Save the model
tf.keras.models.save_model(model, os.path.join(project_root, "models", "personality_model.keras"))

# Save the scaler
joblib.dump(scaler, os.path.join(project_root, "models", "scaler.joblib"))