import tensorflow as tf
import joblib
import numpy as np

import os

# Get the absolute path to the project root directory
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_model_and_scaler():
    # Load the model
    model = tf.keras.models.load_model(os.path.join(project_root, "models", "personality_model.keras"))
    # Load the scaler
    scaler = joblib.load(os.path.join(project_root, "models", "scaler.joblib"))
    return model, scaler

def predict_personality(features, model, scaler):
    """
    Make predictions using the trained model.

    Args:
        features: numpy array or list of features in the same order as training data
        model: loaded TensorFlow model
        scaler: loaded StandardScaler

    Returns:
        prediction: 0 for Introvert, 1 for Extrovert
    """
    # Convert to numpy array if it's a list
    features = np.array(features)

    # Reshape if single sample
    if len(features.shape) == 1:
        features = features.reshape(1, -1)

    # Scale the features
    features_scaled = scaler.transform(features)

    # Make prediction
    prediction = model.predict(features_scaled)

    # Convert probability to binary prediction
    return (prediction > 0.5).astype(int)

# Example usage
if __name__ == "__main__":
    # Load model and scaler
    model, scaler = load_model_and_scaler()

    # Example features (replace with your actual features)
    # Make sure the features are in the same order as during training:
    # [Time_spent_Alone, Stage_fear, Social_event_attendance, Going_outside,
    #  Drained_after_socializing, Friends_circle_size, Post_frequency]
    example_features = [4.0, 0.0, 4.0, 6.0, 0.0, 13.0, 5.0]

    # Make prediction
    prediction = predict_personality(example_features, model, scaler)
    print(f"Prediction: {'Extrovert' if prediction[0] == 1 else 'Introvert'}")