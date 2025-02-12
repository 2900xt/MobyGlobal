from tensorflow.keras.models import load_model

# Load the model from the .h5 file
model = load_model('Moby5.h5')

# Example usage: Make predictions with the loaded model
# Assuming you have some input data `x` ready
# predictions = model.predict(x)

# Print the model summary to verify it's loaded correctly
model.summary()