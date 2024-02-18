from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd

# # Step 1: Load the saved model
model = load_model('yoga_poses_5/weights.best.hdf5')

# Step 2: Prepare your input data (this is just a placeholder, adjust as necessary)
# Example for a model expecting input shape of (None, 10)
data_path = 'yoga_image.csv'
data = pd.read_csv(data_path, header=None)

# Assuming we're visualizing the first row for demonstration
print(data)
row = data.iloc[0]

print(row)

x = np.array(row)

x = x.reshape(1, 51)

print(x)



# # Step 3: Make a prediction
prediction = model.predict(x)

print("Prediction:", prediction)
