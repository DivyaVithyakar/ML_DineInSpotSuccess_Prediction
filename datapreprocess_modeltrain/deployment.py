import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

# Load model, feature selector, threshold, and encoder
model = pickle.load(open("../data/final_model.sav", "rb"))
selector = pickle.load(open("../data/selector.sav", "rb"))
threshold = float(open("../data/threshold.txt", "r").read())
encoder = pickle.load(open("../data/label_encoders.sav", 'rb'))

# Sample raw input (must match all input features before selection)
# Order: ['City', 'Latitude', 'Cuisines', 'Average Cost for two', 'Price range', 'Votes']

test_data = pd.DataFrame([{
    'City': "Namakkal",  # City not seen during training
    'Latitude': 11.1523,
    'Cuisines': "Biryani, North Indian",  # Cuisines is the only categorical feature
    'Average Cost for two': 1000,
    'Price range': 1,
    'Votes': 500
}])

# Handle unseen labels for the 'City' column using OrdinalEncoder
ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

# Reshape the data into a 2D array (single feature)
city_encoded = ordinal_encoder.fit_transform([[test_data['City'][0]]])[0][0]
cuisines_encoded = encoder['Cuisines'].transform([test_data['Cuisines'][0]])[0]

# Update the test data with encoded values
test_data_encoded = test_data.copy()
test_data_encoded['City'] = city_encoded
test_data_encoded['Cuisines'] = cuisines_encoded

# Ensure the test_data_encoded contains all features in the correct order
test_data_encoded = test_data_encoded[['City', 'Latitude', 'Cuisines', 'Average Cost for two', 'Price range', 'Votes']]

# Predict probability
probability = model.predict_proba(test_data_encoded)[0][1]
prediction = 1 if probability >= threshold else 0

# Final output
print(f"🔍 Predicted Success Probability: {probability:.2f}")
if prediction == 1:
    print("🌟 This restaurant is likely to be successful! Great choice for a visit or investment. 🍽️")
else:
    print("⚠️ This restaurant might not perform as well. Proceed with caution or consider alternatives.")
