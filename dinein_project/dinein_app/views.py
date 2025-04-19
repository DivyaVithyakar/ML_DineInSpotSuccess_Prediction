import pickle
import numpy as np
import pandas as pd
from django.shortcuts import render, redirect
from django.urls import reverse
from sklearn.preprocessing import OrdinalEncoder
from .forms import DineSpotInputForm

# Load pre-trained components
model = pickle.load(open("../data/final_model.sav", "rb"))
selector = pickle.load(open("../data/selector.sav", "rb"))
threshold = float(open("../data/threshold.txt", "r").read())
encoder = pickle.load(open("../data/label_encoders.sav", 'rb'))

# Initialize OrdinalEncoder for unseen city handling
ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
# NOTE: You must fit this encoder on the same cities used during training

def home(request):
    if request.method == 'POST':
        form = DineSpotInputForm(request.POST)
        if form.is_valid():
            data = form.cleaned_data

            # Create DataFrame from input
            input_df = pd.DataFrame([{
                'City': data['city'],
                'Latitude': data['latitude'],
                'Cuisines': data['cuisines'],
                'Average Cost for two': data['average_cost_for_two'],
                'Price range': data['price_range'],
                'Votes': data['votes']
            }])

            # Encode 'City' and 'Cuisines'
            city_encoded = ordinal_encoder.fit_transform([[input_df['City'][0]]])[0][0]
            cuisines_encoded = encoder['Cuisines'].transform([input_df['Cuisines'][0]])[0]

            input_df['City'] = city_encoded
            input_df['Cuisines'] = cuisines_encoded

            # Ensure correct feature order
            input_df = input_df[['City', 'Latitude', 'Cuisines', 'Average Cost for two', 'Price range', 'Votes']]


            # input_selected = selector.transform(input_df)
            input_selected = input_df  # Skip selector if it's causing shape mismatch

            # Predict
            probability = model.predict_proba(input_selected)[0][1]
            prediction = 1 if probability >= threshold else 0

            if prediction == 1:
                result = f"🌟 Predicted Success Probability: {probability:.2f}. This restaurant is likely to be successful! Great choice for a visit or investment. 🍽️"
            else:
                result = f"⚠️ Predicted Success Probability: {probability:.2f}. This restaurant might not perform as well. Proceed with caution or consider alternatives."

            request.session['result'] = result
            return redirect(reverse('dinein_result'))

    else:
        form = DineSpotInputForm()
    return render(request, 'dinespot/home.html', {'form': form})

def result(request):
    result = request.session.get('result', None)
    return render(request, 'dinespot/result.html', {'result': result})
from django.shortcuts import render
