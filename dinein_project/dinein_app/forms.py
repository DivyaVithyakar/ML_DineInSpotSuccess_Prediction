from django import forms

class DineSpotInputForm(forms.Form):
    city = forms.CharField(label='City')
    latitude = forms.FloatField(label='Latitude')
    cuisines = forms.CharField(label='Cuisines')
    average_cost_for_two = forms.IntegerField(label='Average Cost(₹)')
    price_range = forms.IntegerField(label='Price Range')
    votes = forms.IntegerField(label='Number of Votes')
