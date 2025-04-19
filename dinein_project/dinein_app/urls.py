from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='dinein_home'),
    path('result/', views.result, name='dinein_result'),
]