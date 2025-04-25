# 🍽️ ML_DineInSpotSuccess_Prediction

This project predicts the success of dine-in restaurants based on various features like location, cost, cuisines, and more. It combines a machine learning model with a Django-based web interface.

---

## 🚀 Features

- Train and use ML models to predict restaurant success
- User-friendly Django frontend for input
- Real-time prediction and result display
- Input validation and clean UI

---

## 🧠 Machine Learning

- Model: Random Forest, Grid Search
- Preprocessing: Label encoding, missing value handling, feature selection
- Dataset: Zomato
- Target Variable: Success (`Yes` / `No`)

---

## 🛠️ Tech Stack

- Python 3.9
- Django 4.2
- Scikit-learn
- Pandas, NumPy
- HTML/CSS for frontend

## Getting Started – Run Locally

Follow these steps to set up the project on your local machine:

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/ML_DineInSpotSuccess_Prediction.git
cd ML_DineInSpotSuccess_Prediction
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate  
```
### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Prepare the dataset

```bash
Place the dataset file (e.g., zomato.csv) in the appropriate folder as mentioned in your code (usually a /data/ or /input/ directory).
```

### 5. Run migrations (for Django)

```bash
python manage.py migrate
```

### 6. Start the Django server

```bash
python manage.py runserver
```

### 7. Open in browser

```bash
Visit http://127.0.0.1:8000 in your web browser to access the application.
```




