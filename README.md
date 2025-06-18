
# 🌸 Women's Health & Nutrition Deficiency Prediction App

A user-friendly web application to predict nutrition deficiencies and support women's health, including a menstrual cycle tracker. Built using Machine Learning and Streamlit with an attractive, responsive interface.

---

## 🧠 Project Overview

This project predicts nutritional deficiencies based on user input like symptoms or body conditions. The app is designed specifically for women's health and includes additional features like:

- 📊 Visual health charts
- 🌱 Personalized suggestions
- 🩺 Machine Learning prediction using Decision Tree
- 🩸 Menstrual cycle tracker with animations
- 💖 Girly-themed UI for better user experience

---

## 🛠️ Tech Stack

- **Frontend:** Streamlit, HTML/CSS (custom styling), Lottie animations
- **Backend:** Python
- **ML Model:** Decision Tree Classifier (scikit-learn)
- **Data Handling:** Pandas
- **Visualization:** Matplotlib / Plotly
- **Deployment:** GitHub / Streamlit Cloud

---

## 📁 Folder Structure

```
nutrition_deficiency_predictor/
│
├── app.py                        # Main Streamlit app
├── model/
│   └── decision_tree_model.pkl   # Trained ML model
├── dataset/
│   └── nutrition_data.csv        # Input dataset
├── pages/
│   └── MenstrualCycleTracker.py  # Optional multipage tracker
├── assets/
│   └── animations/, images/      # Icons, lotties, styling
├── README.md

```

---

## 🚀 How to Run Locally

1. **Clone the Repository**
   ```bash
   git clone https://github.com/vedanti310/nutrition_deficiency_predictor.git
   cd nutrition_deficiency_predictor
   ```

2. **Install Required Packages**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the App**
   ```bash
   streamlit run app.py
   ```

---


## 📌 Features

- ✅ Predicts nutrition deficiencies
- 🎨 Beautiful UI
- 📆 Tracks menstrual cycles visually
- 📊 Dynamic plots and health insights
- 💡 Health and diet suggestions

---



## 🧠 Machine Learning Model

- Model used: `DecisionTreeClassifier`
- Training done on labeled symptoms vs deficiencies
- Encoded using `LabelEncoder` for categorical inputs

---




