## 🧠 Alzheimer’s Disease Prediction System

    A complete Machine Learning pipeline to predict the risk of Alzheimer’s Disease using patient health and lifestyle data.
    The project includes model training, evaluation, visualization, and a Streamlit web app for real-time predictions.

##  🚀 Features
        ✅ Multiple ML models:
            Logistic Regression
            SVM
            Decision Tree
            Random Forest
            Gradient Boosting (XGB)

        ✅ Advanced Evaluation:
            Accuracy, Recall, F1 Score
            ROC Curve & AUC
            Confusion Matrix
            Feature Importance

        ✅ Multi Test Size Experiment:
            0.2, 0.25, 0.3 split comparison
            Best model selected per test size
            Final global best model selection
        
        ✅ Data Handling:
            Under-sampling for class imbalance
            Feature scaling using StandardScaler

        ✅ Visualization:
            Accuracy vs Recall graphs
            ROC curves
            Feature importance plots
        
        ✅ Streamlit Web App:
            User-friendly input form
            Real-time prediction
            Risk probability display

##  📁 Project Structure
        Alzheimers-Prediction/
        │
        ├── data/
        │   └── alzheimers_disease_data.csv
        │
        ├── models/
        │   ├── best_model.pkl
        │   ├── scaler.pkl
        │   └── feature_columns.pkl
        │
        ├── notebooks/
        │   └── alzheimer_analysis.ipynb
        │
        ├── outputs/
        │   ├── plots/
        │   └── reports/
        │
        ├── src/
        │   ├── data_loader.py
        │   ├── preprocessing.py
        │   ├── train_*.py
        │   └── evaluate.py
        │
        ├── app/
        │   └── app.py
        │
        ├── main.py
        └── README.md

##  ⚙️ Installation
        1️⃣ Clone Repository
            git clone <your-repo-link>
            cd Alzheimers-Prediction

        2️⃣ Create Virtual Environment
            python -m venv venv
            venv\Scripts\activate

        3️⃣ Install Dependencies
            pip install -r requirements.txt

        ▶️ How to Run

        Train Models
        python main.py

        This will:

            Train all models
            Evaluate performance
            Save best model
            🔹 Run Notebook (Analysis)
            jupyter notebook

            Open:
            notebooks/alzheimer_analysis.ipynb

##  Run Streamlit App
        streamlit run app/app.py
        🧪 Model Evaluation Strategy

        We use a weighted scoring approach:

        Score = 0.6 × Recall + 0.3 × Accuracy + 0.1 × F1
        Why?
        Recall → Detect disease cases (priority)
        Accuracy → Overall correctness
        F1 → Balance between precision & recall

## 📊 Metrics Used
        Accuracy
        Recall (Critical for disease detection)
        Precision
        F1 Score
        ROC-AUC
        📈 Outputs

        Generated in:

        outputs/plots/

        Includes:

        Accuracy vs Recall graphs
        ROC curves
        Feature importance plots
        Confusion matrices

## 🧠 Input Features
        Functional Assessment
        ADL (Daily Living Activities)
        MMSE Score
        Memory Complaints
        Behavioral Problems
        BMI
        Diet Quality
        Alcohol Consumption
        Physical Activity
        Sleep Quality
        Cholesterol (HDL & Total)
        Systolic Blood Pressure
        Age

## 🌐 Streamlit App

        The UI allows users to:

        Enter patient data
        View prediction instantly
        See probability of Alzheimer’s risk

## 🎯 Final Model Selection
        Best model selected per test size
        Final model selected using highest weighted score

## 🧠 Key Insights
        Recall is prioritized due to medical importance
        Lifestyle + cognitive features strongly influence prediction
        Ensemble models (RF, XGB) perform best

## 📌 Future Improvements
        SHAP explainability
        Hyperparameter tuning
        Deep Learning models
        Real clinical dataset integration

## 👨‍💻 Author

        Bharat Singh Lovewanshi

## ⭐ If You Like This Project

Give it a ⭐ on GitHub!

## Thank You