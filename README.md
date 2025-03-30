# Churn Prediction using Artificial Neural Networks (ANN)

This repository contains an **Artificial Neural Network (ANN)** model to predict customer churn using the `Churn_Modelling.csv` dataset.

## 📝 Project Overview
- Predicts whether a customer will leave a bank using an ANN.
- Uses **TensorFlow/Keras** for model implementation.
- **Data Preprocessing:** Label Encoding, One-Hot Encoding, and Standardization.
- **Model Architecture:** Multi-layer ANN with ReLU and Sigmoid activations.
- **Performance Metrics:** Confusion matrix and accuracy score.

## 💂️ Files
- `Churn_Modelling.csv` → Dataset
- `churn_prediction.ipynb` → Jupyter Notebook for training the ANN.
- `bank.keras` → Saved trained ANN model.

---

## 🚀 Setup & Installation

### 1️⃣ Clone the Repository
```sh
git clone https://github.com/your-username/churn-prediction-ann.git
cd churn-prediction-ann
```

### 2️⃣ Install Dependencies
Ensure you have Python installed, then install the required libraries:
```sh
pip install numpy pandas tensorflow scikit-learn
```

### 3️⃣ Run the Notebook
Open and execute `churn_prediction.ipynb` to preprocess the data and train the model.

---

## 📊 Data Preprocessing

### **🔹 Steps Involved**
1. **Loading Data:** Read `Churn_Modelling.csv` using pandas.
2. **Feature Selection:** Extract relevant columns from the dataset.
3. **Encoding Categorical Variables:**  
   - Label Encoding for **Gender**.
   - One-Hot Encoding for **Geography**.
4. **Feature Scaling:** Standardize numerical values using `StandardScaler`.

### **📀 Code**
```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

# Load dataset
dataset = pd.read_csv("Churn_Modelling.csv")
x = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values

# Encode Gender (Label Encoding)
le = LabelEncoder()
x[:, 2] = le.fit_transform(x[:, 2])

# Encode Geography (One-Hot Encoding)
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
x = np.array(ct.fit_transform(x))

# Split Data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Feature Scaling
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
```

---

## 🧐 ANN Model Architecture

| Layer  | Neurons | Activation |
|--------|---------|------------|
| Input Layer | Encoded customer data | - |
| Hidden Layer 1 | 6 | ReLU |
| Hidden Layer 2 | 6 | ReLU |
| Output Layer | 1 | Sigmoid |

- **Loss Function:** Binary Crossentropy  
- **Optimizer:** Adam  
- **Batch Size:** 32  
- **Epochs:** 1000  

### **📀 Model Training Code**
```python
import tensorflow as tf

# Initialize ANN
ann = tf.keras.models.Sequential()

# Add Hidden Layers
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

# Add Output Layer
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Compile Model
ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the Model
ann.fit(x_train, y_train, batch_size=32, epochs=1000)

# Save Model
ann.save('bank.keras')
```

---

## 🔍 Making Predictions

### **Predicting a Single Customer’s Churn Probability**
```python
# Load Trained Model
model = tf.keras.models.load_model('bank.keras')

# Example Customer Data: [Geography, Credit Score, Gender, Age, Balance, etc.]
sample_input = np.array([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])

# Apply Same Feature Scaling
sample_input_scaled = sc.transform(sample_input)

# Predict Churn
prediction = model.predict(sample_input_scaled)
print("Churn Prediction:", prediction > 0.5)  # Output: True (Will Churn) or False (Won't Churn)
```

---

## 📊 Model Evaluation

### **Confusion Matrix & Accuracy Score**
```python
from sklearn.metrics import confusion_matrix, accuracy_score

# Convert Probabilities to Binary (0 or 1)
y_pred = (ann.predict(x_test) > 0.5)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Accuracy Score
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)
```

---

## 🔥 Future Improvements
- **Hyperparameter tuning** for better accuracy.
- **Feature engineering** to improve model performance.
- **Comparison with other models** (Random Forest, XGBoost).
- **Deploy the model** using Flask/FastAPI for real-world applications.

---

## 💚 License
This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.


