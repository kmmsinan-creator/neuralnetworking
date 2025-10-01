# Titanic Survival Classifier (TensorFlow.js)

This project is a **browser-based machine learning app** that predicts Titanic passenger survival using the Kaggle dataset.  
It is built with **TensorFlow.js** and **tfjs-vis**, and runs entirely in the browser without a backend.

---

## 🚩 Fixed Issues

### 1. Comma Escape Problem in CSV
**Problem:**  Titanic dataset has passenger names like  
"Cumings, Mrs. John Bradley (Florence Briggs Thayer)".  
If you split CSV lines just by ,, the name breaks into multiple columns → data loads incorrectly.  
**Fix:** Use PapaParse library instead of manual string splitting.  
PapaParse correctly reads CSV files with quotes and commas inside text fields.  

Example:
Papa.parse(fileText, { header: true, dynamicTyping: true, skipEmptyLines: true });


---

### 2. Evaluation Table Not Showing
**Problem:** After training, evaluation metrics (accuracy, precision, recall, F1, confusion matrix) didn’t show.  
Cause → mismatch in how TensorFlow.js logs metrics (acc vs accuracy) and missing DOM updates.  
**Fix:** Updated the evaluation step to:  
- Use logs.acc || logs.accuracy safely.  
- Calculate confusion matrix and metrics manually from predictions.  
- Always update the evaluation <div> with results.  

---

### 3. Summarizing the Code Logic

Here is the high-level workflow of the app:

**Load Data**  
- User uploads train.csv and test.csv.  
- Data parsed with PapaParse.  

**Inspect Data**  
- Display preview tables (first 10 rows).  
- Show basic visualizations (survival by sex/class) using tfjs-vis.  

**Preprocess Data**  
- Fill missing values with median (Age, Fare) or mode (Embarked).  
- Normalize numerical features.  
- One-hot encode categorical features (Sex, Pclass, Embarked).  
- Optionally add engineered features: FamilySize and IsAlone.  

**Create Model**  
- Build a neural network in TensorFlow.js:  
  - Input layer  
  - Dense hidden layers with ReLU  
  - Output layer with Sigmoid  
- Compile with Adam optimizer + binary crossentropy loss.  

**Train Model**  
- Split data into training/validation sets.  
- Train the model for several epochs.  
- Visualize training progress (loss & accuracy curves).  

**Evaluate Model**  
- Predict on validation set.  
- Show confusion matrix and metrics: Accuracy, Precision, Recall, F1.  
- Allow adjusting decision threshold with a slider.  

**Predict & Export**  
- Predict survival on test set.  
- Export results as submission.csv for Kaggle.  
