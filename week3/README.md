# Titanic Binary Classifier - TensorFlow.js (Browser)

This project provides a **shallow neural network binary classifier** for the Kaggle Titanic dataset, implemented entirely in the browser using TensorFlow.js and tfjs-vis. No server or backend required — ready for instant hosting via GitHub Pages.

## 🚀 Demo

**Try it out instantly in your browser by uploading `train.csv` and `test.csv` from [Kaggle Titanic](https://www.kaggle.com/c/titanic/data).**

## Features

- **Client-side only**: All computation runs on your device, zero backend
- **Upload your own CSV**: No hardcoded data; works with user-provided `train.csv`/`test.csv`
- **Data preview and stats**: Inspect table preview, missing values, shape
- **Interactive preprocessing**: Median/mode imputation, standardization, one-hot encoding, optional family features
- **Live model training**: Single hidden layer, early stopping, live tfjs-vis loss/acc plots
- **ROC/AUC, metrics, confusion matrix** with interactive threshold slider
- **Export for Kaggle**: Download `submission.csv`, probability CSV, and trained model
- **Deploy easily**: Just push files to a GitHub repo, enable Pages

## Usage

### 1. **Prepare your dataset**
   - Download `train.csv` and `test.csv` from the [Titanic Kaggle competition](https://www.kaggle.com/c/titanic/data).

### 2. **Run the app**
   - Open `index.html` directly, or visit the [GitHub Pages](#deployment) link if deployed.

### 3. **Workflow**
   1. **Data Load:** Upload both CSVs, inspect data preview and missing values.
   2. **Preprocessing:** Click to engineer features and impute missing values. Optionally add FamilySize/IsAlone.
   3. **Model:** Click to create a shallow neural network.
   4. **Training:** Start training. Watch tfjs-vis output for live performance.
   5. **Metrics:** Adjust the ROC threshold slider; view confusion matrix, precision, recall, F1, AUC.
   6. **Prediction:** Predict probabilities for the test set.
   7. **Export:** Download predictions and model (for Kaggle submission).

### 4. **Deployment**

- Create a public GitHub repository (e.g., `username/titanic-tfjs`).
- Commit at least these two files:
