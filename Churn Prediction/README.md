# 🛒 E-Commerce Customer Churn Prediction  
## Deep Learning | TensorFlow | Streamlit Web App

---

## 📌 Project Overview

Customer churn is a major challenge in the e-commerce industry, directly impacting revenue and customer lifetime value.  
This project focuses on predicting customer churn using a deep learning model built with TensorFlow and deploying it as a fully functional, browser-based web application.

The final application is publicly accessible online and allows users to input customer details and instantly obtain churn predictions.

---

## 🎯 Objectives

- Build an accurate customer churn prediction model
- Apply automated hyperparameter tuning for optimal performance
- Deploy a fully functional browser-based web application
- Provide an intuitive and visually appealing UI/UX

---

## 🧠 Model Description

- **Algorithm**: Artificial Neural Network (ANN)
- **Framework**: TensorFlow / Keras
- **Hyperparameter Tuning**: KerasTuner (Random Search)
- **Loss Function**: Binary Cross-Entropy
- **Optimizer**: Adam
- **Output**: Churn Probability (0–1)

### Final Model Performance
- **Accuracy**: ~94.5%
- **ROC–AUC Score**: ~0.98
- Strong class separation and reliable real-time inference

---

## 📊 Key Input Features

- Tenure  
- Hours Spent on App  
- Number of Devices Registered  
- Satisfaction Score  
- Cashback Amount  

All input features are preprocessed and scaled using the same pipeline as model training.

---

## 🌐 Web Application

- **Framework**: Streamlit
- **Deployment Platform**: Streamlit Community Cloud
- **Access**: Publicly available (no localhost dependency)

### Application Capabilities
- User-friendly input form
- Real-time churn probability prediction
- Clear churn / non-churn classification
- Clean and responsive UI design

---

## 🏗️ Project Structure
├── app.py # Streamlit web application

├── churn_final_model.keras # Final tuned TensorFlow model

├── scaler.pkl # Feature scaler

├── requirements.txt # Python dependencies

├── README.md # Project documentation

----


---

## 🚀 Deployment Details

The application is deployed using Streamlit Community Cloud and integrated with GitHub.

### Deployment Workflow
1. Push all project files to GitHub
2. Connect the GitHub repository to Streamlit Cloud
3. Deploy the `app.py` file
4. Obtain a publicly accessible URL

This deployment approach ensures the application is fully browser-based and accessible online, satisfying academic evaluation requirements.

---

## 📈 Analysis & Justification

- A baseline neural network was initially developed to validate data preprocessing and model feasibility.
- Automated hyperparameter tuning using KerasTuner was applied to obtain the final optimized model.
- The final model achieves a strong balance between:
  - High predictive performance (ROC–AUC ≈ 0.98)
  - Fast inference suitable for real-time web deployment

This makes the solution practical and scalable for real-world e-commerce scenarios.

---

## 🗣️ Presentation Summary (Why, What, How)

### Why
Customer churn leads to revenue loss and reduced customer lifetime value. Early churn prediction enables proactive retention strategies.

### What
A deep learning–based customer churn prediction system with a live, browser-based interface.

### How
Data preprocessing → Neural network modeling → Hyperparameter tuning → Model evaluation → Web deployment using Streamlit.

---

## 🧪 Future Enhancements

- Batch churn prediction using CSV upload
- Explainable AI techniques (SHAP / LIME)
- Integration with CRM or business dashboards
- Enhanced UI/UX for mobile devices

---

## 👨‍💻 Technologies Used

- Python  
- TensorFlow / Keras  
- KerasTuner  
- Scikit-learn  
- Streamlit  
- GitHub  

---

## 📜 License

This project is developed strictly for academic purposes as part of a Deep Learning course.

---

## ✅ Final Note

This project demonstrates:
- An end-to-end deep learning workflow
- Model optimization using automated hyperparameter tuning
- Deployment of an AI model as a real-world web application
- Strong emphasis on usability, performance, and clarity
