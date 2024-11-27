# **An Efficient Deep Learning Method to Predict Crop Yield**

## **Project Overview**
This project focuses on using efficient deep learning methods to predict crop yields accurately. Models like Feed Forward Neural Networks (FNN), Long Short-Term Memory (LSTM), Convolutional Neural Networks (CNN), Gated Recurrent Unit (GRU), Recurrent Neural Network (RNN), Multi Layer Perceptron (MLP) were evaluated for their ability to analyze agricultural and environmental data, offering insights for sustainable agriculture.

---

## **Features**
- Crop yield prediction using key factors such as crop type, season, area, prodction ,state ,crop year, rainfall,  pesticide and fertilizer use.  
- Comparison of multiple deep learning models for accuracy and performance.  
- Utilization of evaluation metrics like MAE, MSE, and R².  
- Visual representation of results using scatterplots and R² analysis.  

---

## **Dataset**
- **Source**: Kaggle Indian Agriculture Crop Production database.  
- **Size**: 19,689 entries with 10 key columns.  
- **Key Features**:  
  - Crop Type, Season, State, Area, Production, Crop Year, Rainfall, Fertilizer & Pesticide Usage.  
  - Target Variable: Crop Yield (tons/ha).  

---

## **System Workflow**
1. **Data Collection**: Extract data from reliable agricultural sources.  
2. **Data Preprocessing**:  
   - **Scaling Numerical Features**: Normalize values to a comparable range (e.g., 0 to 1).  
   - **Encoding Categorical Variables**: Convert non-numeric columns (e.g., Crop, Season, State) into numeric format using label encoding.  
   - **Handling Missing Values**: Apply mean or median imputation for numerical columns and mode imputation for categorical ones.  
   - **Feature and Target Separation**: Split the dataset into independent variables (features) and the dependent variable (target: crop yield).  
   - **Data Splitting**: Divide data into training (60%), validation (20%), and testing (20%) sets.  
3. **Model Training**:  
   - Train six models with ReLU/Sigmoid activation functions.  
   - Use early stopping to prevent overfitting.  
4. **Prediction Module**:  
   - Feed preprocessed data into trained models.  
   - Generate crop yield predictions.  
5. **Evaluation**:  
   - Analyze model performance using metrics like MAE, MSE, and R². 

---

## **Key Results**
| **Model**              | **MAE**   | **MSE**   | **R²**    | **Interpretation**                                                                                   |
|------------------------|-----------|-----------|-----------|------------------------------------------------------------------------------------------------------|
| **FNN**                | 0.03688   | 0.1656    | 83.79%    | Best performer, capturing complex relationships and showing strong generalization.                  |
| **LSTM**               | 0.02746   | 0.1722    | 83.15%    | Excellent for temporal data, capturing trends effectively.                                          |
| **CNN**                | 0.02538   | 0.1766    | 82.72%    | Excels in spatial feature analysis but lacks temporal modeling.                                     |
| **GRU**                | 0.04062   | 0.2397    | 76.55%    | Simpler than LSTM, leading to faster training but slightly less accurate predictions.               |
| **RNN**                | 0.05598   | 0.5127    | 49.84%    | Moderate accuracy, struggling with long-term dependencies.                                          |
| **MLP**                | 0.05448   | 0.5369    | 47.47%    | Lowest performance due to its simpler architecture, serving as a baseline.                          |

---

## **Libraries Used**
1. **Deep Learning**:  
   - TensorFlow, Keras, PyTorch.  
2. **Data Processing**:  
   - Pandas, NumPy, Scikit-learn.  
3. **Visualization**:  
   - Matplotlib, Seaborn.  

---

## **System Requirements**
### **Hardware**:  
- Processor: Intel Core i7/i9 or AMD Ryzen 7/9.  
- RAM: Minimum 8GB, recommended 16GB.  
- Storage: 256GB SSD (minimum), 512GB SSD (recommended).  

### **Software**:  
- IDE: Google Colab (GPU-enabled).  
- Language: Python 3.  

---

## **Future Work**
1. **Integration of Additional Data Sources**:  
   - Include real-time data (e.g., soil health, microclimates).  
2. **Regional and Crop-Specific Models**:  
   - Develop models tailored to specific crops and regions for improved accuracy.  

---

