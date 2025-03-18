# 📊 LG Aimers - Display Defect Prediction  

This repository contains the code and related materials from the **LG Aimers** competition, where the goal was to develop a machine learning model to predict **defective displays** in the manufacturing process. The competition focused on improving quality control by detecting potential defects early.  

🏆 **Ranked 62nd out of 1123 participants** in the competition.  

## 🔍 Project Overview  

- **Objective**: Predict defective displays using data collected from the manufacturing process  
- **Technologies Used**: Python, XGBoost, Optuna  
- **Pipeline**: Data preprocessing → Feature engineering → Model training & tuning → Performance evaluation  
- **Optimization Strategies**: Hyperparameter tuning with Optuna (1000 trials) and Seed Ensemble  
- **Evaluation Metric**: F1-score  
- **Training Environment**: Google Colab (T4 GPU)  

⚠ **Note**: Due to competition regulations, the dataset used in this project **cannot** be shared.  

## 📁 Directory Structure  

```
📂 LG_Aimers_Display_Defect
│── preprocess.py    # Data preprocessing module  
│── optimize.py      # Hyperparameter tuning module (Optuna 1000 trials)  
│── train.py         # Model training module (XGBoost with Seed Ensemble)  
│── inference.py     # Inference module  
│── main.py          # Main script integrating all modules  
│── requirements.txt # Required dependencies  
│── README.md        # Project documentation  
```

## 🚀 Key Highlights  

1. **Data Preprocessing (`preprocess.py`)**  
   - Handling missing values and outliers  
   - Feature scaling and selection  

2. **Hyperparameter Optimization (`optimize.py`)**  
   - Optuna (1000 trials) for tuning XGBoost hyperparameters  

3. **Model Training (`train.py`)**  
   - XGBoost as the primary model  
   - Seed Ensemble for performance improvement  
   - **Trained using Google Colab's T4 GPU**  

4. **Inference (`inference.py`)**  
   - Running predictions on new data  

5. **Main Execution (`main.py`)**  
   - Integrates all modules and runs the complete pipeline  
   - ⚠ **Users must set the `PATH` variable in `main.py` to specify their dataset location**  

## 📌 Usage Instructions  

1. Clone the repository:  
   ```bash
   git clone https://github.com/your_username/LG_Aimers_Display_Defect.git
   cd LG_Aimers_Display_Defect
   ```

2. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```

3. Set the dataset path in `main.py`:  
   ```python
   PATH = "your/dataset/path.csv"
   ```

4. Run the full pipeline:  
   ```bash
   python main.py
   ```

## 🏆 Results & Insights  

- **Final Model**: XGBoost with Seed Ensemble  
- **Best Hyperparameters**: Optimized using Optuna (1000 trials)  
- **Performance Metric**: Evaluated using F1-score  
- **Feature Importance Analysis**  

### 🔸 Handling Class Imbalance  
Since the dataset was collected from real-world manufacturing data, **defective samples were extremely rare**, leading to a severe **target imbalance** problem.  
To address this issue, significant effort was put into balancing the dataset, including using **`scale_pos_weight`** in XGBoost.  

### 🔸 Why XGBoost?  
Among various models tested, **XGBoost was chosen** because:  
1. **Tree-based models** were the most practical under the given constraints.  
2. **Google Colab's T4 GPU** was used for training, and XGBoost was well-suited for this environment.  

## 📜 License  

This project is licensed under the [MIT License](LICENSE).  
