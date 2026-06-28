# 🚦 Road Traffic Accident (RTA) Severity Prediction

A data science project that analyzes road traffic accident data to identify patterns related to road conditions, weather, time of day, and driver behavior — and predicts accident severity using a KNN classification model.

---

## 📂 Dataset

**File:** `RTA_Dataset.csv`  
**Total Records:** 12,316 rows | 32 columns  
**Target Variable:** `Accident_severity`

| Class | Description |
|---|---|
| Slight Injury | Minor injuries |
| Serious Injury | Significant injuries |
| Fatal Injury | Death involved |

**Key Features:**

| Feature | Description |
|---|---|
| `Age_band_of_driver` | Driver age group |
| `Driving_experience` | Years of experience |
| `Weather_conditions` | Weather at time of accident |
| `Light_conditions` | Lighting during accident |
| `Road_surface_type` | Type of road surface |
| `Types_of_Junction` | Junction type |
| `Cause_of_accident` | Primary cause |
| `Number_of_casualties` | Number of people injured |
| `Number_of_vehicles_involved` | Vehicles involved |

---

## 🛠️ Technologies Used

- **Python 3**
- **Pandas, NumPy** – data loading & manipulation
- **Matplotlib, Seaborn** – data visualization
- **Scikit-learn** – label encoding, chi-square feature selection, KNN model, evaluation metrics
- **Imbalanced-learn (SMOTE)** – oversampling for class imbalance

---

## 📊 Project Workflow

### 1. Data Loading & Exploration
- Loaded dataset, checked shape, data types, and statistical summary

### 2. Handling Missing Values
- Dropped columns with >2500 missing values (`Service_year_of_vehicle`, `Defect_of_vehicle`, `Work_of_casuality`, `Fitness_of_casuality`, `Time`)
- Filled remaining categorical nulls with mode

### 3. Data Visualization
- Accident severity distribution (count plot)
- Scatter plot & joint plot: casualties vs vehicles involved
- Correlation heatmap for numerical features
- Count plots for all categorical features

### 4. Feature Selection (Chi-Square Test)
- Applied Label Encoding on categorical features
- Used Chi-Square test to rank features by F-score and p-value
- Dropped 10 low-importance features: `Owner_of_vehicle`, `Type_of_vehicle`, `Road_surface_conditions`, `Pedestrian_movement`, `Casualty_severity`, `Educational_level`, `Day_of_week`, `Sex_of_driver`, `Road_allignment`, `Sex_of_casualty`

### 5. Encoding & Preprocessing
- Applied One-Hot Encoding (`get_dummies`) on remaining categorical columns
- Separated features (`X`) and target (`y`)

### 6. Handling Class Imbalance
- Applied **SMOTE** (Synthetic Minority Oversampling Technique) to balance all 3 classes

### 7. Model Training — KNN
- Train/Test split: **70% / 30%**
- Model: `KNeighborsClassifier(n_neighbors=5)`

### 8. Evaluation
- Classification Report (Precision, Recall, F1-Score)
- Accuracy Score
- Confusion Matrix with visual display

---

## 🚀 How to Run

1. **Clone the repository**
   ```bash
   git clone <https://github.com/Siddhi-Shinde-dev/PRODIGY_DS_05>
   cd rta-accident-analysis
   ```

2. **Install dependencies**
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
   ```

3. **Place the dataset**  
   Make sure `RTA Dataset.csv` is in the same directory as the notebook.

4. **Run the notebook**
   ```bash
   jupyter notebook DS_Task05.ipynb
   ```

---

## 📁 Project Structure

```
rta-accident-analysis/
│
├── RTA_Dataset.csv        
├── DS_Task05.ipynb        
└── README.md             
```

---

## 📈 Key Insights

- Most accidents involve **2 vehicles** and result in **1 casualty**
- No strong direct correlation between number of casualties and vehicles involved
- **SMOTE** applied to handle class imbalance across 3 severity levels
- Chi-Square feature selection reduced dimensionality by removing 10 low-impact features

---

## 🙋‍♀️ Author

**Siddhi Shinde**  
