# Microorganism Classification Using Machine Learning 
**Authors**: Andrei Dolmatov, Aleksandra Karabutova  
**Course**: Hands-on Machine Learning (Summer Semester 2024)  

---

## 📌 Project Description  
This project classifies **microorganisms (algae)** from microscopic images using machine learning. Algae serve as ecological indicators, helping assess environmental stress. 
Furthermore, the same models can by used in some diagnostic applications in which very rapid identification of a particular gene or genetic species becomes essential, while identification of all genes is not necessary. For example, in patients with septic shock from bacterial infections, identification of antibiotic-resistance genes is essential because the mortality rate increases 7.6% per hour of delay in administering correct antibiotics. Unfortunately, it takes more than 24 h to grow up the bacteria recovered from the blood of an infected patient, identify the species, and then determine to which antibiotics the organism is resistant, leading to a very high mortality rate for such infections.
Bacterial antibiotic resistance is becoming a significant health threat, and rapid identification of antibiotic-resistant bacteria is essential to save lives and reduce the spread of antibiotic resistance.

**Goal**: Accurately classify algae species based on **24 morphological features** (e.g., area, perimeter, eccentricity).  

### Key Challenges  
- 🎯 **Imbalanced dataset** (10 unevenly distributed classes).  
- 🔄 **84% duplicate data** (only 4,874 unique samples out of 30,527).  
- ⚖️ **Feature scaling/transformation** required (outliers, skewed distributions).  

---

## 🗃️ Dataset  
- **Source**: Kaggle (https://www.kaggle.com/datasets/sayansh001/microbes-dataset).  
- **Samples**: 30,527 (4,874 unique after deduplication).  
- **Features**: Area, Perimeter, Centroid, Solidity, Eccentricity, Convex Hull, EquivDiameter, Extent, etc.

---

## ⚙️ Preprocessing  
### 1. Handling Duplicates  
We tested all models on **two dataset versions**:  
- **With Duplicates**: 30,527 samples (84% duplicates)  
- **Without Duplicates**: 4,874 unique samples  
- Removed duplicates **after train-test split** to avoid leakage.  
- **Finding**: Models trained *with duplicates* consistently outperformed those without, suggesting duplicates may represent augmented/pre-processed data rather than noise.

### 2. Balancing Classes  
- **SMOTE**: Synthetic oversampling for minority classes.  

### 3. Feature Scaling  
- Best transformer: `PowerTransformer` (Yeo-Johnson).  
- Alternative: `RobustScaler` (for outlier resistance).  

---

## 📊 Model Comparison (Full Results)  

### 1️⃣ **With Duplicates**  
| Model             | Macro Recall | Macro F1 | Cohen’s Kappa |  
|-------------------|--------------|----------|---------------|  
| Gradient Boosting | 0.9772       | 0.9848   | 0.9901        |  
| Random Forest     | 0.9678       | 0.9723   | 0.9810        |  
| KNN               | 0.9642       | 0.9655   | 0.9758        |  
| SVC               | 0.9616       | 0.9662   | 0.9762        |  
| MLP               | 0.9638       | 0.9680   | 0.9749        |  

### 2️⃣ **Without Duplicates**  
| Model             | Macro Recall | Macro F1 | Cohen’s Kappa |  
|-------------------|--------------|----------|---------------|  
| Gradient Boosting | 0.7290       | 0.7656   | 0.7408        |  
| Random Forest     | 0.5672       | 0.5279   | 0.5101        |  
| KNN               | 0.4660       | 0.4030   | 0.3489        |  
| SVC               | 0.4976       | 0.4413   | 0.3875        |  
| MLP               | 0.4500       | 0.4424   | 0.4337        |  

**Performance Drop**: Removing duplicates reduced metrics by **30-60%** across models.  

---

### Key Insights  
- 🏆 **Gradient Boosting** outperformed others.

---

## 🚀 How to Run  
1. Clone the repo:  
 ```bash
 git clone https://github.com/your-username/repo-name.git
 cd microbes-ml
 jupyter notebooks:
- Main_With duplicates.ipynd - EDA and models for data *with* duplicates
- Add_Without duplicates.ipynb - EDA and models for data *without* duplicates

microbes.csv - data