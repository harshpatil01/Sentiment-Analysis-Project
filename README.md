

# 📝 Sentiment Analysis of Customer Reviews Using NLP & Machine Learning  

## 📌 Overview  
This project leverages **Natural Language Processing (NLP)** and **Machine Learning** techniques to analyze and classify **Amazon fine food reviews** as **Positive or Negative**. The dataset contains over **568,454 customer reviews spanning 10+ years**.  
We compare multiple ML models to find the best-performing one and deploy the final model as a **Streamlit Web Application**.  

---

## 📊 Table of Contents
- [Dataset Overview](#dataset-overview)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Data Preprocessing](#data-preprocessing)
- [Machine Learning Models](#machine-learning-models)
- [Model Evaluation & Results](#model-evaluation--results)
- [Deployment Using Streamlit](#deployment-using-streamlit)
- [Installation & Usage](#installation--usage)
- [Conclusion](#conclusion)


---

## 📂 Dataset Overview  
**Dataset:** *Amazon Fine Food Reviews*  
https://snap.stanford.edu/data/web-FineFoods.html

### **📌 Context**  
This dataset consists of **Amazon fine food reviews**, covering a period of **10+ years** (1999 - 2012). The dataset includes **product and user information, review scores, and plain text reviews**. It also contains reviews from **all Amazon categories**.  

### **📌 Data Summary**  
✅ **Reviews from:** *October 1999 - October 2012*  
✅ **Total Reviews:** *568,454*  
✅ **Total Users:** *256,059*  
✅ **Total Products:** *74,258*  

---

### **📌 Features:**  
| Feature | Description |
|---------|------------|
| **Id** | Unique identifier for each review |
| **ProductId** | Unique identifier for the product |
| **UserId** | Unique identifier for the reviewer |
| **ProfileName** | Name of the reviewer |
| **HelpfulnessNumerator** | Number of helpful votes |
| **HelpfulnessDenominator** | Total votes for helpfulness |
| **Score** | Rating (1 to 5) |
| **Summary** | Short review title |
| **Text** | Full review content |

### **📌 Sentiment Labels:**  
- **Positive (Score > 3)**  
- **Negative (Score ≤ 3)**  

---

## 🔎 Exploratory Data Analysis (EDA)  

### **🔹 Steps Performed:**  
✅ Checking for missing values  
✅ Analyzing sentiment distribution  
✅ Finding the most commonly used words  

### 📈 **Visuals & Code**  

#### **1️⃣ Distribution of Review Scores**  
```python
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8,5))
sns.countplot(x=df['Score'], palette="coolwarm")
plt.title("Distribution of Review Scores")
plt.xlabel("Score (1-5)")
plt.ylabel("Count")
plt.show()
```
📌 Insight: More positive reviews than negative ones.

## 📊 Exploratory Data Analysis (EDA)

### **2️⃣ Most Frequent Words in Reviews**
```python
from wordcloud import WordCloud
import matplotlib.pyplot as plt

text = " ".join(review for review in df['Text'])
wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()
```
📌 Insight: Common words include “good”, “great”, “love”, “bad”, “terrible”.

## 🛠 Data Preprocessing  

### **🔹 Steps Taken:**  
✅ Removing Stopwords & Punctuation  
✅ Tokenization & Lemmatization  
✅ TF-IDF Vectorization for Feature Extraction  
✅ Convert Scores into Binary Sentiment Labels  

---

### **🔹 Converting Scores into Sentiment Labels**
```python
df["Sentiment"] = df["Score"].apply(lambda x: 1 if x > 3 else 0)  # 4-5 → Positive, 1-2 → Negative
```

## 🔹 TF-IDF Vectorization  

To convert the text reviews into numerical representations, **TF-IDF (Term Frequency-Inverse Document Frequency)** was used. This helps in understanding the importance of words in the dataset.

### **🔹 Implementing TF-IDF Vectorization**
```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=5000)  # Keep top 5000 words
X_tfidf = vectorizer.fit_transform(df["Text"])
```

## 🤖 Machine Learning Models  

In this project, multiple **Machine Learning models** were implemented to classify customer reviews as **Positive or Negative**.

### **🔹 Implemented Models:**  
✅ **Logistic Regression** - A simple and efficient classification model for text data.  
✅ **Support Vector Machine (SVM)** - Works well for text classification, particularly in high-dimensional spaces.  
✅ **Decision Tree Classifier** - A rule-based approach to classify sentiments, but prone to overfitting.  
✅ **Random Forest Classifier** - An ensemble learning method that reduces overfitting and improves accuracy.  
✅ **XGBoost** - A powerful gradient boosting algorithm that outperformed other models.  
✅ **Stochastic Gradient Descent (SGD)** - Fast and efficient for large-scale text classification.  

📌 **Each model was trained, evaluated, and compared based on accuracy, precision, recall, and F1-score.**  
**XGBoost achieved the highest accuracy (90%)**, making it the best-performing model in this project. 🚀  

## 🔹 Training Machine Learning Models  

After preprocessing the dataset and converting text reviews into **TF-IDF features**, multiple **Machine Learning models** were trained to classify sentiments.

### **🔹 Splitting the Dataset**  
The dataset was split into **80% training** and **20% testing** to evaluate model performance effectively.

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, df["Sentiment"], test_size=0.2, random_state=42)

# Define models
models = {
    "Logistic Regression": LogisticRegression(),
    "SVM": SVC(kernel="linear"),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss"),
    "SGD Classifier": SGDClassifier(loss="hinge"),
}

# Train models
for name, model in models.items():
    model.fit(X_train, y_train)
```

🔹 **Why These Models?**

Each model has a unique strength:
	•	**Logistic Regression** - Simple and interpretable for binary classification.
	•	**SVM** - Effective in high-dimensional spaces.
	•	**Decision Tree** - Captures non-linear patterns.
	•	**Random Forest** - Reduces overfitting and improves accuracy.
	•	**XGBoost** - A powerful gradient boosting algorithm that improves performance.
	•	**SGD Classifier** - Efficient for large-scale text data.

 ## 📊 Model Evaluation & Results  

After training the models, we evaluated their performance using key classification metrics:  
- **Accuracy**: Measures overall correctness of predictions.  
- **Precision**: Percentage of correctly predicted positive sentiments out of all predicted positives.  
- **Recall**: Percentage of correctly predicted positive sentiments out of all actual positives.  
- **F1-score**: Harmonic mean of precision and recall.  

### 🔹 Performance Comparison  

| Model                | Accuracy | Precision | Recall | F1-score |
|----------------------|----------|----------|--------|----------|
| Logistic Regression | 85%      | 84%      | 85%    | 85%      |
| SVM                 | 88%      | 87%      | 88%    | 88%      |
| Random Forest       | 87%      | 86%      | 87%    | 87%      |
| XGBoost             | 90%      | 89%      | 90%    | 90%      |
| Decision Tree       | 82%      | 80%      | 82%    | 81%      |

📌 **Insight:**  
- **XGBoost** performed the best with **90% accuracy**, making it the top choice for deployment.  
- **SVM and Random Forest** also performed well, achieving above **87% accuracy**.  
- **Decision Tree** had the lowest performance due to overfitting on the training data.


## 🔹 Confusion Matrices for Each Model  

Confusion matrices help us visualize the performance of each model by showing the number of correctly and incorrectly classified instances.

📌 **Understanding the Confusion Matrix:**  
- **True Positives (TP):** Correctly predicted positive sentiments.  
- **True Negatives (TN):** Correctly predicted negative sentiments.  
- **False Positives (FP):** Incorrectly predicted positive sentiments.  
- **False Negatives (FN):** Incorrectly predicted negative sentiments.  

### **🔹 Generating Confusion Matrices for All Models**  

```python
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Function to plot confusion matrix
def plot_conf_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

# Generate confusion matrices for each model
for name, model in models.items():
    y_pred = model.predict(X_test)
    plot_conf_matrix(y_test, y_pred, name)
```




## 🌐 Deployment Using Streamlit  

To make the sentiment analysis model accessible, we deployed it using **Streamlit**, allowing users to interact with the model through a simple web interface.


### 🔹 How to Run the App Locally  

```sh
streamlit run app/deployment.py
```

## Installation & Usage

### Clone the Repository

```sh
git clone https://github.com/your-username/Sentiment-Analysis-Project.git
cd Sentiment-Analysis-Project
```

## 🏆 Conclusion

🎯 ***XGBoost*** performed the best, achieving 90% accuracy.
🎯 ***NLP-based Sentiment Analysis*** provides valuable insights into customer opinions.
🎯 ***Future Improvements*** can enhance accuracy, scalability, and real-time performance.
