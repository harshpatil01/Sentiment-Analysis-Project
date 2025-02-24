

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
- [Visualizations & Insights](#visualizations--insights)  
- [Future Improvements](#future-improvements)  
- [Conclusion](#conclusion)  

---

## 📂 Dataset Overview  
**Dataset:** *Amazon Fine Food Reviews*  

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

# Split the dataset into training and testing sets
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
