

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
