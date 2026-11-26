import streamlit as st
import joblib
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# ✅ Set Streamlit Page Configuration
st.set_page_config(page_title="Sentiment Analysis App", page_icon="💬", layout="wide")

# ✅ Check if required files exist before loading
required_files = [
    "logistic_regression_model.pkl", 
    "random_forest_model.pkl", 
    "xgboost_model.pkl",
    "svm_model.pkl",
    "decision_tree_model.pkl",
    "sgd_classifier_model.pkl",
    "tfidf_vectorizer.pkl",
    "Reviews.csv"
]

missing_files = [f for f in required_files if not os.path.exists(f)]
if missing_files:
    st.error(f"🚨 Missing Files: {missing_files}.\nPlease ensure all model files are present in the working directory.")
    st.stop()  # Stop execution if files are missing

# ✅ Cache model loading for faster performance
@st.cache_resource
def load_models():
    return {
        "Logistic Regression": joblib.load("logistic_regression_model.pkl"),
        "Random Forest": joblib.load("random_forest_model.pkl"),
        "XGBoost": joblib.load("xgboost_model.pkl"),
        "SVM": joblib.load("svm_model.pkl"),
        "Decision Tree": joblib.load("decision_tree_model.pkl"),
        "SGD Classifier": joblib.load("sgd_classifier_model.pkl"),
    }

models = load_models()

# ✅ Load TF-IDF Vectorizer (cached)
@st.cache_resource
def load_vectorizer():
    return joblib.load("tfidf_vectorizer.pkl")

vectorizer = load_vectorizer()

# ✅ Load Dataset (Cache for Speed)
@st.cache_resource
def load_data():
    df = pd.read_csv("Reviews.csv", usecols=["Text", "Score"]).dropna()
    df["Sentiment"] = df["Score"].apply(lambda x: 1 if x > 3 else 0)
    return df

df = load_data()

# ✅ Cache TF-IDF transformation
@st.cache_resource
def prepare_tfidf():
    X = df["Text"]
    y = df["Sentiment"]
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    return X_test_tfidf, y_test

X_test_tfidf, y_test = prepare_tfidf()

# ✅ Function to evaluate models
def model_evaluation(model, model_name):
    y_pred = model.predict(X_test_tfidf)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Plot Confusion Matrix
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix - {model_name}")
    st.pyplot(fig)
    
    # Show Model Performance Scores
    st.write(f"### Model Performance - {model_name}")
    st.write(f"**Accuracy:** {report['accuracy']:.3f}")
    st.write(f"**Precision (Positive Class):** {report['1']['precision']:.3f}")
    st.write(f"**Recall (Positive Class):** {report['1']['recall']:.3f}")
    st.write(f"**F1-Score (Positive Class):** {report['1']['f1-score']:.3f}")

# ✅ Streamlit UI
st.markdown("<h1 style='text-align: center; color: #FF4B4B;'>📝 Sentiment Analysis Web App</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: gray;'>Analyze customer reviews using different ML models</h4>", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["📊 Model Performance", "📈 Exploratory Data Analysis (EDA)"])

# ✅ Tab 1 - Model Performance & Confusion Matrix
with tab1:
    selected_model = st.selectbox("Select a Model to Evaluate:", list(models.keys()))
    model_evaluation(models[selected_model], selected_model)

# ✅ Tab 2 - EDA (Exploratory Data Analysis)
with tab2:
    st.subheader("📌 Dataset Overview")
    st.write("First 5 rows of the dataset:")
    st.dataframe(df.head())

    # Sentiment Distribution
    st.subheader("📊 Sentiment Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x=df["Sentiment"], palette=["#FF4B4B", "#32CD32"], ax=ax)
    ax.set_xticklabels(["Negative", "Positive"])
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Sentiments in Dataset")
    st.pyplot(fig)

    # Word Cloud
    st.subheader("☁️ Most Frequent Words in Reviews")
    text = " ".join(df["Text"].astype(str)[:10000])  # Limiting to 10k rows for performance
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)

    fig_wc, ax_wc = plt.subplots(figsize=(10, 5))
    ax_wc.imshow(wordcloud, interpolation="bilinear")
    ax_wc.axis("off")
    st.pyplot(fig_wc)

    # Review Length Analysis
    st.subheader("📏 Review Length Distribution")
    df["Review_Length"] = df["Text"].apply(lambda x: len(str(x).split()))

    fig_len, ax_len = plt.subplots()
    sns.histplot(df["Review_Length"], bins=30, kde=True, color="purple", ax=ax_len)
    ax_len.set_xlabel("Number of Words in Review")
    ax_len.set_ylabel("Frequency")
    ax_len.set_title("Distribution of Review Lengths")
    st.pyplot(fig_len)

st.markdown("---")
st.markdown("<h5 style='text-align: center; color: gray;'>📌 Developed with ❤️ using Streamlit & Machine Learning</h5>", unsafe_allow_html=True)