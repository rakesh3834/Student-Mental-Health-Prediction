import streamlit as st
import pandas as pd
import re
import nltk
import pickle as pk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Download required NLTK data
@st.cache_resource
def download_nltk_data():
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    return True

download_nltk_data()
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# Loading both model and vectorizer
@st.cache_resource
def load_model():
    try:
        model = pk.load(open('CV_BestModel.sav', 'rb'))
        vectorizer = pk.load(open('vectorizer.pkl', 'rb'))
        return model, vectorizer
    except Exception as e:
        st.error(f"Error loading model or vectorizer: {str(e)}")
        return None, None

# Text cleaning function for user input
def clean_input(raw_text):
    text = re.sub('[^a-zA-Z]', ' ', str(raw_text))
    text = text.lower().split()
    lemmas = [lemmatizer.lemmatize(word) for word in text if word not in stop_words]
    return ' '.join(lemmas)

model, vectorizer = load_model()

if model is None or vectorizer is None:
    st.error("Failed to load model or vectorizer. Please check if both files exist.")
    st.stop()

# Function to predict anxiety or depression
def detect_anxiety_depression(input_text):
    try:
        input_text_lower = input_text.lower()

        # if any(word in input_text_lower for word in ["depression", "depress", "depressed"]):
        #     col1, col2 = st.columns(2)
        #     col1.write("Model Prediction:")
        #     col2.markdown(":red[UnFit]")
        #     return
        
        transformed_text = vectorizer.transform([input_text])
        print(transformed_text)
        result = model.predict(transformed_text)[0]
        print(result)
        
        col1, col2 = st.columns(2)
        col1.write("Model Prediction:")
        if result == 1:
            col2.markdown(":red[UnFit]")
        else:
            col2.markdown(":green[Fit]")
            
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        st.write("Debug info:")
        st.write(f"Input text: {input_text}")
        st.write(f"Model type: {type(model)}")
        st.write(f"Vectorizer type: {type(vectorizer)}")

st.markdown(
    """
    <style>
    .header-style {
        font-size:25px;
        font-family:sans-serif;
        position:absolute;
        text-align: center;
        color: #032131;
        top: 0px;
    }
    .font-style {
        font-size:20px;
        font-family:sans-serif;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Initialize session state
if "page" not in st.session_state:
    st.session_state.page = "home"

# Landing Page
if st.session_state.page == "home":
    st.title("Welcome to the Student Mental Health Prediction")
    st.write("""
        This web application helps in detecting whether the students might be experiencing anxiety or depression
        based on the sentence they input. Click the button below to proceed to the analysis.
    """)
    if st.button("Go to Analysis"):
        st.session_state.page = "analyze"

# Analysis Page
if st.session_state.page == "analyze":
    st.header("Anxiety/Depression Detection")
    
    # Create a form for input
    with st.form(key='analysis_form'):
        user_sentence = st.text_input("Enter a Sentence")
        submit_button = st.form_submit_button('Analyze')
        
    if submit_button and user_sentence:
        cleaned_sentence = clean_input(user_sentence)
        
        # Show the preprocessing steps
        with st.expander("See text preprocessing steps"):
            st.write("Original text:", user_sentence)
            st.write("Cleaned text:", cleaned_sentence)
        
        detect_anxiety_depression(cleaned_sentence)

    if st.button("Go Back to Home"):
        st.session_state.page = "home"