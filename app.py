import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')  # Download WordNet for lemmatization
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer  # Importing WordNetLemmatizer
import re
import string
import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import joblib
from sklearn.feature_extraction.text import CountVectorizer
import seaborn as sns

# initialize Lemmatizer
lemmatizer = WordNetLemmatizer()

# loading trained model (random forest) and vectorizer (tf-idf)
model_stress = joblib.load('rf_model_stress.pkl')
model_anxiety = joblib.load('rf_model_anxiety.pkl')
model_depression = joblib.load('rf_model_depression.pkl')
vectorizer = joblib.load('tfidf_vectorizer-2.pkl')

# loading preprocessed dataset
data = pd.read_csv("Lemmatized_Twitter_Texts.csv")

# function for text cleaning with lemmatization
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+|@[^\s]+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    
    # apply lemmatization to each token
    lemmatized_text = ' '.join([lemmatizer.lemmatize(word) for word in word_tokens if word.lower() not in stop_words])
    
    # remove non-alphanumeric characters after lemmatization
    lemmatized_text = re.sub(r'[^a-zA-Z0-9\s]', '', lemmatized_text)
    
    return lemmatized_text

# streamlit app
st.set_page_config(page_title="HeadSpace", layout="wide")

# tab navigation
with st.sidebar:
    selected_tab = option_menu(
        menu_title="HeadSpace Navigation",
        options=["Welcome", "Share Your Thoughts", "What Should I Do Next?", "About the Creator"],
        icons=["house", "chat", "question-circle", "info-circle"],
        default_index=0,
        orientation="vertical",
    )

# custom style for titles
def apply_custom_styles():
    custom_css = """
    /* Import Poppins font from Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');

    /* Center the title and change the font */
    h1 {
        text-align: center;
        font-family: 'Poppins', sans-serif !important;
        font-size: 68px !important;  /* Adjust size as needed */
        color: rgb(33, 174, 174) !important;  /* Dark Teal */
    }

    /* Make sure the streamlit container also has no padding */
    .block-container {
        padding: 0 !important;
    }
    """
    st.markdown(f"<style>{custom_css}</style>", unsafe_allow_html=True)

# call function to apply custom styles
apply_custom_styles()

# tab 1
if selected_tab == "Welcome":

    st.image("logo.png", width=1500)

    st.markdown("<h1>Welcome to HeadSpace!</h1>", unsafe_allow_html=True)

    # display the statistic fact and image
    col1, col2, col3 = st.columns([1, 2, 1])  # Adjust the middle column for centering
    with col2:
        st.image("statistic.png", width=700)

    # adding the statistic text below the image
    st.markdown("""
    <style>
        /* Import Poppins font from Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');
        
        /* Center the text and apply styles */
        .center-text {
            text-align: center;
            font-family: 'Poppins', sans-serif !important;
            color: #2F4F4F;  /* Dark teal color */
            line-height: 1.6; /* Adjust line spacing */
        }
        .title {
            font-size: 48px !important; /* Larger font size for title */
            font-weight: 600; /* Bold title */
            margin-bottom: 5px; /* Add spacing below the title */
        }
        .subtitle {
            font-size: 32px !important; /* Smaller font size for supporting text */
        }
    </style>
    <div class="center-text">
        <p class="title">Did You Know?</p>
        <p class="subtitle">According to the National Health and Morbidity Survey 2023,<br>
        1 in 6 children aged five to 15 in Malaysia suffers from mental health problems.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <style>
        /* Import Poppins font from Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');
        
        /* Style for the section title */
        .center-text {
            text-align: center;
            font-family: 'Poppins', sans-serif !important;
            color: #2F4F4F; /* Dark teal color */
            line-height: 1.6; /* Adjust line spacing */
        }
        .title {
            font-size: 48px !important; /* Larger font size for title */
            font-weight: 600; /* Bold title */
            margin-bottom: 20px; /* Add spacing below the title */
        }
        /* Style for the boxes */
        .box-container {
            display: flex;
            justify-content: center;
            gap: 20px;
            margin-top: 20px;
        }
        .box {
            background-color:rgb(245, 201, 192); /* Light pink background */
            border: 1px solid #D3D3D3; /* Light gray border */
            border-radius: 10px;
            padding: 20px;
            width: 25%; /* Box width */
            text-align: center;
            font-family: 'Poppins', sans-serif !important;
            font-size: 20px;
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1); /* Subtle shadow */
        }
    </style>
    <div class="center-text">
        <p class="title">Why HeadSpace?</p>
    </div>
    <div class="box-container">
        <div class="box">
            <p><strong>Identify Your Symptoms</strong></p>
            <p>We help you recognise early signs of mental health challenges and provide clarity about your emotional state.</p>
        </div>
        <div class="box">
            <p><strong>Support and Guidance</strong></p>
            <p>Our platform guides you towards the support you may not realise you need, empowering you to take the first step.</p>
        </div>
        <div class="box">
            <p><strong>Raise Awareness</strong></p>
            <p>We aim to increase understanding of mental health issues and foster a healthier, more informed community in Malaysia.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <style>
        /* Import Poppins font from Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');
        
        /* Style for the section title */
        .center-text {
            text-align: center;
            font-family: 'Poppins', sans-serif !important;
            color: #2F4F4F; /* Dark teal color */
            line-height: 1.6; /* Adjust line spacing */
        }
        .title {
            font-size: 48px !important; /* Larger font size for title */
            font-weight: 600; /* Bold title */
            margin-bottom: 20px; /* Add spacing below the title */
        }

        /* Style for the manual list */
        .manual-list {
            font-family: 'Poppins', sans-serif;
            font-size: 20px;
            color: #2F4F4F;
            list-style-type: decimal;
            line-height: 1.8;
            padding-left: 30px;
        }

        .manual-list li {
            margin-bottom: 10px;
        }
    </style>
    <div class="center-text">
        <p class="title"> </p>
        <p class="title">How to Use HeadSpace?</p>
    </div>
    
    <ul class="manual-list">
        <li><strong>Step 1:</strong> Click on the ">" button on the top left of the app and navigate to the "Share Your Thoughts" tab.</li>
        <li><strong>Step 2:</strong> Express yourself by typing your thoughts, and we will classify the result for you.</li>
        <li><strong>Step 3:</strong> Wondering what you should do next? Take yourself to the "What Should I Do Next?" tab for helpful guidance.</li>
        <li><strong>Step 4:</strong> Learn more about the creator and the project by visiting the "About the Creator" tab.</li>
    </ul>
""", unsafe_allow_html=True)


# tab 2
elif selected_tab == "Share Your Thoughts":

    st.markdown("<br>" * 1, unsafe_allow_html=True)  # add 1 new line

    st.title("Share Your Thoughts")

    st.markdown("""
    <style>
        /* Import Poppins font from Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');
        
        /* Style the input text area */
        .stTextArea textarea {
            font-family: 'Poppins', sans-serif !important;
            font-size: 20px !important;
            text-align: center;
            padding: 15px;
        }
    </style>
    """, unsafe_allow_html=True)

    # taking user input
    user_input = st.text_area(" --------------------------------------------------- ", 
        "Write your thoughts or share your feelings! Express yourself here...")

    # customise button
    st.markdown("""
    <style>
        /* Import Poppins font from Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');
        
        /* Style the button */
        .stButton button {
            font-family: 'Poppins', sans-serif !important; /* change font */
            font-size: 28px !important; /* increase font size */
            padding: 15px 30px; /* increase padding for a bigger button */
            background-color: #008080; /* turqoise background */
            color: white;
            border: none;
            border-radius: 8px;
        }

        .stButton button:hover {
            background-color:rgb(38, 109, 127); /* darker green when hovering */
        }

        /* Center the button */
        .stButton {
            display: flex;
            justify-content: center;
            align-items: center;
            margin-top: 20px; /* Add margin for spacing */
        }
    </style>
    """, unsafe_allow_html=True)

    # button to analyze text
    if st.button("Analyze Text"):
        if not user_input.strip():  # check if the input is empty or only contains whitespace
            st.error("OH UH, it's empty! Please share your thoughts before we proceed.")
        elif len(user_input.strip()) < 5:  # check if the input is too short
            st.warning("Hmm, your input is quite short. Could you elaborate a bit more?")
        else:
            try:
                # clean the text and process predictions
                cleaned_text = clean_text(user_input)  # Apply cleaning with lemmatization
                vectorized_input = vectorizer.transform([cleaned_text])
                prediction_stress = model_stress.predict(vectorized_input)
                prediction_anxiety = model_anxiety.predict(vectorized_input)
                prediction_depression = model_depression.predict(vectorized_input)
 
                severity_map = {
                    '0 - Normal' : 'No signs of mental health issues detected.',
                    '1 - Mild': 'Mild signs detected.',
                    '2 - Moderate' : 'Moderate signs detected.',
                    '3 - Severe' : 'Severe signs detected.'
                }

                st.markdown("""
                    <style>
                    /* Import Poppins font from Google Fonts */
                    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');
        
                    /* Style for Prediction Results title */
                    .prediction-title {
                        font-family: 'Poppins', sans-serif !important;
                        font-size: 32px !important; 
                        font-weight: bold;
                        text-align: center;
                        color: #2F4F4F;  
                    }

                    /* Style for Prediction results */
                    .prediction-result {
                        font-family: 'Poppins', sans-serif !important;
                        font-size: 20px !important; 
                        color: #2F4F4F;  
                    }
                </style>
                """, unsafe_allow_html=True)

                # display Prediction Results with styling
                st.markdown('<p class="prediction-title">Prediction Results</p>', unsafe_allow_html=True)

                # display stress, anxiety, and depression results with styled font
                st.markdown(f'<p class="prediction-result"><b>Stress Level</b>: {severity_map[prediction_stress[0]]}</p>', unsafe_allow_html=True)
                st.markdown(f'<p class="prediction-result"><b>Anxiety Level</b>: {severity_map[prediction_anxiety[0]]}</p>', unsafe_allow_html=True)
                st.markdown(f'<p class="prediction-result"><b>Depression Level</b>: {severity_map[prediction_depression[0]]}</p>', unsafe_allow_html=True)

                severity_map2 = {
                    '0 - Normal' : 0,
                    '1 - Mild': 1,
                    '2 - Moderate' : 2,
                    '3 - Severe' : 3
                }

                # Plot severity levels
                st.markdown('<p class="prediction-title">Severity Level Breakdown</p>', unsafe_allow_html=True)
                conditions = ['Stress', 'Anxiety', 'Depression']
                severity_values = [severity_map2[prediction_stress[0]], severity_map2[prediction_anxiety[0]], severity_map2[prediction_depression[0]]]
                fig = plt.figure(figsize=(8, 6))
                sns.barplot(x=conditions, y=severity_values, palette='coolwarm')
                plt.title('Mental Health Condition Severity Levels')
                plt.ylabel('Severity Level')
                plt.ylim(0, 3)
                st.pyplot(fig)

            except Exception as e:
                st.error(f"An error occurred during analysis: {e}")

# tab 3
elif selected_tab == "What Should I Do Next?":

    st.markdown("<br>" * 1, unsafe_allow_html=True)  # add 1 new line

    st.title("What Should I Do Next?")

    st.markdown("""
    <style>

     /* Card Styling */
        .card {
            width: 22%;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            text-align: center;
            font-family: 'Poppins', sans-serif !important;
            cursor: pointer; /* Indicate interactivity */
            margin-bottom: 20px;
        }

        .card h4 {
            font-weight: bold;
        }

        .no-signs { background-color: #d0f8d0; }
        .mild-signs { background-color: #fdf2b4; }
        .moderate-signs { background-color: #f8d0b8; }
        .severe-signs { background-color: #f8b0b0; }
    </style>
    """, unsafe_allow_html=True)

    # Interactivity with Cards
    col1, col2, col3, col4 = st.columns(4)  # Four columns for four cards

    with col1:
        if st.button("No signs detected"):
            st.info("You're doing great! Keep it up and maintain a healthy lifestyle.")

    with col2:
        if st.button("Mild signs detected"):
            st.warning("It's okay to feel this way sometimes. Talk to a friend or take some time for self-care.")

    with col3:
        if st.button("Moderate signs detected"):
            st.warning("This could be concerning. Consider seeking support from loved ones or a professional.")

    with col4:
        if st.button("Severe signs detected"):
            st.error("Please reach out to a mental health professional immediately. You are not alone.")

    # Helpline Link
    st.markdown("""
    <div style="text-align: center; margin-top: 20px;">
        <a href="https://findahelpline.com/countries/my" target="_blank" style="font-family: 'Poppins', sans-serif; font-size: 18px; color: #007BFF;">
            Consider reaching out to Find Helpline (Malaysia) to find someone to talk to.
        </a>
    </div>
    """, unsafe_allow_html=True)

# tab 4
elif selected_tab == "About the Creator":
    
    st.markdown("<br>" * 1, unsafe_allow_html=True)  # add 1 new line
    
    st.title("About the Creator")

    st.markdown("""
    <style>
        .header {
            font-family: 'Poppins', sans-serif;
            font-size: 28px;
            text-align: center;
            margin-bottom: 20px;
        }

        .subheader {
            font-family: 'Poppins', sans-serif;
            font-size: 24px;
            text-align: center;
            margin-bottom: 20px;
            font-style: italic;
        }
        .content {
            font-family: 'Poppins', sans-serif !important;
            font-size: 18px !important;
            text-align: center;
            margin-bottom: 20px;
            line-height: 2.0;  /* adjust line height for better readability */
        }
        .image-container {
            display: flex;
            justify-content: center;
            flex-wrap: wrap;
            gap: 10px;
            margin-top: 20px;
        }
    </style>
    
    <div class="header">
        SHAMIERA BATRISYIA BILQIS BINTI BADRUL HISHAM
    </div>
        
    <div class="subheader">
        3rd Year Data Science Student at Universiti Malaya
    </div>
    
    <div class="content">
        <p>My Multi Level Classification in Mental Health Detection project is my first Data Science Project where I aimed to develop a mental health detector that provides multiclass classification. Most mental health predictors focus only on one target, but for this project, there are three different targets: Stress, Anxiety, and Depression, with four different classifications ranging from normal, mild, moderate, to severe.</p>
    <div class="content">
        <p>The dataset I used was extracted from the Twitter API. It originally contained 19,000 rows of Twitter texts from users in Malaysia and underwent expert annotation to classify the severity of stress, anxiety, and depression based on the text, allowing us to have a DASS dataset.</p>
    <div class="content">
        <p>During data preprocessing, I generated a bar graph to see the top words associated with each class of the target.</p>
    <div class="content">
        <p>These top words may affect the result of predictions. If you notice any discrepancies (e.g., if the model predicts severity incorrectly), please seek professional help immediately.</p>
    <div class="content">
        <p>Thank you for checking out my project.</p>
    </div>
    """, unsafe_allow_html=True)

    st.title("Top Words for Stress")

    # display images for stress
    st.image('Stress0.png', use_container_width=True)
    st.image('Stress1.png', use_container_width=True)
    st.image('Stress2.png', use_container_width=True)
    st.image('Stress3.png', use_container_width=True)

    st.title("Top Words for Anxiety")

    # display images for anxiety
    st.image('Anxiety0.png', use_container_width=True)
    st.image('Anxiety1.png', use_container_width=True)
    st.image('Anxiety2.png', use_container_width=True)
    st.image('Anxiety3.png', use_container_width=True)

    st.title("Top Words for Depression")

    # display images for depression
    st.image('Depression0.png', use_container_width=True)
    st.image('Depression1.png', use_container_width=True)
    st.image('Depression2.png', use_container_width=True)
    st.image('Depression3.png', use_container_width=True)
