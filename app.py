import streamlit as st
import pickle

model = pickle.load(open("spam_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.set_page_config(
    page_title="Spam Email Classifier",
    page_icon="📧",
    layout="centered"
)

st.markdown("""
    <style>
        .stApp {
            background: linear-gradient(135deg, #89f7fe 0%, #66a6ff 100%);
            background-attachment: fixed;
        }
        .main-box {
            background-color: white;
            border-radius: 20px;
            padding: 40px;
            box-shadow: 0 6px 15px rgba(0,0,0,0.1);
            margin-top: 30px;
        }
        h1 {
            text-align: center;
            color: #1B4F72;
            font-size: 40px;
            margin-bottom: 20px;
        }
        .stButton>button {
            background-color: #1B4F72;
            color: white;
            font-size: 18px;
            border-radius: 10px;
            padding: 10px 30px;
            border: none;
            transition: 0.3s;
        }
        .stButton>button:hover {
            background-color: #2874A6;
            transform: scale(1.05);
        }
        textarea {
            border-radius: 10px !important;
            border: 2px solid #2874A6 !important;
        }
        [data-testid="stSidebar"] {
            background-color: #F2F4F4;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1>📧 Spam Email Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Detect whether an email or message is <b>Spam</b> or <b>Not Spam</b> using AI 🤖</p>", unsafe_allow_html=True)

st.markdown("<div class='main-box'>", unsafe_allow_html=True)

user_input = st.text_area("✉️ Enter your email or message below:", height=150)

if st.button("🔍 Check Message"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a message first!")
    else:
        input_vector = vectorizer.transform([user_input])
        prediction = model.predict(input_vector)[0]
        proba = model.predict_proba(input_vector)[0]
        confidence = round(max(proba) * 100, 2)

        if prediction == 1:
            st.error(f"🚨 This message is **SPAM!**\n\n🤖 Confidence: {confidence}%")
        else:
            st.success(f"✅ This message is **NOT SPAM.**\n\n🤖 Confidence: {confidence}%")

st.markdown("</div>", unsafe_allow_html=True)

st.sidebar.header("📘 About This App")
st.sidebar.write("""
This AI-powered tool detects spam messages and phishing attempts using a trained
machine learning model.

**Technologies used:**
- Streamlit
- Scikit-learn
- Python
- Natural Language Processing (NLP)
""")

st.sidebar.markdown("---")
st.sidebar.info("Developed with ❤️ by Madhuri")
