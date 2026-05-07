import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# --- APP CONFIGURATION ---
st.set_page_config(page_title="SentinelMail AI", page_icon="🛡️", layout="wide")

# Custom Styling
st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    .stButton>button { width: 100%; border-radius: 20px; height: 3em; background-color: #ff4b4b; color: white; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# --- HEADER ---
st.title("🛡️ SentinelMail AI")
st.subheader("Enterprise-Grade Spam Detection for Business")
st.markdown("---")

# --- DATA & MODEL LOGIC ---
@st.cache_resource
def load_and_train():
    # Load dataset
    df = pd.read_csv("spam.csv", encoding='latin-1')
    df = df.iloc[:, [0, 1]]
    df.columns = ['Category', 'Message']
    
    # Map 'spam' to 0 and 'ham' to 1
    df['Label'] = df['Category'].map({'spam': 0, 'ham': 1})
    
    # Split Data
    X_train, X_test, Y_train, Y_test = train_test_split(df['Message'], df['Label'], test_size=0.2, random_state=3)
    
    # Transform text to numbers (TF-IDF)
    feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
    X_train_features = feature_extraction.fit_transform(X_train)
    X_test_features = feature_extraction.transform(X_test)
    
    # Train Model (Logistic Regression)
    model = LogisticRegression()
    model.fit(X_train_features, Y_train)
    
    # Calculate Accuracy
    acc = accuracy_score(Y_test, model.predict(X_test_features))
    return model, feature_extraction, acc

# Initialize Model
try:
    model, v_model, accuracy = load_and_train()
    
    # --- UI LAYOUT ---
    col1, col2 = st.columns([2, 1])

    with col1:
        st.write("### 📝 Analyze Message")
        input_text = st.text_area("Enter the email or SMS content below:", placeholder="Type or paste here...", height=200)
        
        if st.button("Verify Security"):
            if input_text.strip():
                # Prediction
                user_input_features = v_model.transform([input_text])
                prediction = model.predict(user_input_features)
                
                if prediction[0] == 1:
                    st.balloons()
                    st.success("### ✅ RESULT: CLEAN (HAM)\nThis message is verified as safe.")
                else:
                    st.error("### 🚨 RESULT: SPAM DETECTED\nWarning: This message shows patterns of a scam/spam.")
            else:
                st.warning("Please enter text to analyze.")

    with col2:
        st.write("### 📊 Model Performance")
        st.metric(label="Model Accuracy", value=f"{accuracy*100:.2f}%")
        st.info("""
        **How it works:**
        Using **Logistic Regression**, the AI analyzes the 'weight' of words. 
        Spammy words like 'Winner' or 'Free' trigger the detector based on historical patterns in the dataset.
        """)

except Exception as e:
    st.error("Error loading data. Ensure 'spam.csv' is uploaded to GitHub.")

st.markdown("---")
st.caption("BBA AI Course Project | Developed with Python & Streamlit")
