 
import streamlit as st
import pickle

# Load the model and vectorizer
with open('naive_bayes_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('count_vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Streamlit UI
st.title('Spam Message Classifier')

st.write("Enter a message to classify whether it is Spam or Ham:")

# User input
user_input = st.text_area("Your Message")

# Function to predict spam or ham
def predict_spam(text):
    text_vec = vectorizer.transform([text])
    prediction = model.predict(text_vec)
    return "Spam" if prediction[0] == 1 else "Ham"

# Predict when the user submits input
if st.button('Classify'):
    if user_input:
        result = predict_spam(user_input)
        st.success(f"Prediction: {result}")
    else:
        st.warning("Please enter a message to classify.")
