import tensorflow as tf
import streamlit as st

pt_en_translator = tf.saved_model.load('models/saved_models/translator')

def translate(input_sentence):
    return pt_en_translator(input_sentence).numpy().decode('utf-8')

# App title
st.title("Tradutor PT-EN")

# Function to update translation
def update_translation():
    input_text = st.session_state["input_text"]
    if input_text.strip():
        translated_text = translate(input_text)
    else:
        translated_text = ""
    st.session_state["translated_text"] = translated_text

# Input text area for Portuguese
st.text_area(
    "Texto em Português",
    height=70,
    key="input_text",
    on_change=update_translation,  # Trigger translation when input changes
)

# Output text area for English translation (read-only)
st.text_area(
    "Tradução em Inglês",
    value=st.session_state.get("translated_text", ""),
    height=70,
    key="output_text",
    disabled=True,  # Makes this text area read-only
)