import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer

# Define the model name
model_name = "microsoft/DialoGPT-medium"

# Load the model and tokenizer directly from Hugging Face
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Ensure session state for new input and history is initialized
if 'new_input' not in st.session_state:
    st.session_state.new_input = ""
if 'history' not in st.session_state:
    st.session_state.history = []

# Function to generate a response
def generate_response(input_text):
    inputs = tokenizer.encode(input_text + tokenizer.eos_token, return_tensors="pt")
    reply_ids = model.generate(inputs, max_length=100, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(reply_ids[:, inputs.shape[-1]:][0], skip_special_tokens=True)

# Streamlit app layout
st.title("DialoGPT Chatbot")

# Create input box for the new question
new_question = st.text_input("Ask a question:", value=st.session_state.new_input, key='new_input')

# When the user submits a question
if st.button("Submit"):
    if new_question:  # Ensure there's a question to process
        response = generate_response(new_question)  # Generate the response
        # Update session state after generating the response
        st.session_state.history.append((new_question, response))

# Display previous questions and responses
for question, answer in st.session_state.history:
    st.write(f"**Q:** {question}")
    st.write(f"**A:** {answer}")
