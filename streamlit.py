import streamlit as st
from transformers import BartForConditionalGeneration, BartTokenizer

def summarize_text(text):
    # Load the tokenizer and model
    model_name = "facebook/bart-large-cnn"
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)
    
    # Tokenize the input text
    inputs = tokenizer(text, max_length=1024, truncation=True, return_tensors="pt")
    
    # Generate the summary
    summary_ids = model.generate(inputs["input_ids"], min_length=20, max_length=100, num_beams=5, early_stopping=True)
    
    # Decode the generated summary
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    return summary

def main():
    st.title("TextSqueeze")
    st.write("Enter the text you want to summarize below:")

    text = st.text_area("Input Text", height=300)

    if st.button("Summarize"):
        if text.strip():
            summary = summarize_text(text)
            st.write("### Summary")
            st.write(summary)
        else:
            st.write("Please enter some text to summarize.")

if __name__ == "__main__":
    main()
