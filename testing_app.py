import streamlit as st
from PIL import Image
from transformers import AutoModel, AutoTokenizer
import torch
from huggingface_hub import login

login("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")

# Load model and tokenizer
@st.cache_resource  # Cache the model and tokenizer for performance
def load_model():
    model = AutoModel.from_pretrained(
        'openbmb/MiniCPM-V-2_6', 
        trust_remote_code=True,
        attn_implementation='sdpa', 
        torch_dtype=torch.bfloat16
    ).eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-V-2_6', trust_remote_code=True)
    return model, tokenizer

model, tokenizer = load_model()

st.image("product_image.png", caption="Handwritten Text Detection App", use_column_width=True)

# Streamlit UI
st.title("Handwritten Medical Prescription Analysis")
st.write("Upload a medical prescription image and ask a question to extract relevant details.")

# Image uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Question input
question = st.text_input(
    "Enter your question", 
    "The image is a handwritten medical prescription. Extract patient details, diagnosis, medicines, and quantities."
)

if uploaded_file and question:
    # Load and display the uploaded image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Process image and question
    with st.spinner("Processing..."):
        msgs = [{'role': 'user', 'content': [image, question]}]
        try:
            response = model.chat(
                image=image,
                msgs=msgs,
                tokenizer=tokenizer
            )
            st.success("Processing complete!")
            st.subheader("Extracted Information:")
            st.write(response)
        except Exception as e:
            st.error(f"An error occurred: {e}")

st.write("---")
st.markdown(
    """
    **Instructions:**  
    - Upload a clear, readable image of a handwritten medical prescription.
    - Enter a relevant question to extract specific details.
    """
)
