import streamlit as st
from hf_mcq_generator import HF_MCQGenerator
from dotenv import load_dotenv

load_dotenv()
# Initialize the MCQ Generator
mcq_generator = HF_MCQGenerator()

# Sidebar Title and Description
st.sidebar.write("""
## Welcome to the **LLM-Based MCQ Paper Generator**! 🎉

This app uses a powerful language model to generate multiple-choice question papers tailored to your needs. Here's what you can do:
- ✏️ Enter a domain of your choice (e.g., Mathematics, Physics, Coding, etc.).
- 🎓 Select a difficulty level: Student, Graduate, Professional, or Expert.
- 📄 Get a beautifully formatted MCQ paper in seconds!

Start by entering your details below. 🚀
""")

# Initialize Model and Tokenizer
if mcq_generator.model is None or mcq_generator.tokenizer is None:
    with st.sidebar:
        with st.spinner("Initializing the Model and Tokenizer. This may take a moment... 🛠️"):
            try:
                # Initialize Model and Tokenizer
                mcq_generator.login()
                mcq_generator.setup_model()
                mcq_generator.setup_tokenizer()
                st.success("Model and tokenizer initialized successfully! ✅")

                # Input Fields
                st.sidebar.subheader("Input Parameters : ")
                domain = st.sidebar.text_input("Enter the domain : ", placeholder="e.g., Mathematics, Physics, Coding")
                difficulty = st.sidebar.selectbox("Select Difficulty Level : ", ["Student", "Graduate", "Professional", "Expert"])
                paperBtn = st.button("Generate Paper")
                
            except Exception as e:
                st.error(f"An error occurred during initialization: {e}")

# Button to Trigger Paper Generation
if mcq_generator.model is not None and mcq_generator.tokenizer is not None:
    
    if paperBtn:
        if not domain:
            st.error("Please enter a valid domain.")
        else:
            # Generate Paper
            try:
                with st.sidebar:
                    with st.spinner("Generating your MCQ paper... ✨"):
                        paper = mcq_generator.generate_paper(domain, difficulty)
                        st.markdown(paper)
                        st.success("MCQ Paper generated successfully! ✅")

            except Exception as e:
                st.error(f"An error occurred: {e}")
