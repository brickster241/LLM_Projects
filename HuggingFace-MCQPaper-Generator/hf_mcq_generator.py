import os
import torch
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer, BitsAndBytesConfig

class HF_MCQGenerator:
    """
    HF_MCQGenerator is a class designed to generate multiple-choice question (MCQ) papers using a Hugging Face language model.

    Key Features:
    - Login to the Hugging Face Hub.
    - Initialize and configure a language model and tokenizer with quantization support.
    - Generate MCQ papers based on a specified domain and difficulty.
    """

    def __init__(self):
        self.model_name = "meta-llama/Llama-3.1-8B-Instruct"
        self.model = None
        self.tokenizer = None
        self.quant_config = None

    def login(self, token_env_var="HF_TOKEN"):
        """Logs into the Hugging Face Hub using a token from an environment variable."""
        self.hf_token = os.getenv(token_env_var)
        if not self.hf_token:
            raise ValueError("Hugging Face token not found. Please set it in the environment variables.")
        login(token=self.hf_token, add_to_git_credential=True)

    def setup_model(self):
        """Initializes the model with quantization configuration."""
        self.quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=self.model_name,
            device_map="auto",
            quantization_config=self.quant_config,
            trust_remote_code=True
        )

    def setup_tokenizer(self):
        """Initializes the tokenizer and sets the pad token."""
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate_paper(self, domain, difficulty):
        """Generates an MCQ paper based on the specified domain and difficulty."""
        if not self.model or not self.tokenizer:
            raise ValueError("Model and tokenizer must be initialized before generating a paper.")

        SYSTEM_PROMPT = """You are an intelligent assistant tasked with creating multiple-choice question (MCQ) papers based on domain and difficulty.\nDomain can include fields like Mathematics, Physics, Chemistry, Coding Languages (Python, Java, C++, etc.), Algorithms, General Science, Logical Reasoning, etc.\nDifficulty should represent the expertise level, such as Student, Graduate, Professional, or Expert.\nYour primary goals are:\n1. Generate 15 diverse MCQs in Markdown format related to the topic.\n2. Ensure questions and answers are correct and relevant to the specified domain and difficulty.\n3. Maintain an introductory section outlining general instructions including total marks & negative marking for the paper.\n4. Format each question with 4 answer choices labeled (A), (B), (C), and (D).\n5. Avoid repetition and ensure a balanced distribution of challenges and topics within the domain.\n6. When generating questions, focus on clarity, diversity, and accuracy. \nRespond in Markdown. Example Output : {EXAMPLE_OUTPUT}"""

        EXAMPLE_OUTPUT = """# Practice Paper\n\n## Instructions:\n1. This paper contains **15 multiple-choice questions (MCQs)**, each worth **4 marks**.  \n2. **Negative marking**: 1 mark will be deducted for each incorrect answer.  \n3. There is **no negative marking** for unattempted questions.  \n4. Time allotted: **1 hour**.  \n5. Use only standard methods, concepts, or tools relevant to the given topic.  \n\n---\n\n### Question 1  \nIf the roots of the quadratic equation \( x^2 - 3x + 2 = 0 \) are \( \alpha \) and \( \beta \), then the value of \( \alpha^2 + \beta^2 \) is:  \n- (A) 5  \n- (B) 9  \n- (C) 13  \n- (D) 15  \n\n---\n\n### Question 2  \nA body is thrown vertically upward with a speed of \( 20 \, \text{m/s} \). How high does it rise before coming to rest? (Take \( g = 10 \, \text{m/s}^2 \))  \n- (A) 10 m  \n- (B) 20 m  \n- (C) 30 m  \n- (D) 40 m  \n"""

        USER_PROMPT = f"Generate a MCQ Paper based on domain : {domain} and difficulty : {difficulty}. Respond the formatted MCQ paper in Markdown."
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT}
        ]

        # Tokenize input & Generate Response
        model_inputs = self.tokenizer.apply_chat_template(conversation=messages, return_tensors="pt").to("cuda")
        model_output = self.model.generate(model_inputs, max_new_tokens=5000)
        response = self.tokenizer.decode(model_output[0])

        # Remove the special token Headers
        return response.split("<|end_header_id|>")[-1].strip().replace("<|eot_id|>","")
