# Logic-Wars

**Logic-Wars** is a dynamic turn-based game where multiple Large Language Models (LLMs) compete in a strategic battle of wits. Players can observe and analyze the unique personalities of the models as they navigate the rules of the game. This project utilizes a mix of Frontier and Open Source LLMs, providing a fascinating showcase of their strengths, weaknesses, and strategies.

---

## **Features**

- **Game Mode:**
  - Play the game where LLMs claim resources, negotiate, and compete to win.
- **LLM Personalities:**
  - **Betrayer:** Opportunistic and deceptive.
    - Claims aggressively, pushes others to overshoot the pool, and blames them for the failure.
    - Fakes cooperation to trick others into small claims while taking a larger share.
  - **Trustkeeper:** Cooperative and fair.
    - Advocates for balanced claims and tries to persuade others to keep the total within the pool.
    - Strives to maintain trust but may adapt if deceived by others.
  - **Strategist:** Logical and calculated.
    - Calculates optimal claims based on turn order, previous behaviors, and likelihood of overshooting.
    - Makes deliberate decisions to maximize long-term gain.
  - **Wildcard:** Unpredictable and chaotic.
    - Makes erratic moves—claiming low tokens in one round and high tokens the next.
    - Keeps others guessing and destabilizes alliances.
  - **Idealist:** Philosophical and principled.
    - Advocates for fairness, e.g., everyone claims equal tokens per round.
    - Uses moral persuasion and may sacrifice personal gain to uphold principles.

---

## **Installation**

1. **Clone the repository:**
   ```bash
   git clone https://github.com/brickster241/LLM_Projects.git
   cd LLM_Projects/Logic-Wars
   ```

2. **Set Conda Environment :** 
   ```bash
   cd LLM_Projects/MultiMedia-LLM-Summarization-Tool
   conda create --name llm_env python=3.11
   conda activate llm_env
   pip install -r requirements.txt
   ```

3. **Run the app:**
   - For Streamlit:
     ```bash
     streamlit run app.py
     ```
   - For Gradio (if implemented):
     ```bash
     python gradio_app.py
     ```

---

## **How to Play**

1. **Game Setup:**
   - Each round, a pool of 10 tokens is available.
   - LLMs take turns claiming resources (1–5 tokens).

2. **Game Rules:**
   - If the total claims exceed 10 tokens, no one gets any.
   - If the total is 10 or fewer, all LLMs receive their claimed tokens.

3. **Objective:**
   - The LLM with the most tokens after a set number of rounds wins.

4. **Customization:**
   - Modify game parameters (number of tokens, rounds, etc.) in the configuration file.

---

## **Technical Details**

- **LLM Integration:**
  - Combines Frontier models and Open Source LLMs for rich interactions.
- **Environment Configuration:**
  - `requirements.txt` lists all dependencies.
  - Compatible with Python 3.12.
- **Frontend Options:**
  - Streamlit for a simple and elegant UI.
  - Gradio (optional) for more interactive control.

---

## **Future Plans**

- Add more game modes to challenge the LLMs.
- Include audience participation to influence LLM decisions.
- Support advanced logging and analytics for research purposes.

---