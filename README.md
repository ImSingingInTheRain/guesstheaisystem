# 🃏 AI Guess Who

AI Guess Who is a single-player Streamlit game that helps people practice telling apart AI systems from
traditional software. Each round the app draws a card describing a real or fictional technology (you can
also pick the specific card you want to play from the sidebar). Your goal is to decide whether it qualifies
as an AI system.

## How to play

1. Review the short "System overview" summary for the hidden card.
2. Ask up to five yes/no questions from the curated list. The computer answers each question and labels it
   with how strongly it indicates the card is AI or not.
3. Make your call at any time by choosing **Guess: AI system** or **Guess: Not AI**.
4. Reveal the result to see whether you were correct, read a detailed description of the system, and quickly
   start a new round (random or with a specific card).

The questions reflect the EU AI Act definition of an AI system. Use the answers to gather clues about how the
system behaves, how autonomous it is, and whether it adapts over time.

## Running locally

1. Install the requirements:

   ```bash
   pip install -r requirements.txt
   ```

2. Start the Streamlit app:

   ```bash
   streamlit run streamlit_app.py
   ```

Enjoy the game!

