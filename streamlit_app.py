import random
import re
import textwrap
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import streamlit as st

# =========================
# Types & constants
# =========================


@dataclass
class Question:
    id: str
    text: str
    indicators: Dict[str, str]  # "yes"/"no" -> indicator label
    prop: str  # boolean property on a card


@dataclass
class Card:
    id: str
    name: str
    is_ai_system: bool  # hidden in UI until reveal
    description: str
    summary: str
    props: Dict[str, bool] = field(default_factory=dict)


MAX_QUESTIONS = 5

# =========================
# Questions (as specified)
# =========================


QUESTIONS: List[Question] = [
    # 1. Inference
    Question(
        "inf_input",
        "Does this system receive input data to solve a problem?",
        {"yes": "neutral", "no": "not_ai_ind"},
        "receives_input",
    ),
    Question(
        "inf_rules_only",
        "Does it work following only human-programmed rules?",
        {"yes": "not_ai_ind", "no": "def_not_ai"},
        "rules_only",
    ),
    Question(
        "inf_output",
        "Does it generate output data that provides a solution?",
        {"yes": "neutral", "no": "not_ai_ind"},
        "generates_output",
    ),
    Question(
        "inf_influence",
        "Can its outputs influence humans or other systems?",
        {"yes": "neutral", "no": "not_ai_ind"},
        "influences",
    ),

    # 2. Outputs
    Question(
        "out_predicts",
        "Does the system predict something?",
        {"yes": "ai_ind", "no": "not_ai_ind"},
        "predicts",
    ),
    Question(
        "out_creates_content",
        "Does it create content (text, images, video, audio)?",
        {"yes": "ai_ind", "no": "not_ai_ind"},
        "creates_content",
    ),
    Question(
        "out_recommends",
        "Does it provide recommendations (e.g., suggesting actions or products)?",
        {"yes": "ai_ind", "no": "not_ai_ind"},
        "recommends",
    ),
    Question(
        "out_takes_decisions",
        "Does it take decisions directly, without waiting for a human?",
        {"yes": "ai_ind", "no": "neutral"},
        "takes_decisions_direct",
    ),

    # 3. Autonomy
    Question(
        "auton_can_do_on_own",
        "Can the system do something on its own once it receives input (without a person pressing every button)?",
        {"yes": "ai_ind", "no": "def_not_ai"},
        "can_do_on_own",
    ),
    Question(
        "auton_full_autonomy",
        "Does it act with full autonomy, making decisions that directly affect the world without human review?",
        {"yes": "ai_ind", "no": "def_not_ai"},
        "acts_full_autonomy",
    ),
    Question(
        "auton_limited",
        "Does it have limited autonomy (provides outputs but still needs humans to decide or act)?",
        {"yes": "ai_ind", "no": "neutral"},
        "limited_autonomy",
    ),
    Question(
        "auton_non_autonomous",
        "Is it non-autonomous, only working step by step when a human tells it exactly what to do?",
        {"yes": "def_not_ai", "no": "ai_ind"},
        "non_autonomous",
    ),

    # 4. Adaptiveness
    Question(
        "adapt_never_changes",
        "Does the system stay the same and never change how it behaves?",
        {"yes": "neutral", "no": "ai_ind"},
        "never_changes",
    ),
    Question(
        "adapt_online_learns",
        "Does it adapt and learn from new data while it operates?",
        {"yes": "def_ai", "no": "neutral"},
        "adapts_online",
    ),
]

# =========================
# Cards
# =========================


def wrap_desc(s: str) -> str:
    return "\n".join(textwrap.wrap(s, width=100))


BASE_DEFAULTS = {
    "receives_input": True,
    "rules_only": False,
    "generates_output": True,
    "influences": True,
    "predicts": False,
    "creates_content": False,
    "recommends": False,
    "takes_decisions_direct": False,
    "can_do_on_own": True,
    "acts_full_autonomy": False,
    "limited_autonomy": True,
    "non_autonomous": False,
    "never_changes": False,
    "adapts_online": False,
}


def mk_props(**overrides) -> Dict[str, bool]:
    p = BASE_DEFAULTS.copy()
    p.update(overrides)
    return p


CARDS_AI: List[Card] = [
    Card(
        id="ai_chatbot",
        name="Chatbot",
        is_ai_system=True,
        description=wrap_desc(
            "Input: A text message asking to summarize a document. "
            "How it works: learned from millions of texts, images, videos, audio how to answer questions. "
            "Objective: Predict how to respond and generate human-like content. "
            "Output: A written summary of the document."
        ),
        summary=wrap_desc(
            "This system answers user inquiries by summarizing documents based on text messages. "
            "It has been developed using a vast range of information to create human-like responses, ultimately "
            "delivering a concise written summary."
        ),
        props=mk_props(
        "receives_input": True,
        "rules_only": False,
        "generates_output": True,
        "influences": True,
        "predicts": True,
        "creates_content": True,
        "recommends": False,
        "takes_decisions_direct": False,
        "can_do_on_own": True,
        "acts_full_autonomy": False,
        "limited_autonomy": True,
        "non_autonomous": False,
        "never_changes": True,
        "adapts_online": False,
        ),
    ),
    Card(
        id="ai_spam_filter",
        name="Spam Filter",
        is_ai_system=True,
        description=wrap_desc(
            "Input: Incoming email (subject, sender, body text). "
            "How it works: when developed it is shown thousands of emails labeled as spam or not spam. "
            "Objective: Predict whether a message is spam. "
            "Output: the incoming email goes to inbox or spam folder."
        ),
        summary=wrap_desc(
            "This system assesses incoming emails by analyzing their subject, sender, and body content. "
            "It sorts emails into the inbox or spam folder."
        ),
        props=mk_props(
        "receives_input": True,
        "rules_only": False,
        "generates_output": True,
        "influences": True,
        "predicts": True,
        "creates_content": False,
        "recommends": False,
        "takes_decisions_direct": True,
        "can_do_on_own": True,
        "acts_full_autonomy": True,
        "limited_autonomy": False,
        "non_autonomous": False,
        "never_changes": True,
        "adapts_online": False,
        ),
    ),
    Card(
        id="ai_drug_disc",
        name="Drug Discovery System",
        is_ai_system=True,
        description=wrap_desc(
            "Input: Molecular structures and chemical properties. "
            "How it works: a model learned how to identify molecular patterns. "
            "Objective: Predict which molecules act like known drugs. "
            "Output: a list of potential drug candidates."
        ),
        summary=wrap_desc(
            "This tool helps identify potential drug candidates by analyzing molecular structures and chemical "
            "properties. It recognizes patterns within these molecules to suggest which ones may act similarly "
            "to established drugs."
        ),
        props=mk_props(
        "receives_input": True,
        "rules_only": False,
        "generates_output": True,
        "influences": True,
        "predicts": True,
        "creates_content": False,
        "recommends": True,
        "takes_decisions_direct": False,
        "can_do_on_own": True,
        "acts_full_autonomy": False,
        "limited_autonomy": True,
        "non_autonomous": False,
        "never_changes": True,
        "adapts_online": False,
        ),
    ),
    Card(
        id="ai_reco",
        name="Personalized Recommendation System",
        is_ai_system=True,
        description=wrap_desc(
            "Input: a person’s browsing history, searches, clicks, viewing habits. "
            "How it works: it updates recommendations based on how a person behaves online. "
            "Objective: Predict what content/products are most relevant. "
            "Output: suggests items tailored to the user."
        ),
        summary=wrap_desc(
            "This service tailors suggestions for content or products by tracking a user's online behavior, "
            "such as browsing history and viewing habits."
        ),
        props=mk_props(
        "receives_input": True,
        "rules_only": False,
        "generates_output": True,
        "influences": True,
        "predicts": True,
        "creates_content": False,
        "recommends": True,
        "takes_decisions_direct": False,
        "can_do_on_own": True,
        "acts_full_autonomy": False,
        "limited_autonomy": True,
        "non_autonomous": False,
        "never_changes": False,
        "adapts_online": True,
        ),
    ),
    Card(
        id="ai_asr",
        name="Voice-to-Text Assistant",
        is_ai_system=True,
        description=wrap_desc(
            "Input: Spoken audio from user. "
            "How it works: it is developed to understand which typed characters represent a sound. "
            "Objective: Convert speech into text. "
            "Output: transcribed text."
        ),
        summary=wrap_desc(
            "This tool converts spoken audio into written text by interpreting the sounds of human speech. "
            "It facilitates communication by transcribing what is said into a clear text format."
        ),
        props=mk_props(
        "receives_input": True,
        "rules_only": False,
        "generates_output": True,
        "influences": True,
        "predicts": True,
        "creates_content": True,
        "recommends": False,
        "takes_decisions_direct": False,
        "can_do_on_own": True,
        "acts_full_autonomy": False,
        "limited_autonomy": True,
        "non_autonomous": False,
        "never_changes": True,
        "adapts_online": False,
        ),
    ),
    Card(
        id="ai_img_cls",
        name="Image Classifier",
        is_ai_system=True,
        description=wrap_desc(
            "Input: Photo uploaded by user. "
            "How it works: Trained on millions of images. "
            "Objective: Categorize object(s) in the photo. "
            "Output: classify what is in the picture."
        ),
        summary=wrap_desc(
            "This application analyzes uploaded photos to categorize the objects within them."
            "It classifies pictures accurately, for instance, suggesting if an image features a dog or a cat."
        ),
        props=mk_props(
        "receives_input": True,
        "rules_only": False,
        "generates_output": True,
        "influences": True,
        "predicts": True,
        "creates_content": False,
        "recommends": True,
        "takes_decisions_direct": False,
        "can_do_on_own": True,
        "acts_full_autonomy": False,
        "limited_autonomy": True,
        "non_autonomous": False,
        "never_changes": True,
        "adapts_online": False,
        ),
    ),
    Card(
        id="ai_screen",
        name="Job Applicant Screening Tool",
        is_ai_system=True,
        description=wrap_desc(
            "Input: Candidate CVs and application details. "
            "How it works: developed using past hiring outcomes to learn which candidate characteristics are a good fit. "
            "Objective: Predict candidate–job match. "
            "Output: ranked candidates / fit score."
        ),
        summary=wrap_desc(
            "This system evaluates candidate applications by examining CVs and related details. It utilizes "
            "historical hiring data to suggest which candidates are the best fits for job positions, assisting "
            "in the selection process."
        ),
        props=mk_props(
        "receives_input": True,
        "rules_only": False,
        "generates_output": True,
        "influences": True,
        "predicts": True,
        "creates_content": False,
        "recommends": True,
        "takes_decisions_direct": False,
        "can_do_on_own": True,
        "acts_full_autonomy": False,
        "limited_autonomy": True,
        "non_autonomous": False,
        "never_changes": True,
        "adapts_online": False,
        ),
    ),
]


CARDS_NON_AI: List[Card] = [
    Card(
        id="na_excel",
        name="Excel Spreadsheet",
        is_ai_system=False,
        description=wrap_desc(
            "Input: Numbers typed into a sheet. "
            "How it works: Uses formulas or filters written by the user. "
            "Objective: Add, subtract, sort, or calculate. "
            "Output: A table or chart with the requested result."
        ),
        summary=wrap_desc(
            "This tool allows users to input numerical data into a structured sheet where they can perform "
            "calculations and analyses using formulas. It generates tables or charts based on the provided data."
        ),
        props=mk_props(
            rules_only=True,
            predicts=False,
            recommends=False,
            takes_decisions_direct=False,
            can_do_on_own=False,
            limited_autonomy=False,
            non_autonomous=True,
            never_changes=True,
            influences=False,
        ),
    ),
    Card(
        id="na_db_search",
        name="Database Search",
        is_ai_system=False,
        description=wrap_desc(
            "Input: A request like 'find all customers who bought something last month'. "
            "How it works: Follows the search rules exactly as written. "
            "Objective: Retrieve matching entries. "
            "Output: A list of results from the database."
        ),
        summary=wrap_desc(
            "This system retrieves specific data from a database by processing straightforward inquiries. "
            "It follows exact search rules to produce a list of matching entries."
        ),
        props=mk_props(
            rules_only=True,
            predicts=False,
            recommends=False,
            takes_decisions_direct=False,
            can_do_on_own=False,
            limited_autonomy=False,
            non_autonomous=True,
            never_changes=True,
            influences=False,
        ),
    ),
    Card(
        id="na_sales_dash",
        name="Sales Dashboard",
        is_ai_system=False,
        description=wrap_desc(
            "Input: Sales records (region, product, date). "
            "How it works: Uses built-in formulas to add up totals and averages. "
            "Objective: Show how much was sold, where, and when. "
            "Output: Charts and graphs."
        ),
        summary=wrap_desc(
            "This tool compiles and analyzes sales data, displaying totals and averages through built-in "
            "formulas. It provides visual representations like charts and graphs to illustrate sales performance."
        ),
        props=mk_props(
            rules_only=True,
            predicts=False,
            recommends=False,
            takes_decisions_direct=False,
            can_do_on_own=False,
            limited_autonomy=False,
            non_autonomous=True,
            never_changes=True,
            influences=True,
        ),
    ),
    Card(
        id="na_survey",
        name="Survey Summary Tool",
        is_ai_system=False,
        description=wrap_desc(
            "Input: Responses from a questionnaire. "
            "How it works: Counts answers and applies simple statistics. "
            "Objective: Summarize opinions or satisfaction levels. "
            "Output: Percentages and scores."
        ),
        summary=wrap_desc(
            "This system summarizes feedback from questionnaires by counting answers and calculating basic "
            "statistics. It generates percentages and scores to reflect levels of opinion or satisfaction."
        ),
        props=mk_props(
            rules_only=True,
            predicts=False,
            recommends=False,
            takes_decisions_direct=False,
            can_do_on_own=False,
            limited_autonomy=False,
            non_autonomous=True,
            never_changes=True,
            influences=True,
        ),
    ),
    Card(
        id="na_inventory_forecaster",
        name="Inventory Forecaster (Averaging)",
        is_ai_system=False,
        description=wrap_desc(
            "Input: Sales from the past week. "
            "How it works: Takes the average of items sold per day. "
            "Objective: Estimate how many items will be sold tomorrow. "
            "Output: a single number prediction."
        ),
        summary=wrap_desc(
            "This tool uses historical sales data to estimate future sales activity. It calculates average daily "
            "sales to produce a numerical prediction for items expected to be sold on the following day."
        ),
        props=mk_props(
            rules_only=True,
            predicts=True,
            recommends=False,
            takes_decisions_direct=False,
            can_do_on_own=False,
            limited_autonomy=False,
            non_autonomous=True,
            never_changes=True,
            influences=True,
        ),
    ),
    Card(
        id="na_ticket_eta",
        name="Customer Service Time Estimator (Averaging)",
        is_ai_system=False,
        description=wrap_desc(
            "Input: Records of past support tickets. "
            "How it works: Calculates the average time it took to close them. "
            "Objective: Predict how long new tickets will take. "
            "Output: an estimated resolution time."
        ),
        summary=wrap_desc(
            "This tool analyzes past support inquiry records to approximate the expected time for resolving new "
            "tickets. It determines an average closure time to help gauge future support cases."
        ),
        props=mk_props(
            rules_only=True,
            predicts=True,
            recommends=False,
            takes_decisions_direct=False,
            can_do_on_own=False,
            limited_autonomy=False,
            non_autonomous=True,
            never_changes=True,
            influences=True,
        ),
    ),
]


ALL_CARDS: List[Card] = CARDS_AI + CARDS_NON_AI
CARDS_BY_ID: Dict[str, Card] = {c.id: c for c in ALL_CARDS}

# =========================
# Helpers
# =========================


def parse_card_sections(desc: str) -> Dict[str, str]:
    """Parse Input/How it works/Objective/Output sections for pretty display."""
    sections = {"Input": "", "How it works": "", "Objective": "", "Output": ""}
    pattern = r"(Input|How it works|Objective|Output):"
    parts = re.split(pattern, desc)
    current = None
    buf: List[str] = []
    for part in parts:
        if part in sections.keys():
            if current and buf:
                sections[current] = " ".join(buf).strip()
            current = part
            buf = []
        else:
            buf.append(part)
    if current and buf:
        sections[current] = " ".join(buf).strip()
    for k, v in sections.items():
        sections[k] = v.strip()
    return sections


def get_true_answer(card: Card, q: Question) -> str:
    return "yes" if bool(card.props.get(q.prop, False)) else "no"


def reset_game(pick_card_id: Optional[str] = None):
    card = CARDS_BY_ID[pick_card_id] if pick_card_id else random.choice(ALL_CARDS)
    st.session_state.game = {
        "card_id": card.id,
        "asked": [],  # (q_id, computer_answer, indicator)
        "completed": False,
        "user_final_guess": None,  # 'ai' | 'not_ai'
        "user_guess_correct": None,
    }
    st.session_state.pop("last_answer_feedback", None)


# =========================
# Streamlit UI
# =========================


st.set_page_config(page_title="AI Guess Who", page_icon="🃏", layout="centered")
st.title("🃏 AI Guess Who")

# --- Sidebar: new game ---
with st.sidebar:
    st.header("New game")
    st.caption("Draw a new hidden card to challenge yourself again.")
    pick_method = st.radio("Card selection", ["Random draw", "Pick card"], horizontal=True)
    chosen_id = None
    if pick_method == "Pick card":
        sel = st.selectbox(
            "Choose a card", options=ALL_CARDS, format_func=lambda c: c.name
        )
        chosen_id = sel.id

    if st.button("Start new game", type="primary", use_container_width=True):
        reset_game(chosen_id)
        st.rerun()

# Initialize state
if "game" not in st.session_state:
    reset_game()

game = st.session_state.game
current_card = CARDS_BY_ID[game["card_id"]]

st.markdown(
    """
    <style>
    button[data-testid="baseButton-secondary"] {
        background: linear-gradient(135deg, #ffffff, #f5f7fb);
        border: 1px solid rgba(15, 23, 42, 0.12);
        border-radius: 0.9rem;
        box-shadow: 0 10px 25px rgba(15, 23, 42, 0.08);
        padding: 1.2rem 1rem;
        text-align: left;
        white-space: normal;
        line-height: 1.4;
        font-weight: 600;
        color: #0e1117;
    }
    button[data-testid="baseButton-secondary"]:hover {
        border-color: rgba(59, 130, 246, 0.55);
        box-shadow: 0 12px 30px rgba(59, 130, 246, 0.18);
    }
    button[data-testid="baseButton-secondary"]:focus {
        outline: 2px solid rgba(59, 130, 246, 0.45);
        outline-offset: 2px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.subheader("🕵️ Guess the hidden system")
st.write(
    "Ask up to **five** yes/no questions, then decide whether the card below describes an AI system."
)

with st.container(border=True):
    st.markdown("#### System overview")
    st.markdown(f"**{current_card.name}**")
    st.write(current_card.summary)

feedback = st.session_state.pop("last_answer_feedback", None)
if feedback:
    st.success(f"Computer answers: **{feedback['answer'].capitalize()}**")
    st.caption(f"Question: {feedback['question']}")
    if feedback.get("limit_reached"):
        st.info("You've reached the maximum number of questions. Make your final guess below.")

asked_records: List[Tuple[str, str, str]] = game["asked"]
st.progress(
    len(asked_records) / MAX_QUESTIONS,
    text=f"Questions asked: {len(asked_records)} / {MAX_QUESTIONS}",
)

can_ask_more = (
    (len(asked_records) < MAX_QUESTIONS)
    and (not game["completed"])
    and (game["user_final_guess"] is None)
)
remaining = [
    q for q in QUESTIONS if q.id not in {qid for (qid, _, _) in asked_records}
]
if can_ask_more and remaining:
    st.subheader("🗣️ Ask a question")
    st.caption("Select a card to ask the computer about the hidden system.")
    num_cols = 2 if len(remaining) > 1 else 1
    cols = st.columns(num_cols, gap="large")
    for idx, question in enumerate(remaining):
        target_col = cols[idx % num_cols]
        with target_col:
            pressed = st.button(
                question.text,
                key=f"q_btn_{question.id}",
                type="secondary",
                use_container_width=True,
            )
        if pressed:
            ans = get_true_answer(current_card, question)
            indicator_label = question.indicators.get(ans, "neutral")
            game["asked"].append((question.id, ans, indicator_label))
            st.session_state.last_answer_feedback = {
                "answer": ans,
                "question": question.text,
                "limit_reached": len(game["asked"]) >= MAX_QUESTIONS,
            }
            st.rerun()
elif not remaining and not game["completed"] and game["user_final_guess"] is None:
    st.info("You've asked every question. It's time to make your final guess!")

if asked_records:
    st.subheader("📋 Q&A so far")
    for idx, (q_id, ans, _lbl) in enumerate(asked_records, start=1):
        q_obj = next(q for q in QUESTIONS if q.id == q_id)
        st.markdown(f"**Q{idx}.** {q_obj.text}  \n• **Answer:** {ans.capitalize()}")

st.divider()
st.subheader("🎯 Your guess")
if not game["completed"]:
    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        if st.button("Guess: AI system", type="primary", use_container_width=True):
            game["user_final_guess"] = "ai"
            game["completed"] = True
            st.rerun()
    with c2:
        if st.button("Guess: Not AI", type="primary", use_container_width=True):
            game["user_final_guess"] = "not_ai"
            game["completed"] = True
            st.rerun()
    with c3:
        st.caption("You can guess at any time — you don't have to use all five questions.")
else:
    st.caption("You've already locked in your answer. See the reveal below.")

if len(asked_records) >= MAX_QUESTIONS and not game["completed"]:
    st.warning("Maximum questions reached. Make your final guess!")

if game["completed"] and game["user_final_guess"] is not None:
    st.divider()
    st.subheader("🪪 Reveal")
    truth_is_ai = current_card.is_ai_system
    your_guess_ai = game["user_final_guess"] == "ai"
    game["user_guess_correct"] = truth_is_ai == your_guess_ai

    if game["user_guess_correct"]:
        st.success("**Correct!** 🎉")
    else:
        st.error("**Incorrect.**")

    secs = parse_card_sections(current_card.description)
    with st.container(border=True):
        st.markdown(
            f"### {current_card.name} — {'AI System' if truth_is_ai else 'Non-AI System'}"
        )
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Input**")
            st.write(secs.get("Input", ""))
            st.markdown("**Objective**")
            st.write(secs.get("Objective", ""))
        with c2:
            st.markdown("**How it works**")
            st.write(secs.get("How it works", ""))
            st.markdown("**Output**")
            st.write(secs.get("Output", ""))

    with st.expander("Post-game: indicators for each answer"):
        for idx, (q_id, ans, lbl) in enumerate(game["asked"], start=1):
            q_obj = next(q for q in QUESTIONS if q.id == q_id)
            label_text = {
                "def_ai": "Definitive: AI",
                "def_not_ai": "Definitive: Not AI",
                "ai_ind": "AI indicator",
                "not_ai_ind": "Not-AI indicator",
                "neutral": "Neutral",
            }[lbl]
            st.write(f"Q{idx}: {q_obj.text} → {ans.capitalize()} — {label_text}")

    st.divider()
    st.subheader("🔁 Play again")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("New random game", type="primary"):
            reset_game()
            st.rerun()
    with col2:
        options = [f"{c.name}|{c.id}" for c in ALL_CARDS]
        sel = st.selectbox("Or pick specific card", options=options, key="again_pick")
        chosen = sel.split("|")[-1]
        if st.button("Start with chosen card", type="primary"):
            reset_game(chosen)
            st.rerun()

st.markdown("---")
st.caption(
    "Ask smart questions, track the answers, and decide whether the hidden card describes an AI system."
)

