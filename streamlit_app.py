import html
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

QUESTION_STATUS_STYLES = {
    "yes": {"bg": "#dcfce7", "border": "#bbf7d0", "text": "#166534"},
    "no": {"bg": "#fee2e2", "border": "#fecaca", "text": "#991b1b"},
}

QUESTION_ICONS: Dict[str, str] = {
    "inf_input": "📥",
    "inf_rules_only": "📜",
    "inf_output": "📤",
    "inf_influence": "🌐",
    "out_predicts": "🔮",
    "out_creates_content": "🎨",
    "out_recommends": "🤝",
    "out_takes_decisions": "⚙️",
    "auton_can_do_on_own": "🚀",
    "auton_full_autonomy": "🤖",
    "auton_limited": "🕹️",
    "auton_non_autonomous": "🧭",
    "adapt_never_changes": "🧱",
    "adapt_online_learns": "📈",
}

INDICATOR_DETAILS: Dict[str, Dict[str, str]] = {
    "def_ai": {
        "title": "Strong AI evidence",
        "description": "This answer makes it very clear the hidden card describes an AI system.",
        "bg": "#e0f2fe",
        "border": "#bae6fd",
        "text": "#0c4a6e",
        "icon": "💡",
    },
    "def_not_ai": {
        "title": "Strong NOT-AI evidence",
        "description": "This answer strongly signals the system is not AI-driven.",
        "bg": "#fee2e2",
        "border": "#fecaca",
        "text": "#7f1d1d",
        "icon": "⛔",
    },
    "ai_ind": {
        "title": "Leaning toward AI",
        "description": "This answer suggests the card likely represents an AI system, though more clues help.",
        "bg": "#ecfdf5",
        "border": "#bbf7d0",
        "text": "#065f46",
        "icon": "🤖",
    },
    "not_ai_ind": {
        "title": "Leaning toward NOT-AI",
        "description": "This answer leans toward the system not being AI, but it's not definitive.",
        "bg": "#fef3c7",
        "border": "#fde68a",
        "text": "#92400e",
        "icon": "🧩",
    },
    "neutral": {
        "title": "Neutral clue",
        "description": "Both AI and non-AI systems commonly can share this trait.",
        "bg": "#f1f5f9",
        "border": "#e2e8f0",
        "text": "#1e293b",
        "icon": "⚖️",
    },
}

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
        {"yes": "def_not_ai", "no": "ai_ind"},
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
        {"yes": "ai_ind", "no": "neutral"},
        "creates_content",
    ),
    Question(
        "out_recommends",
        "Does it provide recommendations (e.g., suggesting actions or products)?",
        {"yes": "ai_ind", "no": "neutral"},
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
        "Can the system do something on its own once it receives input (a human does not have to tell it exactly what to do and how to do it)?",
        {"yes": "ai_ind", "no": "def_not_ai"},
        "can_do_on_own",
    ),
    Question(
        "auton_full_autonomy",
        "Does it act with full autonomy, making decisions that directly affect the world without human review?",
        {"yes": "def_ai", "no": "neutral"},
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
            "Input: Detailed user prompts and any documents or snippets they want summarized. "
            "How it works: A large language model trained on diverse texts reasons over the prompt to identify key themes and supporting details. "
            "Objective: Produce an accurate, friendly response that distills the material into an easy-to-read summary. "
            "Output: Polished written paragraphs capturing the main points of the source material."
        ),
        summary=wrap_desc(
            "This system answers user inquiries by summarizing documents based on text messages. "
            "It has been developed using a vast range of information to create human-like responses, ultimately "
            "delivering a concise written summary."
        ),
        props=mk_props(
            receives_input=True,
            rules_only=False,
            generates_output=True,
            influences=True,
            predicts=True,
            creates_content=True,
            recommends=False,
            takes_decisions_direct=False,
            can_do_on_own=True,
            acts_full_autonomy=False,
            limited_autonomy=True,
            non_autonomous=False,
            never_changes=True,
            adapts_online=False,
        ),
    ),
    Card(
        id="ai_spam_filter",
        name="Spam Filter",
        is_ai_system=True,
        description=wrap_desc(
            "Input: Incoming email metadata such as sender, subject line, body text, and embedded links. "
            "How it works: A supervised model compares each message against patterns learned from thousands of labeled spam and safe emails. "
            "Objective: Judge the likelihood that the message is unwanted or malicious. "
            "Output: Automatically routes the email into the inbox or spam folder with an optional warning flag."
        ),
        summary=wrap_desc(
            "This system assesses incoming emails by analyzing their subject, sender, and body content. "
            "It sorts emails into the inbox or spam folder."
        ),
        props=mk_props(
            receives_input=True,
            rules_only=False,
            generates_output=True,
            influences=True,
            predicts=True,
            creates_content=False,
            recommends=False,
            takes_decisions_direct=True,
            can_do_on_own=True,
            acts_full_autonomy=True,
            limited_autonomy=False,
            non_autonomous=False,
            never_changes=True,
            adapts_online=False,
        ),
    ),
    Card(
        id="ai_drug_disc",
        name="Drug Discovery System",
        is_ai_system=True,
        description=wrap_desc(
            "Input: Molecular structures, chemical descriptors, and lab screening results. "
            "How it works: It analyzes structure–activity relationships learned from prior experiments to spot promising molecular motifs. "
            "Objective: Prioritize compounds that behave like effective drugs and warrant further testing. "
            "Output: Ranked shortlists of candidate molecules with scores or annotations."
        ),
        summary=wrap_desc(
            "This tool helps identify potential drug candidates by analyzing molecular structures and chemical "
            "properties. It recognizes patterns within these molecules to suggest which ones may act similarly "
            "to established drugs."
        ),
        props=mk_props(
            receives_input=True,
            rules_only=False,
            generates_output=True,
            influences=True,
            predicts=True,
            creates_content=False,
            recommends=True,
            takes_decisions_direct=False,
            can_do_on_own=True,
            acts_full_autonomy=False,
            limited_autonomy=True,
            non_autonomous=False,
            never_changes=True,
            adapts_online=False,
        ),
    ),
    Card(
        id="ai_reco",
        name="Personalized Recommendation System",
        is_ai_system=True,
        description=wrap_desc(
            "Input: Browsing activity, search history, ratings, and recent purchases for each person. "
            "How it works: Collaborative and content-based algorithms update preference profiles as people interact with content. "
            "Objective: Surface items the user is most likely to enjoy or act on next. "
            "Output: Dynamic recommendation carousels, emails, or notifications tailored to the individual."
        ),
        summary=wrap_desc(
            "This service tailors suggestions for content or products by tracking a user's online behavior, "
            "such as browsing history and viewing habits."
        ),
        props=mk_props(
            receives_input=True,
            rules_only=False,
            generates_output=True,
            influences=True,
            predicts=True,
            creates_content=False,
            recommends=True,
            takes_decisions_direct=False,
            can_do_on_own=True,
            acts_full_autonomy=False,
            limited_autonomy=True,
            non_autonomous=False,
            never_changes=False,
            adapts_online=True,
        ),
    ),
    Card(
        id="ai_asr",
        name="Voice-to-Text Assistant",
        is_ai_system=True,
        description=wrap_desc(
            "Input: Real-time or recorded speech audio from the user. "
            "How it works: A speech recognition model converts acoustic signals into phonemes and maps them to words using language modeling. "
            "Objective: Transcribe spoken language accurately with punctuation and casing. "
            "Output: Editable text transcripts ready for messaging, captioning, or note taking."
        ),
        summary=wrap_desc(
            "This tool converts spoken audio into written text by interpreting the sounds of human speech. "
            "It facilitates communication by transcribing what is said into a clear text format."
        ),
        props=mk_props(
            receives_input=True,
            rules_only=False,
            generates_output=True,
            influences=True,
            predicts=True,
            creates_content=True,
            recommends=False,
            takes_decisions_direct=False,
            can_do_on_own=True,
            acts_full_autonomy=False,
            limited_autonomy=True,
            non_autonomous=False,
            never_changes=True,
            adapts_online=False,
        ),
    ),
    Card(
        id="ai_img_cls",
        name="Image Classifier",
        is_ai_system=True,
        description=wrap_desc(
            "Input: Digital images uploaded or captured by the user. "
            "How it works: A convolutional neural network detects visual patterns learned from millions of labeled photos. "
            "Objective: Determine the most probable category or labels for the objects in view. "
            "Output: Predicted classes and confidence scores describing the image."
        ),
        summary=wrap_desc(
            "This application analyzes uploaded photos to categorize the objects within them."
            "It classifies pictures accurately, for instance, suggesting if an image features a dog or a cat."
        ),
        props=mk_props(
            receives_input=True,
            rules_only=False,
            generates_output=True,
            influences=True,
            predicts=True,
            creates_content=False,
            recommends=True,
            takes_decisions_direct=False,
            can_do_on_own=True,
            acts_full_autonomy=False,
            limited_autonomy=True,
            non_autonomous=False,
            never_changes=True,
            adapts_online=False,
        ),
    ),
    Card(
        id="ai_screen",
        name="Job Applicant Screening Tool",
        is_ai_system=True,
        description=wrap_desc(
            "Input: Candidate résumés, application responses, and defined job requirements. "
            "How it works: Models compare applicant attributes against historical hiring data and competency criteria. "
            "Objective: Highlight applicants who best match the role while flagging potential risks. "
            "Output: Ranked candidate lists, fit scores, and notes for recruiters."
        ),
        summary=wrap_desc(
            "This system evaluates candidate applications by examining CVs and related details. It utilizes "
            "historical hiring data to suggest which candidates are the best fits for job positions, assisting "
            "in the selection process."
        ),
        props=mk_props(
            receives_input=True,
            rules_only=False,
            generates_output=True,
            influences=True,
            predicts=True,
            creates_content=False,
            recommends=True,
            takes_decisions_direct=False,
            can_do_on_own=True,
            acts_full_autonomy=False,
            limited_autonomy=True,
            non_autonomous=False,
            never_changes=True,
            adapts_online=False,
        ),
    ),
]


CARDS_NON_AI: List[Card] = [
    Card(
        id="na_excel",
        name="Excel Spreadsheet",
        is_ai_system=False,
        description=wrap_desc(
            "Input: Numbers, labels, and formulas typed into worksheet cells. "
            "How it works: Built-in spreadsheet functions perform arithmetic and sorting exactly as configured. "
            "Objective: Organize data and calculate totals or analyses requested by the user. "
            "Output: Updated tables, charts, or cell values reflecting the entered formulas."
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
            "Input: Structured queries specifying filters, fields, and sorting rules. "
            "How it works: The database engine executes the query literally against indexed tables without adapting the logic. "
            "Objective: Retrieve records that match the specified conditions. "
            "Output: A list or table of rows returned by the query."
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
            "Input: Transaction records including region, product, quantity, revenue, and dates. "
            "How it works: Predefined aggregations and charts total the data and refresh visuals when new records load. "
            "Objective: Present sales performance trends for quick review. "
            "Output: Interactive charts, KPIs, and summary tables driven by the formulas."
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
            "Input: Completed survey responses collected from participants. "
            "How it works: Deterministic routines count selections, average ratings, and compute simple statistics. "
            "Objective: Describe overall sentiment or satisfaction levels without interpretation. "
            "Output: Percentages, mean scores, and basic tables ready for reports."
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
        name="Inventory Forecaster",
        is_ai_system=False,
        description=wrap_desc(
            "Input: Historical daily sales quantities for each item. "
            "How it works: Calculates a rolling average of recent activity using fixed business rules without adapting over time. "
            "Objective: Estimate the next day's demand to guide restocking. "
            "Output: A single numeric forecast per item."
        ),
        summary=wrap_desc(
            "This tool uses historical sales data to estimate future sales activity. It calculates average daily sales to produce a numerical prediction for items expected to be sold on the following day."
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
        name="Customer Service Time Estimator",
        is_ai_system=False,
        description=wrap_desc(
            "Input: Past support ticket timestamps such as creation, assignment, and resolution times. "
            "How it works: Applies formula-based averages and optionally adds fixed buffers to represent workload. "
            "Objective: Provide expectations for how long new tickets may remain open. "
            "Output: An estimated resolution time communicated to customers or agents."
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

CARD_ICONS: Dict[str, str] = {
    "ai_chatbot": "💬",
    "ai_spam_filter": "🚫📧",
    "ai_drug_disc": "🧪",
    "ai_reco": "🎯",
    "ai_asr": "🎙️",
    "ai_img_cls": "🖼️",
    "ai_screen": "📋",
    "na_excel": "📊",
    "na_db_search": "🗄️",
    "na_sales_dash": "📈",
    "na_survey": "📝",
    "na_inventory_forecaster": "📦",
    "na_ticket_eta": "⏱️",
}

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
        border-radius: 0.95rem;
        box-shadow: 0 10px 25px rgba(15, 23, 42, 0.08);
        padding: 1.15rem 1.2rem;
        text-align: left;
        white-space: normal;
        line-height: 1.45;
        font-weight: 600;
        color: #0e1117;
        display: flex;
        gap: 0.75rem;
        align-items: flex-start;
        font-size: 0.98rem;
    }
    button[data-testid="baseButton-secondary"]:hover {
        border-color: rgba(59, 130, 246, 0.55);
        box-shadow: 0 12px 30px rgba(59, 130, 246, 0.18);
    }
    button[data-testid="baseButton-secondary"]:focus {
        outline: 2px solid rgba(59, 130, 246, 0.45);
        outline-offset: 2px;
    }
    .question-card {
        background: var(--q-bg, #ffffff);
        border: 1px solid var(--q-border, rgba(148, 163, 184, 0.45));
        border-left: 5px solid var(--q-border, rgba(148, 163, 184, 0.45));
        border-radius: 1.05rem;
        padding: 1rem 1.2rem;
        display: flex;
        gap: 0.9rem;
        align-items: flex-start;
        box-shadow: 0 12px 32px rgba(15, 23, 42, 0.06);
        margin-bottom: 1rem;
    }
    .question-card__icon {
        font-size: 1.6rem;
        line-height: 1;
    }
    .question-card__title {
        font-size: 1rem;
        font-weight: 600;
        color: #0f172a;
        line-height: 1.45;
    }
    .question-card__answer {
        margin-top: 0.45rem;
        font-size: 0.92rem;
        font-weight: 500;
        color: #1e293b;
    }
    .question-card__answer span {
        background: rgba(255, 255, 255, 0.55);
        padding: 0.1rem 0.4rem;
        border-radius: 999px;
        font-weight: 700;
        font-size: 0.85rem;
        margin-left: 0.35rem;
    }
    .indicator-card {
        border-radius: 1rem;
        padding: 1rem 1.15rem;
        margin-bottom: 0.9rem;
        border: 1px solid #e2e8f0;
        background: #f8fafc;
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.6);
    }
    .indicator-card__header {
        display: flex;
        gap: 0.75rem;
        align-items: flex-start;
    }
    .indicator-card__icon {
        font-size: 1.35rem;
        line-height: 1;
    }
    .indicator-card__title {
        font-weight: 600;
        color: #0f172a;
        line-height: 1.4;
        font-size: 0.98rem;
    }
    .indicator-card__answer {
        font-size: 0.92rem;
        color: #334155;
        margin-top: 0.3rem;
    }
    .indicator-card__badge {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        border-radius: 999px;
        padding: 0.2rem 0.65rem;
        font-size: 0.82rem;
        font-weight: 600;
        margin-top: 0.6rem;
    }
    .indicator-card__description {
        margin-top: 0.45rem;
        font-size: 0.88rem;
        color: #475569;
        line-height: 1.5;
    }
    .system-overview-card {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.12), rgba(59, 130, 246, 0));
        border: 1px solid rgba(59, 130, 246, 0.18);
        border-radius: 1.1rem;
        padding: 1.3rem 1.4rem;
        display: flex;
        gap: 1.05rem;
        align-items: flex-start;
        box-shadow: 0 18px 38px rgba(15, 23, 42, 0.08);
        margin-bottom: 1.5rem;
    }
    .system-overview-card__icon {
        font-size: 2.35rem;
        line-height: 1;
        filter: drop-shadow(0 6px 12px rgba(37, 99, 235, 0.25));
    }
    .system-overview-card__body {
        flex: 1;
    }
    .system-overview-card__eyebrow {
        font-size: 0.82rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #475569;
        font-weight: 600;
    }
    .system-overview-card__title {
        font-size: 1.55rem;
        font-weight: 700;
        color: #0f172a;
        margin-top: 0.3rem;
    }
    .system-overview-card__summary {
        margin-top: 0.75rem;
        font-size: 1rem;
        line-height: 1.65;
        color: #1e293b;
    }
    .reveal-card {
        background: #ffffff;
        border: 1px solid rgba(148, 163, 184, 0.35);
        border-radius: 1.2rem;
        padding: 1.4rem;
        box-shadow: 0 16px 38px rgba(15, 23, 42, 0.09);
        margin-top: 1rem;
    }
    .reveal-card__header {
        display: flex;
        gap: 1.1rem;
        align-items: center;
        margin-bottom: 1.2rem;
    }
    .reveal-card__icon {
        font-size: 2.1rem;
        line-height: 1;
    }
    .reveal-card__title {
        font-size: 1.45rem;
        font-weight: 700;
        color: #0f172a;
    }
    .reveal-card__type {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        font-size: 0.85rem;
        font-weight: 600;
        border-radius: 999px;
        padding: 0.25rem 0.75rem;
        margin-top: 0.35rem;
        border: 1px solid transparent;
    }
    .reveal-card__type--ai {
        background: rgba(59, 130, 246, 0.12);
        color: #1d4ed8;
        border-color: rgba(59, 130, 246, 0.45);
    }
    .reveal-card__type--not-ai {
        background: rgba(16, 185, 129, 0.12);
        color: #047857;
        border-color: rgba(16, 185, 129, 0.45);
    }
    .reveal-card__grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 1.1rem;
    }
    .reveal-card__section {
        background: linear-gradient(135deg, rgba(248, 250, 252, 0.9), rgba(226, 232, 240, 0.6));
        border-radius: 0.95rem;
        padding: 1rem;
        border: 1px solid rgba(148, 163, 184, 0.25);
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.6);
    }
    .reveal-card__section-label {
        font-size: 0.88rem;
        font-weight: 700;
        letter-spacing: 0.02em;
        color: #0f172a;
        text-transform: uppercase;
    }
    .reveal-card__section-body {
        margin-top: 0.45rem;
        font-size: 0.95rem;
        line-height: 1.55;
        color: #1e293b;
    }
    .reveal-card__empty {
        color: #94a3b8;
        font-style: italic;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.subheader("🕵️ Can you guess if this is an AI system?")
st.write(
    "Ask up to **five** yes/no questions, then decide whether the card below describes an AI system."
)

icon = CARD_ICONS.get(current_card.id, "🧠")
summary_html = html.escape(current_card.summary).replace("\n", "<br>")
st.markdown(
    f"""
    <div class="system-overview-card">
        <div class="system-overview-card__icon">{icon}</div>
        <div class="system-overview-card__body">
            <div class="system-overview-card__eyebrow">System overview</div>
            <div class="system-overview-card__title">{html.escape(current_card.name)}</div>
            <div class="system-overview-card__summary">{summary_html}</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

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
if remaining or asked_records:
    st.subheader("🗣️ Ask a question")
    if remaining and can_ask_more:
        st.caption("Select a card to ask the computer about the hidden system.")
    elif remaining:
        st.caption("Question limit reached. Review your asked cards below.")
    else:
        st.caption("You've asked every available question.")

    num_cols = 2 if len(QUESTIONS) > 1 else 1
    cols = st.columns(num_cols, gap="large")
    asked_lookup = {qid: ans for (qid, ans, _lbl) in asked_records}

    for idx, question in enumerate(QUESTIONS):
        target_col = cols[idx % num_cols]
        with target_col:
            ans = asked_lookup.get(question.id)
            if ans:
                style = QUESTION_STATUS_STYLES.get(
                    ans, {"bg": "#f1f5f9", "border": "#e2e8f0", "text": "#0f172a"}
                )
                label = "Yes" if ans == "yes" else "No"
                question_text_html = html.escape(question.text).replace("\n", "<br>")
                icon = QUESTION_ICONS.get(question.id, "❓")
                st.markdown(
                    f"""
                    <div class="question-card" style="--q-bg:{style['bg']};--q-border:{style['border']};">
                        <div class="question-card__icon">{icon}</div>
                        <div class="question-card__body">
                            <div class="question-card__title">{question_text_html}</div>
                            <div class="question-card__answer" style="color:{style['text']};">
                                Computer answered<span>{label}</span>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                icon = QUESTION_ICONS.get(question.id, "❓")
                pressed = st.button(
                    f"{icon}  {question.text}",
                    key=f"q_btn_{question.id}",
                    type="secondary",
                    use_container_width=True,
                    disabled=not can_ask_more,
                )
                if pressed and can_ask_more:
                    ans = get_true_answer(current_card, question)
                    indicator_label = question.indicators.get(ans, "neutral")
                    game["asked"].append((question.id, ans, indicator_label))
                    st.rerun()

elif not remaining and not game["completed"] and game["user_final_guess"] is None:
    st.info("You've asked every question. It's time to make your final guess!")

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
    reveal_icon = CARD_ICONS.get(current_card.id, "🧠")
    type_class = "ai" if truth_is_ai else "not-ai"
    type_label = "🤖 AI system" if truth_is_ai else "🧰 Non-AI tool"
    section_order = [
        ("Input", "🔌"),
        ("Objective", "🎯"),
        ("How it works", "⚙️"),
        ("Output", "📤"),
    ]
    section_html_parts: List[str] = []
    for section_name, emoji in section_order:
        copy = secs.get(section_name, "")
        if copy:
            formatted = html.escape(copy).replace("\n", "<br>")
        else:
            formatted = "<span class='reveal-card__empty'>No details provided.</span>"
        section_html_parts.append(
            textwrap.dedent(
                f"""
                <div class=\"reveal-card__section\">
                    <div class=\"reveal-card__section-label\">{emoji} {section_name}</div>
                    <div class=\"reveal-card__section-body\">{formatted}</div>
                </div>
                """
            ).strip()
        )
    sections_html = "\n".join(section_html_parts)
    reveal_html = textwrap.dedent(
        f"""
        <div class=\"reveal-card\">
            <div class=\"reveal-card__header\">
                <div class=\"reveal-card__icon\">{reveal_icon}</div>
                <div>
                    <div class=\"reveal-card__title\">{html.escape(current_card.name)}</div>
                    <div class=\"reveal-card__type reveal-card__type--{type_class}\">{type_label}</div>
                </div>
            </div>
            <div class=\"reveal-card__grid\">
                {sections_html}
            </div>
        </div>
        """
    ).strip()
    st.markdown(
        reveal_html,
        unsafe_allow_html=True,
    )

    with st.expander("Find out what each question reveals about the system"):
        st.markdown(
            "Review how each answer guided your deduction, and see which clues were decisive.",
        )
        for idx, (q_id, ans, lbl) in enumerate(game["asked"], start=1):
            q_obj = next(q for q in QUESTIONS if q.id == q_id)
            indicator = INDICATOR_DETAILS.get(lbl, INDICATOR_DETAILS["neutral"])
            question_text_html = html.escape(q_obj.text).replace("\n", "<br>")
            answer_label = "Yes" if ans == "yes" else "No"
            question_icon = QUESTION_ICONS.get(q_id, "❓")
            badge_icon = indicator.get("icon", "ℹ️")
            description_html = html.escape(indicator["description"])
            st.markdown(
                f"""
                <div class="indicator-card" style="background:{indicator['bg']};border-color:{indicator['border']};">
                    <div class="indicator-card__header">
                        <div class="indicator-card__icon">{question_icon}</div>
                        <div>
                            <div class="indicator-card__title">Q{idx}. {question_text_html}</div>
                            <div class="indicator-card__answer">Answer: <strong>{answer_label}</strong></div>
                        </div>
                    </div>
                    <div class="indicator-card__badge" style="color:{indicator['text']};border:1px solid {indicator['border']};background: rgba(255, 255, 255, 0.7);">
                        <span>{badge_icon}</span>
                        <span>{indicator['title']}</span>
                    </div>
                    <div class="indicator-card__description" style="color:{indicator['text']};">
                        {description_html}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.divider()
    st.subheader("🔁 Play again")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("New random game", type="primary"):
            reset_game()
            st.rerun()
    with col2:
        sel = st.selectbox(
            "Or pick specific card",
            options=ALL_CARDS,
            format_func=lambda c: c.name,
            key="again_pick_card",
        )
        if st.button("Start with chosen card", type="primary"):
            reset_game(sel.id)
            st.rerun()

st.markdown("---")
st.caption(
    "Ask smart questions, track the answers, and decide whether the hidden card describes an AI system."
)

