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
        "description": "This answer makes it very clear the card describes an AI system.",
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
        "description": "Both AI and non-AI systems can share this trait.",
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
        "Does it only follow fixed rules written by humans?",
        {"yes": "def_not_ai", "no": "ai_ind"},
        "rules_only",
    ),
    Question(
        "inf_output",
        "Does it generate output data that provides a solution to a problem?",
        {"yes": "neutral", "no": "not_ai_ind"},
        "generates_output",
    ),
    Question(
        "inf_influence",
        "Can the system’s results influence what people or other systems do next?",
        {"yes": "neutral", "no": "not_ai_ind"},
        "influences",
    ),

    # 2. Outputs
    Question(
        "out_predicts",
        "Does it guess or forecast something about the future or about data it hasn’t seen before?",
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
        "Does it provide recommendations or suggest choices, actions, or products?",
        {"yes": "ai_ind", "no": "neutral"},
        "recommends",
    ),
    Question(
        "out_takes_decisions",
        "Does it make decisions on its own, without needing human approval first?",
        {"yes": "ai_ind", "no": "neutral"},
        "takes_decisions_direct",
    ),

    # 3. Autonomy
    Question(
        "auton_can_do_on_own",
        "Once it receives an input, can it figure what to do without step-by-step human instructions?",
        {"yes": "ai_ind", "no": "def_not_ai"},
        "can_do_on_own",
    ),
    Question(
        "auton_full_autonomy",
        "Does it act entirely on its own, making real-world decisions without any human checking?",
        {"yes": "def_ai", "no": "neutral"},
        "acts_full_autonomy",
    ),
    Question(
        "auton_limited",
        "Does it figure out on its own how to solve a problem, but leave the final decision to a human?",
        {"yes": "ai_ind", "no": "neutral"},
        "limited_autonomy",
    ),
    Question(
        "auton_non_autonomous",
        "Does it only work if a person gave it exact step-by-step instructions?",
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
        "Can it adjust or improve by learning from new data as it operates?",
        {"yes": "def_ai", "no": "neutral"},
        "adapts_online",
    ),
]

# =========================
# Cards
# =========================


def wrap_desc(s: str) -> str:
    """Normalize description/summary text without forcing manual line breaks."""

    normalized = textwrap.dedent(s).strip()
    # Collapse any internal whitespace so HTML can flow the text naturally based on layout.
    return re.sub(r"\s+", " ", normalized)


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
            "Input: User messages and any attached text (documents, snippets) to summarize. "
            "How it works: It uses an AI model trained on large amounts of text to analyse the user message, understand the key points, and produce a clear summary that answers the user's request. "
            "Output: Generated content — a concise written summary. "
            "Level of autonomy: Limited autonomy — generates content, but a human decides what to do with it. "
            "Adaptiveness: Does not change by itself during use; only improves when humans update the underlying AI model."
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
            "Input: Email details such as sender, subject, body text, and links. "
            "How it works: It uses an AI model trained on examples of spam and legitimate emails to judge how likely a message is unwanted or malicious and route it accordingly. "
            "Output: A decision — the email is delivered to inbox or moved to spam. "
            "Level of autonomy: Can take decisions on its own for this task (automatically moves emails). "
            "Adaptiveness: Does not change by itself during use; only changes when humans update the model or settings."
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
            "Input: Molecular structures, chemical descriptors, and laboratory screening results. "
            "How it works: It uses an AI model trained on past experiment data to find patterns that indicate promising molecules and prioritize which compounds to test next. "
            "Output: A recommendation — a ranked list of candidate molecules. "
            "Level of autonomy: Limited autonomy — provides recommendations; humans decide which compounds to pursue. "
            "Adaptiveness: Does not change by itself during use; only improves when humans retrain or update the model."
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
            "Input: A person's browsing activity, searches, ratings, and recent purchases. "
            "How it works: It uses an AI model trained on interaction data to learn preferences and suggest items a person is likely to engage with next. "
            "Output: A recommendation — tailored product or content suggestions. "
            "Level of autonomy: Limited autonomy — issues recommendations that may be shown automatically; humans (or users) decide whether to act. "
            "Adaptiveness: Adapts at every interaction — updates suggestions as new behavior is observed."
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
            "Input: Live or recorded speech audio. "
            "How it works: It uses an AI model trained on speech and language data to convert sounds into words with punctuation and casing. "
            "Output: Generated content — a text transcript. "
            "Level of autonomy: Limited autonomy — produces transcripts; humans decide how to use them. "
            "Adaptiveness: Does not change by itself during use; only improves when humans update the model."
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
            "How it works: It uses an AI model trained on labeled photos to detect visual patterns and assign the most likely categories. "
            "Output: A prediction — the most likely labels for the image with confidence scores. "
            "Level of autonomy: Limited autonomy — produces predictions; humans decide actions based on them. "
            "Adaptiveness: Does not change by itself during use; only improves when humans update the model."
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
            "Input: Candidate CVs, application answers, and job requirements. "
            "How it works: It uses an AI model trained on historical hiring data and competency criteria to score candidates and highlight likely fits. "
            "Output: A recommendation — ranked candidate list with fit scores. "
            "Level of autonomy: Limited autonomy — provides recommendations; humans make hiring decisions. "
            "Adaptiveness: Does not change by itself during use; only changes when humans retrain or adjust the model/criteria."
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
            "Input: Numbers, text labels, and formulas entered in cells. "
            "How it works: It strictly follows rules set by humans to perform calculations, sorting, and formatting exactly as configured. "
            "Output: Updated cells, tables, and charts. "
            "Level of autonomy: No autonomy — it can only follow human-set rules and never takes decisions on its own. "
            "Adaptiveness: Always unchanged — behavior only changes if humans edit formulas or settings."
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
            "Input: Structured queries with filters, fields, and sort order. "
            "How it works: It strictly follows rules set by humans to run the query literally against stored tables and indexes. "
            "Output: List or table of matching records. "
            "Level of autonomy: No autonomy — it can only follow human-set rules and never takes decisions on its own. "
            "Adaptiveness: Always unchanged — logic only changes if humans change the query or schema."
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
            "Input: Transaction data (region, product, quantity, revenue, dates). "
            "How it works: It strictly follows rules set by humans to aggregate data and refresh charts using predefined formulas. "
            "Output: Charts, KPIs, and summary tables. "
            "Level of autonomy: No autonomy — it can only follow human-set rules and never takes decisions on its own. "
            "Adaptiveness: Always unchanged — visuals/formulas only change if humans update them or the underlying data."
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
            "Input: Completed survey responses. "
            "How it works: It strictly follows rules set by humans to count choices, average ratings, and compute simple statistics. "
            "Output: Percentages, mean scores, and basic tables. "
            "Level of autonomy: No autonomy — it can only follow human-set rules and never takes decisions on its own. "
            "Adaptiveness: Always unchanged — calculations only change if humans modify the rules or inputs."
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
            "Input: Historical daily sales for each item. "
            "How it works: It strictly follows rules set by humans to compute a rolling average of recent sales and estimate the next day's demand. "
            "Output: Next-day quantity estimate per item. "
            "Level of autonomy: No autonomy — it can only follow human-set rules and never takes decisions on its own. "
            "Adaptiveness: Always unchanged — predictions only change if humans adjust parameters or formulas."
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
            "Input: Past ticket timestamps (creation, assignment, resolution). "
            "How it works: It strictly follows rules set by humans to average historical resolution times and apply fixed buffers to reflect workload. "
            "Output: Expected resolution time for new tickets. "
            "Level of autonomy: No autonomy — it can only follow human-set rules and never takes decisions on its own. "
            "Adaptiveness: Always unchanged — estimates only change if humans update rules or the data trends shift."
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
    """Parse structured description fields for the reveal view."""
    section_names = [
        "Input",
        "How it works",
        "Output",
        "Level of autonomy",
        "Adaptiveness",
    ]
    sections = {name: "" for name in section_names}
    pattern = r"(" + "|".join(map(re.escape, section_names)) + r"):"
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


def render_subtle_callout(icon: str, title: str, body: str) -> None:
    """Render a compact, low-contrast information callout."""

    st.markdown(
        f"""
        <div class="subtle-callout">
            <span class="subtle-callout__icon">{icon}</span>
            <div class="subtle-callout__content">
                <div class="subtle-callout__title">{title}</div>
                <div class="subtle-callout__body">{body}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================
# Streamlit UI
# =========================


st.set_page_config(page_title="AI Guess Who?", page_icon="🕵️", layout="centered")
st.title("🕵️  AI Guess Who?")

if "show_privacy_info" not in st.session_state:
    st.session_state.show_privacy_info = False
if "show_ai_info" not in st.session_state:
    st.session_state.show_ai_info = False


toggle_privacy_col, toggle_ai_col, _info_toggle_spacer = st.columns([1, 1, 6])
with toggle_privacy_col:
    st.markdown('<div class="info-toggle">', unsafe_allow_html=True)
    if st.button(
        "Privacy",
        key="privacy_info_button",
        help="Show or hide privacy details.",
    ):
        st.session_state.show_privacy_info = not st.session_state.show_privacy_info
    st.markdown("</div>", unsafe_allow_html=True)
with toggle_ai_col:
    st.markdown('<div class="info-toggle">', unsafe_allow_html=True)
    if st.button(
        "AI info",
        key="ai_info_button",
        help="Show or hide AI-generated content notes.",
    ):
        st.session_state.show_ai_info = not st.session_state.show_ai_info
    st.markdown("</div>", unsafe_allow_html=True)

if st.session_state.show_privacy_info:
    render_subtle_callout(
        "🔐",
        "Privacy & anonymity",
        """
        <p>No personal data is collected—each game session is fully anonymous and none of your answers are saved or stored.</p>
        <ul>
            <li>Gameplay progress lives only inside your local session (the drawn card, the preset questions you picked, whether you finished the round, and your final guess).</li>
            <li>Interactions rely solely on built-in widgets (radio buttons, select boxes, and buttons), so you never submit custom text or files to the app.</li>
            <li>Telemetry is disabled, preventing usage statistics from being collected.</li>
        </ul>
        """,
    )

if st.session_state.show_ai_info:
    render_subtle_callout(
        "🤖",
        "AI-generated content",
        """
        <p>Some of the code and content of this app has been AI generated. Humans have reviewed all AI generated content. Remember to label AI-generated content when sharing it.</p>
        """,
    )

# --- Sidebar: new game ---
with st.sidebar:
    st.header("New game")
    st.caption("Draw a new card to challenge yourself again.")
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
    .info-toggle {
        display: flex;
        justify-content: flex-start;
        align-items: center;
        padding-top: 0.15rem;
        width: fit-content;
    }
    .info-toggle button {
        border-radius: 0.45rem;
        padding: 0.3rem 0.65rem;
        font-size: 0.75rem;
        font-weight: 600;
        background: rgba(148, 163, 184, 0.14);
        border: 1px solid rgba(148, 163, 184, 0.6);
        color: #0f172a;
        white-space: nowrap;
        transition: background 0.15s ease, border-color 0.15s ease,
            color 0.15s ease, box-shadow 0.15s ease;
        box-shadow: 0 2px 6px rgba(15, 23, 42, 0.08);
    }
    .info-toggle button:hover {
        background: rgba(59, 130, 246, 0.2);
        border-color: rgba(59, 130, 246, 0.5);
        color: #1d4ed8;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.18);
    }
    .info-toggle button:focus {
        outline: 2px solid rgba(59, 130, 246, 0.5);
        outline-offset: 1px;
    }
    .subtle-callout {
        background: rgba(241, 245, 249, 0.65);
        border: 1px solid rgba(148, 163, 184, 0.45);
        border-radius: 0.85rem;
        padding: 0.75rem 0.95rem;
        display: flex;
        gap: 0.75rem;
        align-items: flex-start;
        color: #0f172a;
        font-size: 0.92rem;
        line-height: 1.5;
        margin-bottom: 0.75rem;
    }
    .subtle-callout__icon {
        font-size: 1.35rem;
        line-height: 1;
        position: relative;
        top: 0.1rem;
    }
    .subtle-callout__content {
        flex: 1;
    }
    .subtle-callout__title {
        font-weight: 600;
        margin-bottom: 0.15rem;
        font-size: 0.95rem;
    }
    .subtle-callout__body {
        color: #334155;
    }
    .subtle-callout__body ul {
        margin: 0.4rem 0 0.1rem 1.1rem;
        padding: 0;
    }
    .subtle-callout__body li {
        margin-bottom: 0.25rem;
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
    div[data-testid="stExpander"] > details {
        border-radius: 1rem;
        border: 1px solid rgba(59, 130, 246, 0.18);
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.08), rgba(59, 130, 246, 0));
        box-shadow: 0 16px 32px rgba(15, 23, 42, 0.08);
        padding: 0.75rem 1rem;
        margin-bottom: 1.5rem;
    }
    div[data-testid="stExpander"] > details[open] {
        border-color: rgba(37, 99, 235, 0.35);
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.12), rgba(59, 130, 246, 0.02));
    }
    div[data-testid="stExpander"] > details summary {
        font-weight: 600;
        font-size: 1rem;
        color: #1d4ed8;
    }
    .ai-key-info__note {
        margin-top: 0.8rem;
        background: rgba(59, 130, 246, 0.08);
        border: 1px solid rgba(37, 99, 235, 0.25);
        border-radius: 0.85rem;
        padding: 0.85rem 1rem;
        font-size: 0.9rem;
        line-height: 1.55;
        color: #1d4ed8;
        font-weight: 500;
    }
    .ai-key-info {
        display: grid;
        gap: 1rem;
        margin-top: 0.8rem;
    }
    @media (min-width: 768px) {
        .ai-key-info {
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }
    }
    .ai-key-info__box {
        background: #ffffff;
        border-radius: 1rem;
        border: 1px solid rgba(148, 163, 184, 0.35);
        padding: 1.1rem 1.2rem;
        display: flex;
        gap: 0.85rem;
        align-items: flex-start;
        box-shadow: 0 12px 28px rgba(15, 23, 42, 0.08);
    }
    .ai-key-info__content {
        flex: 1;
    }
    .ai-key-info__icon {
        font-size: 1.8rem;
        line-height: 1;
        filter: drop-shadow(0 6px 12px rgba(59, 130, 246, 0.25));
    }
    .ai-key-info__title {
        font-size: 1rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 0.4rem;
    }
    .ai-key-info__text {
        margin: 0;
        font-size: 0.93rem;
        line-height: 1.55;
        color: #334155;
    }
    .reveal-result {
        display: flex;
        gap: 0.95rem;
        align-items: flex-start;
        background: #ffffff;
        border-radius: 1rem;
        border: 1px solid rgba(148, 163, 184, 0.35);
        padding: 1.15rem 1.25rem;
        margin: 1rem 0 1.2rem;
        box-shadow: 0 18px 40px rgba(15, 23, 42, 0.12);
    }
    .reveal-result__icon {
        font-size: 2.5rem;
        line-height: 1;
        filter: drop-shadow(0 8px 18px rgba(15, 23, 42, 0.18));
    }
    .reveal-result__title {
        font-size: 1.18rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
        color: #0f172a;
    }
    .reveal-result__message {
        font-size: 0.96rem;
        line-height: 1.6;
        color: #1e293b;
    }
    .reveal-result__cta {
        margin-top: 0.6rem;
        font-size: 0.94rem;
        font-weight: 600;
        display: inline-flex;
        gap: 0.3rem;
        align-items: center;
    }
    .reveal-result__cta::after {
        content: ":)";
        font-size: 1rem;
    }
    .reveal-result--correct {
        background: linear-gradient(135deg, rgba(220, 252, 231, 0.85), rgba(187, 247, 208, 0.6));
        border-color: rgba(34, 197, 94, 0.4);
    }
    .reveal-result--correct .reveal-result__title,
    .reveal-result--correct .reveal-result__message,
    .reveal-result--correct .reveal-result__cta {
        color: #166534;
    }
    .reveal-result--incorrect {
        background: linear-gradient(135deg, rgba(254, 226, 226, 0.85), rgba(254, 202, 202, 0.6));
        border-color: rgba(248, 113, 113, 0.45);
    }
    .reveal-result--incorrect .reveal-result__title,
    .reveal-result--incorrect .reveal-result__message,
    .reveal-result--incorrect .reveal-result__cta {
        color: #b91c1c;
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

st.subheader("Can you guess if this is an AI system?")
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

with st.expander("Remind me the key characteristics of an AI system"):
    st.markdown(
        """
        <div class="ai-key-info__note">Note: this game adhere to the definition of AI system set by Article 3 of the EU AI Act</div>
        <div class="ai-key-info">
            <div class="ai-key-info__box">
                <div class="ai-key-info__icon">🧠</div>
                <div class="ai-key-info__content">
                    <div class="ai-key-info__title">AI systems perform inference</div>
                    <p class="ai-key-info__text">
                        They receive input data and use AI models to generate outputs that achieve one or more objectives.
                    </p>
                </div>
            </div>
            <div class="ai-key-info__box">
                <div class="ai-key-info__icon">🧭</div>
                <div class="ai-key-info__content">
                    <div class="ai-key-info__title">AI systems have various levels of autonomy</div>
                    <p class="ai-key-info__text">
                        Some are fully autonomous, but most require humans to take decision and can only recommend, predict or generate content.
                    </p>
                </div>
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
        st.caption("Select a card to ask the computer about the system.")
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

    actual_type_text = "an AI system" if truth_is_ai else "a non-AI tool"
    guessed_type_text = "an AI system" if your_guess_ai else "a non-AI tool"

    if game["user_guess_correct"]:
        result_html = textwrap.dedent(
            f"""
            <div class=\"reveal-result reveal-result--correct\">
                <div class=\"reveal-result__icon\">🎉</div>
                <div>
                    <div class=\"reveal-result__title\">Correct!</div>
                    <div class=\"reveal-result__message\">Great deduction — you correctly identified the card as {actual_type_text}.</div>
                </div>
            </div>
            """
        ).strip()
    else:
        result_html = textwrap.dedent(
            f"""
            <div class=\"reveal-result reveal-result--incorrect\">
                <div class=\"reveal-result__icon\">😞</div>
                <div>
                    <div class=\"reveal-result__title\">Not quite...</div>
                    <div class=\"reveal-result__message\">You guessed {guessed_type_text}, but the card is {actual_type_text}.</div>
                    <div class=\"reveal-result__cta\">Give it another shot — grab a new card and try again!</div>
                </div>
            </div>
            """
        ).strip()

    st.markdown(result_html, unsafe_allow_html=True)

    secs = parse_card_sections(current_card.description)
    reveal_icon = CARD_ICONS.get(current_card.id, "🧠")
    type_class = "ai" if truth_is_ai else "not-ai"
    type_label = "🤖 AI system" if truth_is_ai else "🧰 Non-AI tool"
    section_order = [
        ("Input", "📥"),
        ("How it works", "⚙️"),
        ("Output", "📤"),
        ("Level of autonomy", "🕹️"),
        ("Adaptiveness", "🔄"),
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
    st.markdown(
        """
        <style>
            form#play_again_random,
            form#play_again_specific {
                background: linear-gradient(135deg, #eef2ff 0%, #f8fafc 100%);
                border: 1px solid #dbeafe;
                border-radius: 1rem;
                padding: 1.25rem;
                box-shadow: 0 18px 40px rgba(15, 23, 42, 0.08);
                display: flex;
                flex-direction: column;
                gap: 0.9rem;
                height: 100%;
            }

            form#play_again_specific {
                background: linear-gradient(135deg, #fef3c7 0%, #fffbeb 100%);
                border-color: #fcd34d;
            }

            form#play_again_random .play-again-card__title,
            form#play_again_specific .play-again-card__title {
                font-size: 1.05rem;
                font-weight: 700;
                margin: 0;
                color: #0f172a;
            }

            form#play_again_random .play-again-card__subtitle,
            form#play_again_specific .play-again-card__subtitle {
                font-size: 0.95rem;
                margin: 0;
                color: #475569;
            }

            .play-again-card__label {
                font-size: 0.9rem;
                font-weight: 600;
                color: #1d4ed8;
                margin-bottom: -0.2rem;
            }

            form#play_again_random [data-testid="stFormSubmitButton"] button,
            form#play_again_specific [data-testid="stFormSubmitButton"] button {
                width: 100%;
                border-radius: 999px;
                font-weight: 600;
                font-size: 0.95rem;
                padding: 0.6rem 1rem;
            }

            form#play_again_random [data-testid="stFormSubmitButton"],
            form#play_again_specific [data-testid="stFormSubmitButton"] {
                margin-top: auto;
            }

            form#play_again_specific div[data-baseweb="select"] {
                background: #ffffff;
                border-radius: 0.75rem;
                border: 1px solid #cbd5f5;
                box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.4);
            }

            form#play_again_specific div[data-baseweb="select"]:focus-within {
                box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.25);
                border-color: #2563eb;
            }

            @media (max-width: 640px) {
                form#play_again_random,
                form#play_again_specific {
                    padding: 1rem;
                    gap: 0.75rem;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2, gap="large")
    with col1:
        with st.form("play_again_random"):
            st.markdown(
                "<div class='play-again-card__title'>🎲 New random game</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "<p class='play-again-card__subtitle'>Shuffle the deck for a fresh mystery and restart your deduction journey.</p>",
                unsafe_allow_html=True,
            )
            random_submit = st.form_submit_button(
                "Start a random challenge",
                type="primary",
                use_container_width=True,
            )
            if random_submit:
                reset_game()
                st.rerun()

    with col2:
        with st.form("play_again_specific"):
            st.markdown(
                "<div class='play-again-card__title'>🎯 Pick a specific card</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "<p class='play-again-card__subtitle'>Jump straight to a system you want to revisit or discuss.</p>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "<div class='play-again-card__label'>Choose a card</div>",
                unsafe_allow_html=True,
            )
            sel = st.selectbox(
                "Or pick specific card",
                options=ALL_CARDS,
                format_func=lambda c: c.name,
                key="again_pick_card",
                label_visibility="collapsed",
            )
            specific_submit = st.form_submit_button(
                "Start with chosen card",
                type="primary",
                use_container_width=True,
            )
            if specific_submit:
                reset_game(sel.id)
                st.rerun()

st.markdown("---")
st.caption(
    "Ask smart questions, track the answers, and decide whether the card describes an AI system."
)
