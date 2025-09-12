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
    props: Dict[str, bool] = field(default_factory=dict)


INDICATOR_WEIGHTS = {
    "def_ai": 3,
    "def_not_ai": -3,
    "ai_ind": 1,
    "not_ai_ind": -1,
    "neutral": 0,
}

MAX_QUESTIONS = 5
MAX_NEUTRALS = 1  # applies to Mode A (Computer guesses)

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
        props=mk_props(
            predicts=True,
            creates_content=True,
            recommends=False,
            takes_decisions_direct=False,
            can_do_on_own=True,
            limited_autonomy=True,
            non_autonomous=False,
            never_changes=False,
            adapts_online=False,
            rules_only=False,
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
        props=mk_props(
            predicts=True,
            creates_content=False,
            recommends=False,
            takes_decisions_direct=True,
            can_do_on_own=True,
            limited_autonomy=True,
            non_autonomous=False,
            never_changes=False,
            adapts_online=False,
            rules_only=False,
            influences=True,
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
        props=mk_props(
            predicts=True,
            takes_decisions_direct=False,
            creates_content=False,
            recommends=True,
            can_do_on_own=True,
            limited_autonomy=True,
            adapts_online=False,
            rules_only=False,
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
        props=mk_props(
            predicts=True,
            recommends=True,
            takes_decisions_direct=False,
            creates_content=False,
            can_do_on_own=True,
            limited_autonomy=True,
            adapts_online=True,
            rules_only=False,
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
        props=mk_props(
            predicts=True,
            creates_content=False,
            recommends=False,
            takes_decisions_direct=False,
            can_do_on_own=True,
            limited_autonomy=True,
            adapts_online=False,
            rules_only=False,
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
        props=mk_props(
            predicts=True,
            creates_content=False,
            recommends=False,
            takes_decisions_direct=False,
            can_do_on_own=True,
            limited_autonomy=True,
            adapts_online=False,
            rules_only=False,
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
        props=mk_props(
            predicts=True,
            recommends=True,
            takes_decisions_direct=False,
            creates_content=False,
            can_do_on_own=True,
            limited_autonomy=True,
            adapts_online=False,
            rules_only=False,
            influences=True,
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


def indicator_weight(lbl: str) -> int:
    return INDICATOR_WEIGHTS.get(lbl, 0)


def compute_final_guess_from_indicators(
    asked_triplets: List[Tuple[str, str, str]]
) -> str:
    """Used in Mode A (Computer guesses)."""
    has_def_ai = any(lbl == "def_ai" for _, _, lbl in asked_triplets)
    has_def_not_ai = any(lbl == "def_not_ai" for _, _, lbl in asked_triplets)
    if has_def_ai and not has_def_not_ai:
        return "ai"
    if has_def_not_ai and not has_def_ai:
        return "not_ai"
    if has_def_ai and has_def_not_ai:
        return "cannot_guess"
    score = sum(indicator_weight(lbl) for _, _, lbl in asked_triplets)
    if score > 0:
        return "ai"
    if score < 0:
        return "not_ai"
    return "cannot_guess"


def next_question_candidate(asked_ids: set) -> Optional[Question]:
    """Heuristic for Mode A (Computer chooses)."""

    def priority(q: Question) -> int:
        vals = set(q.indicators.values())
        if "def_ai" in vals or "def_not_ai" in vals:
            return 3
        if "ai_ind" in vals or "not_ai_ind" in vals:
            return 2
        return 1

    candidates = [q for q in QUESTIONS if q.id not in asked_ids]
    if not candidates:
        return None
    candidates.sort(key=priority, reverse=True)
    return candidates[0]


def reset_game(mode: str, pick_card_id: Optional[str] = None):
    """mode: 'computer_guesses' | 'you_guess'"""
    card = CARDS_BY_ID[pick_card_id] if pick_card_id else random.choice(ALL_CARDS)
    st.session_state.game = {
        "mode": mode,
        "card_id": card.id,
        # Common
        "asked": [],  # Mode A: (q_id, player_ans, indicator); Mode B: (q_id, comp_ans, indicator)
        "completed": False,
        # Mode A only
        "neutrals_used": 0,
        "final_guess": None,  # 'ai' | 'not_ai' | 'cannot_guess' (Mode A)
        "player_points": None,
        "per_q_correct": None,
        "computer_guess_correct": None,
        # Mode B only
        "user_final_guess": None,  # 'ai' | 'not_ai'
        "user_guess_correct": None,
    }


# =========================
# Streamlit UI
# =========================


st.set_page_config(page_title="AI Guess Who", page_icon="🃏", layout="centered")
st.title("🃏 AI Guess Who")

# --- Sidebar: Mode & new game ---
with st.sidebar:
    st.header("Mode & Setup")
    mode = st.radio(
        "Choose game mode",
        options=["Computer guesses (app asks)", "You guess (you ask)"],
        index=0,
        help="Switch between roles: either the app tries to guess your card, or you try to guess the app’s hidden card.",
    )
    mode_key = "computer_guesses" if mode.startswith("Computer") else "you_guess"

    pick_method = st.radio("Card selection", ["Random draw", "Pick card"], horizontal=True)
    chosen_id = None
    if pick_method == "Pick card":
        # Do not leak AI/Non-AI in labels
        options = [f"{c.name}|{c.id}" for c in ALL_CARDS]
        sel = st.selectbox("Choose a card", options=options)
        chosen_id = sel.split("|")[-1]

    if st.button("Start new game", type="primary", use_container_width=True):
        reset_game(mode_key, chosen_id)
        st.rerun()

# Initialize state
if "game" not in st.session_state:
    reset_game("computer_guesses")

game = st.session_state.game
current_card = CARDS_BY_ID[game["card_id"]]

# =========================
# MODE A — Computer guesses
# =========================
if game["mode"] == "computer_guesses":
    st.subheader("🪪 Your Card")
    secs = parse_card_sections(current_card.description)
    with st.container(border=True):
        st.markdown(f"### {current_card.name}")
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

    with st.expander("Optional: technical properties (for advanced players)"):
        st.code(current_card.props, language="python")

    st.caption(
        "You know the card details—but not whether it’s an AI system. The Computer will try to guess using yes/no questions."
    )

    # Progress
    asked_records: List[Tuple[str, str, str]] = game["asked"]
    st.progress(
        len(asked_records) / MAX_QUESTIONS,
        text=f"Questions asked: {len(asked_records)} / {MAX_QUESTIONS}",
    )

    # Ask next (computer chooses)
    can_ask_more = (len(asked_records) < MAX_QUESTIONS) and (not game["completed"])
    if can_ask_more:
        q = next_question_candidate({qid for (qid, _, _) in asked_records})
        if q is None:
            game["completed"] = True
            st.rerun()

        st.subheader("🤖 Computer asks")
        st.write(q.text)
        with st.form(f"form_{q.id}", clear_on_submit=True):
            ans = st.radio("Your answer", ["Yes", "No"], index=0, horizontal=True)
            submitted = st.form_submit_button("Submit answer")
        if submitted:
            norm = "yes" if ans.lower().startswith("y") else "no"
            indicator_label = q.indicators.get(norm, "neutral")
            if indicator_label == "neutral":
                game["neutrals_used"] += 1
                if game["neutrals_used"] > MAX_NEUTRALS:
                    st.warning(
                        "Neutral limit exceeded; future neutral answers won’t be prioritized."
                    )
            game["asked"].append((q.id, norm, indicator_label))
            if len(game["asked"]) >= MAX_QUESTIONS:
                game["completed"] = True
            st.rerun()

    # Q&A so far
    if asked_records:
        st.subheader("📋 Q&A so far")
        for idx, (q_id, ans, lbl) in enumerate(asked_records, start=1):
            q_obj = next(q for q in QUESTIONS if q.id == q_id)
            weight = INDICATOR_WEIGHTS[lbl]
            label_text = {
                "def_ai": "Definitive: AI",
                "def_not_ai": "Definitive: Not AI",
                "ai_ind": "AI indicator",
                "not_ai_ind": "Not-AI indicator",
                "neutral": "Neutral",
            }[lbl]
            st.markdown(
                f"**Q{idx}.** {q_obj.text}\n\n"
                f"• **Answer:** {ans.capitalize()}  \n"
                f"• **Indicator:** {label_text} (weight {weight})"
            )

    # Finalization
    if game["completed"]:
        if game["final_guess"] is None:
            game["final_guess"] = compute_final_guess_from_indicators(game["asked"])

        st.divider()
        st.subheader("🎯 Computer’s final guess")
        guess = game["final_guess"]
        if guess == "ai":
            st.success("**This is an AI system.**")
        elif guess == "not_ai":
            st.error("**This is not an AI system.**")
        else:
            st.info("**I cannot guess based on your answers.**")

        with st.expander("How this guess was made"):
            score = sum(INDICATOR_WEIGHTS[lbl] for _, _, lbl in game["asked"])
            st.write(f"Aggregate score: **{score}**")
            st.write(
                f"Definitive AI present: **{any(lbl=='def_ai' for _,_,lbl in game['asked'])}**"
            )
            st.write(
                f"Definitive Not-AI present: **{any(lbl=='def_not_ai' for _,_,lbl in game['asked'])}**"
            )
            st.write(
                f"Neutral answers used: **{game['neutrals_used']}** / {MAX_NEUTRALS}"
            )

        # Reveal correctness against hidden label
        comp_correct = None
        if guess == "ai":
            comp_correct = current_card.is_ai_system
        elif guess == "not_ai":
            comp_correct = (not current_card.is_ai_system)
        else:
            comp_correct = None

        if comp_correct is True:
            st.success("**Computer guess correctness:** Correct → Player wins.")
        elif comp_correct is False:
            st.error("**Computer guess correctness:** Incorrect → Player loses.")
        else:
            st.info("**Computer guess correctness:** Not applicable (no guess).")

        # Player answer accuracy & points
        if game["player_points"] is None or game["per_q_correct"] is None:
            total = 0
            correctness = []
            for (q_id, ans, _lbl) in game["asked"]:
                q = next(q for q in QUESTIONS if q.id == q_id)
                truth = get_true_answer(current_card, q)
                correct = (ans == truth)
                correctness.append(correct)
                total += 2 if correct else -1
            game["player_points"] = total
            game["per_q_correct"] = correctness

        st.markdown("**Per-question truth table:**")
        for i, (q_id, ans, _lbl) in enumerate(game["asked"], start=1):
            q_obj = next(q for q in QUESTIONS if q.id == q_id)
            truth = get_true_answer(current_card, q_obj)
            correct = game["per_q_correct"][i - 1]
            if correct:
                st.write(
                    f"Q{i}: ✅ Your answer **{ans.capitalize()}** matches truth **{truth.capitalize()}**"
                )
            else:
                st.write(
                    f"Q{i}: ❌ Your answer **{ans.capitalize()}** was wrong; truth is **{truth.capitalize()}**"
                )

        st.markdown(
            f"**Scoring:** +2 per correct, −1 per incorrect  \n**Your points:** **{game['player_points']}**"
        )

        st.divider()
        st.subheader("🔁 Play again")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("New random game"):
                reset_game(game["mode"])
                st.rerun()
        with col2:
            options = [f"{c.name}|{c.id}" for c in ALL_CARDS]
            sel = st.selectbox("Or pick specific card", options=options, key="again_pick_a")
            chosen = sel.split("|")[-1]
            if st.button("Start with chosen card"):
                reset_game(game["mode"], chosen)
                st.rerun()

# =========================
# MODE B — You guess
# =========================
else:
    st.subheader("🎴 Hidden Card")
    st.info(
        "The computer is holding a card. Ask up to **5** questions from the list (or guess earlier)."
    )

    # Progress
    asked_records: List[Tuple[str, str, str]] = game["asked"]  # (q_id, comp_ans_yes_no, indicator)
    st.progress(
        len(asked_records) / MAX_QUESTIONS,
        text=f"Questions asked: {len(asked_records)} / {MAX_QUESTIONS}",
    )

    # Ask a question (you choose)
    can_ask_more = (
        (len(asked_records) < MAX_QUESTIONS)
        and (not game["completed"])
        and (game["user_final_guess"] is None)
    )
    remaining = [
        q
        for q in QUESTIONS
        if q.id not in {qid for (qid, _, _) in asked_records}
    ]
    if can_ask_more and remaining:
        st.subheader("🗣️ Ask a question")
        q_labels = [f"{q.text} | {q.id}" for q in remaining]
        picked = st.selectbox(
            "Choose a question to ask:", options=q_labels, key="you_pick_q"
        )
        selected_id = picked.split("|")[-1].strip()
        selected_q = next(q for q in QUESTIONS if q.id == selected_id)

        if st.button("Ask this question", type="primary"):
            # Computer answers truthfully
            ans = get_true_answer(current_card, selected_q)  # 'yes' / 'no'
            indicator_label = selected_q.indicators.get(ans, "neutral")
            game["asked"].append((selected_q.id, ans, indicator_label))
            st.success(f"Computer answers: **{ans.capitalize()}**")
            if len(game["asked"]) >= MAX_QUESTIONS:
                st.info("You've reached the maximum number of questions. Please make your final guess.")
            st.rerun()

    # Show Q&A so far (no indicator hints needed, but we can show them post-round)
    if asked_records:
        st.subheader("📋 Q&A so far")
        for idx, (q_id, ans, _lbl) in enumerate(asked_records, start=1):
            q_obj = next(q for q in QUESTIONS if q.id == q_id)
            st.markdown(f"**Q{idx}.** {q_obj.text}  \n• **Answer:** {ans.capitalize()}")

    # Guess controls (enabled anytime)
    st.divider()
    st.subheader("🎯 Your guess")
    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        if st.button("Guess: AI system"):
            game["user_final_guess"] = "ai"
            game["completed"] = True
            st.rerun()
    with c2:
        if st.button("Guess: Not AI"):
            game["user_final_guess"] = "not_ai"
            game["completed"] = True
            st.rerun()
    with c3:
        st.caption("You can guess at any time before or at 5 questions.")

    # Reveal & outcome
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

        # Reveal the held card & details now
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

        with st.expander("Post-game: indicators for each answer (for learning)"):
            for idx, (q_id, ans, lbl) in enumerate(game["asked"], start=1):
                q_obj = next(q for q in QUESTIONS if q.id == q_id)
                label_text = {
                    "def_ai": "Definitive: AI",
                    "def_not_ai": "Definitive: Not AI",
                    "ai_ind": "AI indicator",
                    "not_ai_ind": "Not-AI indicator",
                    "neutral": "Neutral",
                }[lbl]
                st.write(
                    f"Q{idx}: {q_obj.text} → {ans.capitalize()}  —  {label_text}"
                )

        st.divider()
        st.subheader("🔁 Play again")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("New random game"):
                reset_game(game["mode"])
                st.rerun()
        with col2:
            options = [f"{c.name}|{c.id}" for c in ALL_CARDS]
            sel = st.selectbox("Or pick specific card", options=options, key="again_pick_b")
            chosen = sel.split("|")[-1]
            if st.button("Start with chosen card"):
                reset_game(game["mode"], chosen)
                st.rerun()

# Footer
st.markdown("---")
st.caption(
    "Mode A: You see the card details; the app asks up to 5 questions and guesses.  \n"
    "Mode B: The app hides the card; you ask up to 5 questions and must guess AI vs Not AI at any time."
)

