import random
import textwrap
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import streamlit as st

# =========================
# Utility types & constants
# =========================

@dataclass
class Question:
    id: str
    text: str
    # Map normalized answers "yes"/"no" -> indicator label
    # Allowed labels (per spec): "def_ai", "def_not_ai", "ai_ind", "not_ai_ind", "neutral"
    indicators: Dict[str, str]
    # Name of the boolean property on a card used to derive the true answer ("yes" if True else "no")
    prop: str

# Weighting scheme to aggregate signals into a final guess
INDICATOR_WEIGHTS = {
    "def_ai":  3,
    "def_not_ai": -3,
    "ai_ind":  1,
    "not_ai_ind": -1,
    "neutral": 0,
}

MAX_QUESTIONS = 5
MAX_NEUTRALS = 1

# =========================
# Knowledge base: Questions
# =========================
# Implemented exactly as provided in your specification (including edge cases).
QUESTIONS: List[Question] = [
    # 1. Inference
    Question(
        id="inf_input",
        text="Does this system receive input data to solve a problem?",
        indicators={"yes": "neutral", "no": "not_ai_ind"},
        prop="receives_input",
    ),
    Question(
        id="inf_rules_only",
        text="Does it work following only human-programmed rules?",
        # NOTE: Mapping follows the prompt verbatim:
        # yes = not ai indicator; no = definitive indicator: not ai
        indicators={"yes": "not_ai_ind", "no": "def_not_ai"},
        prop="rules_only",
    ),
    Question(
        id="inf_output",
        text="Does it generate output data that provides a solution?",
        indicators={"yes": "neutral", "no": "not_ai_ind"},
        prop="generates_output",
    ),
    Question(
        id="inf_influence",
        text="Can its outputs influence humans or other systems?",
        indicators={"yes": "neutral", "no": "not_ai_ind"},
        prop="influences",
    ),

    # 2. Outputs
    Question(
        id="out_predicts",
        text="Does the system predict something?",
        indicators={"yes": "ai_ind", "no": "not_ai_ind"},
        prop="predicts",
    ),
    Question(
        id="out_creates_content",
        text="Does it create content (text, images, video, audio)?",
        indicators={"yes": "ai_ind", "no": "not_ai_ind"},
        prop="creates_content",
    ),
    Question(
        id="out_recommends",
        text="Does it provide recommendations (e.g., suggesting actions or products)?",
        indicators={"yes": "ai_ind", "no": "not_ai_ind"},
        prop="recommends",
    ),
    Question(
        id="out_takes_decisions",
        text="Does it take decisions directly, without waiting for a human?",
        indicators={"yes": "ai_ind", "no": "neutral"},
        prop="takes_decisions_direct",
    ),

    # 3. Autonomy
    Question(
        id="auton_can_do_on_own",
        text="Can the system do something on its own once it receives input (without a person pressing every button)?",
        indicators={"yes": "ai_ind", "no": "def_not_ai"},
        prop="can_do_on_own",
    ),
    Question(
        id="auton_full_autonomy",
        text="Does it act with full autonomy, making decisions that directly affect the world without human review?",
        indicators={"yes": "ai_ind", "no": "def_not_ai"},
        prop="acts_full_autonomy",
    ),
    Question(
        id="auton_limited",
        text="Does it have limited autonomy (provides outputs but still needs humans to decide or act)?",
        indicators={"yes": "ai_ind", "no": "neutral"},
        prop="limited_autonomy",
    ),
    Question(
        id="auton_non_autonomous",
        text="Is it non-autonomous, only working step by step when a human tells it exactly what to do?",
        indicators={"yes": "def_not_ai", "no": "ai_ind"},
        prop="non_autonomous",
    ),

    # 4. Adaptiveness
    Question(
        id="adapt_never_changes",
        text="Does the system stay the same and never change how it behaves?",
        indicators={"yes": "neutral", "no": "ai_ind"},
        prop="never_changes",
    ),
    Question(
        id="adapt_online_learns",
        text="Does it adapt and learn from new data while it operates?",
        indicators={"yes": "def_ai", "no": "neutral"},
        prop="adapts_online",
    ),
]

# =======================================
# Knowledge base: Cards and true answers
# =======================================

@dataclass
class Card:
    id: str
    name: str
    is_ai_system: bool
    description: str
    # Properties used by questions above (booleans)
    props: Dict[str, bool] = field(default_factory=dict)

def wrap_desc(s: str) -> str:
    return "\n".join(textwrap.wrap(s, width=80))

# Helper to quickly define properties with sensible defaults
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

# --- AI System Cards ---
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
            adapts_online=False,  # usually not online-learning in operation
            rules_only=False,
        )
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
            takes_decisions_direct=True,     # auto routes to spam/inbox
            can_do_on_own=True,
            limited_autonomy=True,
            non_autonomous=False,
            never_changes=False,
            adapts_online=False,
            rules_only=False,
            influences=True,  # affects other systems / users
        )
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
            recommends=True,   # suggests candidates
            can_do_on_own=True,
            limited_autonomy=True,
            adapts_online=False,
            rules_only=False,
        )
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
            adapts_online=True,   # updates online per description
            rules_only=False,
        )
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
            predicts=True,  # classification/sequence prediction
            creates_content=False,
            recommends=False,
            takes_decisions_direct=False,
            can_do_on_own=True,
            limited_autonomy=True,
            adapts_online=False,
            rules_only=False,
        )
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
        )
    ),
    Card(
        id="ai_screen",
        name="Job Applicant Screening Tool",
        is_ai_system=True,
        description=wrap_desc(
            "Input: Candidate CVs and application details. "
            "How it works: developed using past hiring outcomes to learn which candidate characteristics are a good fit. "
            "Objective: Predict candidate–job match. "
            "Output: (ranked candidates / fit score)."
        ),
        props=mk_props(
            predicts=True,
            recommends=True,         # recommends candidates
            takes_decisions_direct=False,
            creates_content=False,
            can_do_on_own=True,
            limited_autonomy=True,
            adapts_online=False,
            rules_only=False,
            influences=True,
        )
    ),
]

# --- Non-AI System Cards ---
# These still answer truthfully to prediction/recommendation questions where applicable,
# but are considered non-AI because they operate via simple, predefined rules.
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
        )
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
        )
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
            influences=True,  # informs humans, but simple
        )
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
        )
    ),
    Card(
        id="na_inventory_forecaster",
        name="Inventory Forecaster (Averaging)",
        is_ai_system=False,
        description=wrap_desc(
            "Input: Sales from the past week. "
            "How it works: Takes the average of items sold per day. "
            "Objective: Estimate how many items will be sold tomorrow. "
            "Output: A single number prediction."
        ),
        props=mk_props(
            rules_only=True,
            predicts=True,          # yes, via simple average
            recommends=False,
            takes_decisions_direct=False,
            can_do_on_own=False,
            limited_autonomy=False,
            non_autonomous=True,
            never_changes=True,
            influences=True,
        )
    ),
    Card(
        id="na_ticket_eta",
        name="Customer Service Time Estimator (Averaging)",
        is_ai_system=False,
        description=wrap_desc(
            "Input: Records of past support tickets. "
            "How it works: Calculates the average time it took to close them. "
            "Objective: Predict how long new tickets will take. "
            "Output: An estimated resolution time."
        ),
        props=mk_props(
            rules_only=True,
            predicts=True,          # average-based estimate
            recommends=False,
            takes_decisions_direct=False,
            can_do_on_own=False,
            limited_autonomy=False,
            non_autonomous=True,
            never_changes=True,
            influences=True,
        )
    ),
]

ALL_CARDS: List[Card] = CARDS_AI + CARDS_NON_AI
CARDS_BY_ID: Dict[str, Card] = {c.id: c for c in ALL_CARDS}

# =========================
# Game state & logic
# =========================

def reset_game(random_draw: bool = True, chosen_card_id: Optional[str] = None):
    if random_draw:
        card = random.choice(ALL_CARDS)
    else:
        card = CARDS_BY_ID[chosen_card_id] if chosen_card_id else random.choice(ALL_CARDS)

    st.session_state.game = {
        "card_id": card.id,
        "revealed": False,
        "asked": [],              # list of (question_id, answer_yes_no, indicator_label)
        "neutrals_used": 0,
        "completed": False,
        "final_guess": None,      # "ai" | "not_ai" | "cannot_guess"
        "player_points": None,
        "per_q_correct": None,    # list of booleans aligned with asked
        "computer_guess_correct": None,
    }

def normalize_answer(ans: str) -> str:
    return ans.strip().lower()

def indicator_label_to_weight(lbl: str) -> int:
    return INDICATOR_WEIGHTS.get(lbl, 0)

def compute_final_guess(asked: List[Tuple[str, str, str]]) -> str:
    """
    asked: list of (q_id, ans, indicator_label)
    Final guess per rules:
      - If any definitive AI and no definitive Not AI -> "ai"
      - If any definitive Not AI and no definitive AI -> "not_ai"
      - If both present -> "cannot_guess"
      - Else use score: sum(weights) > 0 -> "ai"; < 0 -> "not_ai"; == 0 -> "cannot_guess"
    """
    has_def_ai = any(lbl == "def_ai" for _, _, lbl in asked)
    has_def_not_ai = any(lbl == "def_not_ai" for _, _, lbl in asked)
    if has_def_ai and not has_def_not_ai:
        return "ai"
    if has_def_not_ai and not has_def_ai:
        return "not_ai"
    if has_def_ai and has_def_not_ai:
        return "cannot_guess"
    score = sum(indicator_label_to_weight(lbl) for _, _, lbl in asked)
    if score > 0:
        return "ai"
    elif score < 0:
        return "not_ai"
    else:
        return "cannot_guess"

def get_true_answer(card: Card, q: Question) -> str:
    val = bool(card.props.get(q.prop, False))
    return "yes" if val else "no"

def score_player_answers(card: Card, asked: List[Tuple[str, str, str]]) -> Tuple[int, List[bool]]:
    """
    +2 for each correct yes/no; -1 for each incorrect yes/no
    """
    total = 0
    correctness = []
    for q_id, ans, _ in asked:
        q = next(q for q in QUESTIONS if q.id == q_id)
        truth = get_true_answer(card, q)
        correct = normalize_answer(ans) == truth
        correctness.append(correct)
        total += 2 if correct else -1
    return total, correctness

def next_question_candidate(asked_ids: set, neutrals_used: int) -> Optional[Question]:
    """
    Simple heuristic:
      - Prefer questions not yet asked.
      - Prefer those whose possible outcomes include non-neutral indicators.
      - Allow at most one neutral outcome to be registered (we enforce after answer).
      - Priority ordering: any with a potential 'def_*' in mapping > 'ai_ind'/'not_ai_ind' > 'neutral-only'
      - We keep the natural order within the defined QUESTION list for simplicity.
    """
    def priority(q: Question) -> int:
        vals = set(q.indicators.values())
        # Highest if can produce a definitive outcome
        if "def_ai" in vals or "def_not_ai" in vals:
            return 3
        # Next if produces directional indicators
        if "ai_ind" in vals or "not_ai_ind" in vals:
            return 2
        # all neutral
        return 1

    candidates = [q for q in QUESTIONS if q.id not in asked_ids]
    if not candidates:
        return None
    candidates.sort(key=priority, reverse=True)
    return candidates[0]

def reveal_and_finalize():
    st.session_state.game["revealed"] = True

def complete_game():
    st.session_state.game["completed"] = True

# ==============
# Streamlit UI
# ==============

st.set_page_config(page_title="AI Guess Who", page_icon="🃏", layout="centered")

st.title("🃏 AI Guess Who")

with st.sidebar:
    st.header("Setup")
    if "game" not in st.session_state:
        reset_game(random_draw=True)

    game = st.session_state.game
    current_card = CARDS_BY_ID[game["card_id"]]

    st.markdown("**Card Holder** draws a random card. The Computer (this app) is the Guesser.")
    reveal = st.toggle("Reveal card to player", value=game["revealed"], help="Keep hidden to play honestly; reveal for testing.")
    if reveal != game["revealed"]:
        reveal_and_finalize()

    if reveal:
        st.success(f"**Card:** {current_card.name} — {'AI System' if current_card.is_ai_system else 'Non-AI System'}")
        with st.expander("Card details", expanded=False):
            st.write(current_card.description)
            st.code(current_card.props, language="python")

    st.divider()
    st.subheader("New game")
    pick_method = st.radio("Card selection", ["Random draw", "Pick card"], horizontal=True)
    chosen_id = None
    if pick_method == "Pick card":
        options = [f"{c.name} ({'AI' if c.is_ai_system else 'Non-AI'})|{c.id}" for c in ALL_CARDS]
        sel = st.selectbox("Choose a card to hold", options=options)
        chosen_id = sel.split("|")[-1]
    if st.button("Start new game", type="primary", use_container_width=True):
        reset_game(random_draw=(pick_method=="Random draw"), chosen_card_id=chosen_id)
        st.rerun()

st.caption("Rules: The Computer asks up to 5 yes/no questions (max 1 neutral). Then it guesses whether the hidden card is an AI system.")

# Gameplay area
game = st.session_state.game
current_card = CARDS_BY_ID[game["card_id"]]
asked_records: List[Tuple[str, str, str]] = game["asked"]
asked_ids = {q_id for (q_id, _, _) in asked_records}

# Progress
st.progress(len(asked_records) / MAX_QUESTIONS, text=f"Questions asked: {len(asked_records)} / {MAX_QUESTIONS}")

# Ask next question if we still can
can_ask_more = (len(asked_records) < MAX_QUESTIONS) and (not game["completed"])

if can_ask_more:
    q = next_question_candidate(asked_ids, game["neutrals_used"])
    if q is None:
        game["completed"] = True
        st.rerun()
    st.subheader("🤖 Computer asks:")
    st.write(q.text)
    with st.form(f"form_{q.id}", clear_on_submit=True):
        ans = st.radio("Your answer", options=["Yes", "No"], index=0, horizontal=True)
        submitted = st.form_submit_button("Submit answer")
    if submitted:
        norm = "yes" if ans.lower().startswith("y") else "no"
        indicator_label = q.indicators.get(norm, "neutral")
        # Respect neutral limit: we still record the Q&A, but if this answer yields neutral and we have used one already,
        # we simply count it but do not allow further neutrals later (the rule says max 1 neutral question; this enforces the limit)
        neutrals_used = game["neutrals_used"]
        if indicator_label == "neutral":
            if neutrals_used >= MAX_NEUTRALS:
                st.warning("Neutral limit already reached; this neutral answer still recorded, but no more neutral questions will be allowed.")
            game["neutrals_used"] = neutrals_used + 1

        game["asked"].append((q.id, norm, indicator_label))

        # Auto-complete when max reached
        if len(game["asked"]) >= MAX_QUESTIONS:
            game["completed"] = True
        st.rerun()

# Show asked Q&A so far
if asked_records:
    st.subheader("📋 Q&A so far")
    for idx, (q_id, ans, lbl) in enumerate(asked_records, start=1):
        q_obj = next(q for q in QUESTIONS if q.id == q_id)
        weight = indicator_label_to_weight(lbl)
        label_text = {
            "def_ai": "Definitive: AI",
            "def_not_ai": "Definitive: Not AI",
            "ai_ind": "AI indicator",
            "not_ai_ind": "Not-AI indicator",
            "neutral": "Neutral",
        }[lbl]
        st.markdown(f"**Q{idx}.** {q_obj.text}\n\n• **Answer:** {ans.capitalize()}  \n• **Indicator:** {label_text} (weight {weight})")

# If completed, compute guess
if game["completed"]:
    if game["final_guess"] is None:
        guess = compute_final_guess(game["asked"])
        game["final_guess"] = guess

    st.divider()
    st.subheader("🎯 Computer’s final guess")
    guess = game["final_guess"]
    if guess == "ai":
        st.success("**This is an AI system.**")
    elif guess == "not_ai":
        st.error("**This is not an AI system.**")
    else:
        st.info("**I cannot guess based on your answers.**")

    with st.expander("How this guess was made", expanded=False):
        score = sum(indicator_label_to_weight(lbl) for _, _, lbl in game["asked"])
        st.write(f"Aggregate score: **{score}**")
        st.write(f"Definitive AI present: **{any(lbl=='def_ai' for _,_,lbl in game['asked'])}**")
        st.write(f"Definitive Not-AI present: **{any(lbl=='def_not_ai' for _,_,lbl in game['asked'])}**")
        st.write(f"Neutral answers used: **{game['neutrals_used']}** / {MAX_NEUTRALS}")

    # Player reveals correctness & per-question truth
    st.divider()
    st.subheader("🧪 Reveal & Score")

    # Reveal actual card type against guess
    comp_correct = None
    if guess == "ai":
        comp_correct = current_card.is_ai_system
    elif guess == "not_ai":
        comp_correct = (not current_card.is_ai_system)
    elif guess == "cannot_guess":
        comp_correct = None  # neither correct nor incorrect; outcome is indeterminate

    if comp_correct is True:
        st.success("**Computer guess correctness:** Correct (per Winning rules, the player wins).")
    elif comp_correct is False:
        st.error("**Computer guess correctness:** Incorrect (per Winning rules, the player loses).")
    else:
        st.info("**Computer guess correctness:** Not applicable (no guess).")

    game["computer_guess_correct"] = comp_correct

    # Per-question correctness and scoring
    if game["player_points"] is None or game["per_q_correct"] is None:
        pts, corr = score_player_answers(current_card, game["asked"])
        game["player_points"] = pts
        game["per_q_correct"] = corr

    st.markdown("**Per-question truth table:**")
    for i, (q_id, ans, lbl) in enumerate(game["asked"], start=1):
        q_obj = next(q for q in QUESTIONS if q.id == q_id)
        truth = get_true_answer(current_card, q_obj)
        correct = game["per_q_correct"][i-1]
        if correct:
            st.write(f"Q{i}: ✅ Your answer **{ans.capitalize()}** matches truth **{truth.capitalize()}**")
        else:
            st.write(f"Q{i}: ❌ Your answer **{ans.capitalize()}** was wrong; truth is **{truth.capitalize()}**")

    st.markdown(f"**Scoring system:** +2 per correct, −1 per incorrect  \n**Your points:** **{game['player_points']}**")

    st.divider()
    st.subheader("🔁 Play again")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("New random game"):
            reset_game(random_draw=True)
            st.rerun()
    with col2:
        options = [f"{c.name} ({'AI' if c.is_ai_system else 'Non-AI'})|{c.id}" for c in ALL_CARDS]
        sel = st.selectbox("Or pick specific card", options=options, key="again_pick")
        chosen = sel.split("|")[-1]
        if st.button("Start with chosen card"):
            reset_game(random_draw=False, chosen_card_id=chosen)
            st.rerun()

# Footer
st.markdown("---")
st.caption(
    "Gameplay rules, question indicators, and card definitions are implemented as provided. "
    "Note: Some indicator mappings may look counter-intuitive but are preserved verbatim."
)
