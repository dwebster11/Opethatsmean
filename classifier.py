import streamlit as st
import pandas as pd
import re
import string
from io import BytesIO
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

st.set_page_config(page_title="Text Filter Tool", layout="wide")

st.title("🚫 Text Filter Tool with Exclusion Options and Political Scoring")
st.write("""
Upload a CSV and detect rows containing flagged words or phrases.  
Each text is also scored for political leaning (0 = conservative, 1 = liberal).
You can manage both flagged and excluded word lists below.
""")

# --- Default flagged words/phrases ---
default_flagged_words = [
    "retarded", "nigger", "hitler", "retard", "tard", "fucktard", "fuck off",
    "fuck you", "fag", "faggot", "asshole", "ass hole", "bullshit", "cock",
    "cunt", "crap", "cocksucker", "dick", "dickhead", "fucker", "go to hell",
    "go straight to hell", "slut", "shit", "tranny", "transgender", "twat",
    "splooge", "pussy", "pigfucker", "motherfucker", "mother fucker", "maga",
    "charlie", "kirk", "fuck no", "hell no", "he’ll no", "go fuck yourself",
    "fuck u", "eat shit", "you lost", "trump is king", "trump 2028", "go trump",
    "trump 2024", "red", "nigga", "idiots", "suck my", "make america great again",
    "trump2028", "gay", "f you", "fuck blue", "libtard", "fuh", "spick",
    "i voted for trump", "trump for king", "communist", "socialist", "marxist",
    "commie", "commies", "liberals", "commy", "daddy", "marx", "fucku", "get fucked",
    "suck a dick", "i love trump", "f that", "lib", "midget", "spic", "illegals",
    "illegal immigrants", "open borders", "women’s sports", "womens sports", "2028",
    "cum", "socialism", "demorats", "demons", "demonrats", "i’m a republican",
    "i vote republican", "i’m republican", "i voted for republicans", "scumbag",
    "die", "cunty", "kill yourself", "soros", "leftist", "leftists"
]

# --- Sidebar word management ---
st.sidebar.header("⚙️ Manage Word Lists")

# Initialize session state
if "flagged_words" not in st.session_state:
    st.session_state.flagged_words = default_flagged_words.copy()

if "excluded_words" not in st.session_state:
    st.session_state.excluded_words = []

# --- Manage Flagged Words ---
with st.sidebar.expander("🚨 Manage Flagged Words"):
    flagged_text = st.text_area(
        "Edit flagged words/phrases (comma-separated):",
        value=", ".join(st.session_state.flagged_words),
        height=200
    )
    if st.button("Update Flagged List"):
        new_list = [w.strip().lower() for w in flagged_text.split(",") if w.strip()]
        st.session_state.flagged_words = sorted(set(new_list))
        st.success("✅ Flagged word list updated.")

st.sidebar.write(f"Flagged terms: **{len(st.session_state.flagged_words)}**")

# --- Manage Excluded Words ---
with st.sidebar.expander("❎ Manage Excluded Words"):
    excluded_text = st.text_area(
        "Edit excluded words/phrases (comma-separated):",
        value=", ".join(st.session_state.excluded_words),
        height=200
    )
    if st.button("Update Excluded List"):
        new_list = [w.strip().lower() for w in excluded_text.split(",") if w.strip()]
        st.session_state.excluded_words = sorted(set(new_list))
        st.success("✅ Excluded word list updated.")

st.sidebar.write(f"Excluded terms: **{len(st.session_state.excluded_words)}**")

# --- Political Scoring Setup ---
@st.cache_resource
def load_model():
    # Load pre-trained model and tokenizer (replace with your fine-tuned model)
    model_name = "distilbert-base-uncased"  # Placeholder; use e.g., "matous-volf/political-leaning-politics" after fine-tuning
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
    return tokenizer, model

tokenizer, model = load_model()

def score_political_leaning(texts, batch_size=8):
    """
    Predict political leaning score (0=conservative, 1=liberal) for a list of texts.
    Uses sigmoid to normalize output to [0,1].
    """
    scores = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(
            batch, padding=True, truncation=True, max_length=512, return_tensors="pt"
        )
        with torch.no_grad():
            outputs = model(**inputs)
        batch_scores = torch.sigmoid(outputs.logits).cpu().numpy().flatten()
        scores.extend(batch_scores)
    return np.clip(scores, 0, 1)  # Ensure scores are in [0,1]

# --- File upload ---
uploaded_file = st.file_uploader("📤 Upload a CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    if "text" not in df.columns:
        st.error("❌ CSV must contain a column named 'text'.")
    else:
        st.success("✅ File uploaded successfully!")

        # Normalize text: remove punctuation, lowercase
        def normalize(text):
            if not isinstance(text, str):
                return ""
            text = text.lower().translate(str.maketrans('', '', string.punctuation))
            return text

        df["clean_text"] = df["text"].apply(normalize)

        # --- Political Scoring ---
        st.write("🔍 Computing political leaning scores (0 = conservative, 1 = liberal)...")
        texts = df["text"].astype(str).tolist()  # Convert to strings, handle NaN
        df["political_score"] = score_political_leaning(texts)
        df["leaning"] = df["political_score"].apply(
            lambda x: "Conservative" if x < 0.3 else "Liberal" if x > 0.7 else "Moderate"
        )

        # --- Create regex patterns ---
        def build_pattern(word_list):
            if not word_list:
                return None
            escaped = [re.escape(w) for w in word_list]
            return re.compile(r"\b(" + "|".join(escaped) + r")\b", re.IGNORECASE)

        flagged_pattern = build_pattern(st.session_state.flagged_words)
        excluded_pattern = build_pattern(st.session_state.excluded_words)

        # --- Identify matches ---
        def find_matches(text, pattern):
            if not pattern:
                return []
            matches = pattern.findall(text)
            return sorted(set([m.lower() for m in matches]))

        df["matched_flagged"] = df["clean_text"].apply(lambda x: find_matches(x, flagged_pattern))
        df["matched_excluded"] = df["clean_text"].apply(lambda x: find_matches(x, excluded_pattern))

        # --- Filter logic ---
        df["contains_flagged"] = df["matched_flagged"].apply(bool)
        df["contains_excluded"] = df["matched_excluded"].apply(bool)

        flagged_df = df[df["contains_flagged"] & ~df["contains_excluded"]][
            ["text", "political_score", "leaning", "matched_flagged", "matched_excluded"]
        ]

        st.write(f"**Flagged rows found (after exclusions):** {len(flagged_df)}")
        st.write(f"**Average political score of flagged rows:** {flagged_df['political_score'].mean():.3f}")

        if not flagged_df.empty:
            st.dataframe(flagged_df)

            # --- Download filtered CSV ---
            output = BytesIO()
            flagged_df.to_csv(output, index=False)
            st.download_button(
                label="📥 Download Flagged Rows as CSV",
                data=output.getvalue(),
                file_name="flagged_texts_with_scores.csv",
                mime="text/csv"
            )
        else:
            st.info("✅ No flagged content found after applying exclusions.")
else:
    st.info("⬆️ Upload a CSV file to begin.")
