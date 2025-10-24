import streamlit as st
import pandas as pd
import re
import string
from io import BytesIO

st.set_page_config(page_title="Text Filter Tool", layout="wide")

st.title("Ope That's Mean")
st.write("Upload a CSV and detect rows containing flagged words or phrases. You can also modify the flagged list.")

# --- Default flagged words/phrases ---
default_words = [
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
st.sidebar.header("⚙️ Manage Flagged Words")

if "flagged_words" not in st.session_state:
    st.session_state.flagged_words = default_words.copy()

with st.sidebar.expander("View or Edit Flagged Words"):
    words_text = st.text_area(
        "Edit flagged words/phrases (comma-separated):",
        value=", ".join(st.session_state.flagged_words),
        height=200
    )
    if st.button("Update Word List"):
        new_list = [w.strip().lower() for w in words_text.split(",") if w.strip()]
        st.session_state.flagged_words = sorted(set(new_list))
        st.success("Yay! Word list updated.")

st.sidebar.write(f"Total words/phrases: **{len(st.session_state.flagged_words)}**")

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

        # Compile regex for faster matching
        escaped_words = [re.escape(word) for word in st.session_state.flagged_words]
        pattern = re.compile(r"\b(" + "|".join(escaped_words) + r")\b", flags=re.IGNORECASE)

        # Identify which words matched
        def find_matches(text):
            matches = pattern.findall(text)
            return ", ".join(sorted(set([m.lower() for m in matches]))) if matches else ""

        df["matched_words"] = df["clean_text"].apply(find_matches)
        df["contains_flagged"] = df["matched_words"].apply(lambda x: bool(x))

        flagged_df = df[df["contains_flagged"]].drop(columns=["clean_text"])

        st.write(f"**Flagged rows found:** {len(flagged_df)}")

        if not flagged_df.empty:
            st.dataframe(flagged_df)

            # --- Download filtered CSV ---
            output = BytesIO()
            flagged_df.to_csv(output, index=False)
            st.download_button(
                label="📥 Download Flagged Rows as CSV",
                data=output.getvalue(),
                file_name="flagged_texts_with_matches.csv",
                mime="text/csv"
            )
        else:
            st.info("No flagged content found.")
else:
    st.info("Upload a CSV file to begin.")
