# app.py
"""
Final Streamlit app: Gender Bias Detector + Inclusive Rewriter
Save as app.py and run with: streamlit run app.py

Dependencies:
pip install streamlit pandas numpy scikit-learn spacy nltk matplotlib seaborn joblib
python -m spacy download en_core_web_sm
python -m nltk.downloader punkt stopwords wordnet
"""
import streamlit as st
import pandas as pd
import numpy as np
import re
import os
from collections import Counter
from typing import List, Dict
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# ML / NLP
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import MiniBatchKMeans

# try to import spaCy + NLTK; handle missing gracefully
try:
    import spacy
    nlp = spacy.load("en_core_web_sm", disable=['ner'])
except Exception:
    nlp = None

try:
    import nltk
    from nltk.stem import WordNetLemmatizer
    from nltk.corpus import stopwords
    # ensure minimal downloads (if missing, user should run the download commands)
    try:
        _ = stopwords.words('english')
    except Exception:
        nltk.download('stopwords')
        nltk.download('punkt')
        nltk.download('wordnet')
    lemmatizer = WordNetLemmatizer()
    STOPWORDS = set(stopwords.words('english')) - {"he", "she", "him", "her", "they", "them"}
except Exception:
    # fallback simple tokenizer/lemmatizer if nltk missing
    lemmatizer = None
    STOPWORDS = set()

# ---------------------------
# Base bias dictionary (default)
# ---------------------------
BASE_BIAS_DICT: Dict[str, str] = {
    # pronouns
    "he": "they", "she": "they", "him": "them", "her": "them", "his": "their", "hers": "theirs",
    "himself": "themself", "herself": "themself",
    # gendered occupational words / titles
    "chairman": "chairperson", "chairwoman": "chairperson",
    "businessman": "businessperson", "businesswoman": "businessperson",
    "salesman": "salesperson", "saleswoman": "salesperson",
    "policeman": "police officer", "policewoman": "police officer",
    "fireman": "firefighter", "firewoman": "firefighter",
    "waitress": "server", "waiter": "server",
    "stewardess": "flight attendant", "steward": "flight attendant",
    "congressman": "representative", "congresswoman": "representative",
    "mailman": "mail carrier", "postman": "mail carrier",
    "actress": "actor",
    "housewife": "home manager", "homemaker": "home manager",
    # stereotypical adjectives -> neutral alternatives
    "aggressive": "assertive", "dominant": "confident", "nurturing": "supportive",
    "emotional": "expressive", "soft": "approachable",
    "feminine": "approachable", "masculine": "assertive",
    # identity mappings for common occupations
    "developer": "developer", "engineer": "engineer", "nurse": "nurse",
    "secretary": "administrative assistant", "receptionist": "receptionist",
    "cleaner": "cleaner", "chef": "chef", "driver": "driver",
    "mechanic": "mechanic", "plumber": "plumber", "electrician": "electrician",
    "manager": "manager", "director": "director", "assistant": "assistant",
}

BASE_BIAS_DICT = {k.lower(): v for k, v in BASE_BIAS_DICT.items()}

# ---------------------------
# Session state defaults
# ---------------------------
if 'bias_dict' not in st.session_state:
    st.session_state.bias_dict = dict(BASE_BIAS_DICT)
if 'vectorizer' not in st.session_state:
    st.session_state.vectorizer = None
if 'classifier' not in st.session_state:
    st.session_state.classifier = None

# ---------------------------
# Helpers
# ---------------------------
def rebuild_word_regex(bias_map: Dict[str, str]):
    if not bias_map:
        return re.compile(r"^\b$a")
    # sort by length desc to avoid partial matches
    keys = sorted(bias_map.keys(), key=lambda s: -len(s))
    pattern = r"\b(" + "|".join(re.escape(k) for k in keys) + r")\b"
    return re.compile(pattern, flags=re.IGNORECASE)

def clean_text(txt: str) -> str:
    if not isinstance(txt, str):
        return ""
    s = txt.strip()
    s = re.sub(r'[\r\n]+', ' ', s)
    s = re.sub(r'[^A-Za-z0-9\s/.-]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s.lower()

def simple_tokenize_lemmatize(txt: str) -> List[str]:
    # prefer spaCy + NLTK if available, otherwise fallback naive split
    txt = str(txt)
    if nlp is not None and lemmatizer is not None:
        doc = nlp(txt)
        toks = []
        for tok in doc:
            w = tok.text.strip()
            if not w:
                continue
            if w.lower() in STOPWORDS:
                continue
            lw = lemmatizer.lemmatize(w.lower())
            toks.append(lw)
        return toks
    else:
        # fallback
        toks = [w for w in re.findall(r"\w+", txt.lower()) if w not in STOPWORDS]
        return toks

def detect_biased_tokens(text: str, bias_map: Dict[str,str]) -> List[str]:
    WORD_RE = rebuild_word_regex(bias_map)
    matches = WORD_RE.findall(text)
    found = list({m.lower() for m in matches})
    return found

def replace_biases_in_text(text: str, bias_map: Dict[str,str]) -> str:
    WORD_RE = rebuild_word_regex(bias_map)
    def repl(m):
        token = m.group(0)
        low = token.lower()
        replacement = bias_map.get(low, token)
        if token.isupper():
            return replacement.upper()
        if token[0].isupper():
            return replacement.capitalize()
        return replacement
    return re.sub(WORD_RE, repl, text)

def augment_dataset(df: pd.DataFrame, text_col='text') -> pd.DataFrame:
    rows = []
    swaps = [(" he ", " she "), (" he,", " she,"), (" she ", " he "), (" she,", " he,"),
             (" his ", " her "), (" her ", " his ")]
    for s in df[text_col].astype(str).values:
        s0 = " " + s + " "
        rows.append(s)
        for a,b in swaps:
            if a.strip() in s.lower():
                rows.append(s.replace(a.strip(), b.strip()))
        rows.append(replace_biases_in_text(s, st.session_state.bias_dict))
    unique = list(dict.fromkeys(rows))
    return pd.DataFrame({text_col: unique})

# ---------------------------
# UI
# ---------------------------
st.set_page_config(page_title="Gender Bias Detector & Neutralizer", layout="wide")
st.title("Gender Bias Detector & Inclusive Rewriter")

with st.sidebar:
    st.header("Controls & Dictionary")
    uploaded_data = st.file_uploader("Upload sentences file (TSV/CSV) - optional", type=['tsv','csv','txt'])
    uploaded_occ = st.file_uploader("Upload occupations file (optional)", type=['tsv','csv','txt'])
    augment_toggle = st.checkbox("Enable augmentation for training", value=True)
    st.write("Dictionary entries:", len(st.session_state.bias_dict))
    if st.button("Download current bias dictionary"):
        txt = "\n".join(f"{k} -> {v}" for k,v in st.session_state.bias_dict.items())
        st.download_button("Download", txt, file_name="bias_dict.txt")
    st.markdown("---")
    st.markdown("Edit dictionary (format: source -> replacement)")
    dict_text = "\n".join(f"{k} -> {v}" for k,v in st.session_state.bias_dict.items())
    edited = st.text_area("Edit bias dictionary", value=dict_text, height=220)
    if st.button("Apply dictionary changes"):
        newmap = {}
        for line in edited.splitlines():
            if "->" in line:
                left,right = line.split("->",1)
                key = left.strip().lower()
                val = right.strip()
                if key:
                    newmap[key] = val if val else key
        if newmap:
            st.session_state.bias_dict = newmap
            st.success(f"Dictionary updated: {len(newmap)} entries")
        else:
            st.warning("No valid mappings found. Use format: source -> replacement")

# import occupations if uploaded
if uploaded_occ is not None:
    try:
        occ_df = pd.read_csv(uploaded_occ, sep=None, engine='python', header=0)
    except Exception:
        occ_df = pd.read_csv(uploaded_occ, header=None, names=['occ'])
    occ_list = []
    for col in occ_df.columns:
        if occ_df[col].dtype == object:
            occ_list += occ_df[col].dropna().astype(str).str.strip().tolist()
    occ_list = [o for o in occ_list if len(o)>1]
    added = 0
    for o in occ_list:
        key = o.lower()
        if key not in st.session_state.bias_dict:
            st.session_state.bias_dict[key] = key
            added += 1
    if added:
        st.sidebar.success(f"Imported {added} occupations into dictionary")

# Load dataset (uploaded or sample)
if uploaded_data is not None:
    try:
        if uploaded_data.name.lower().endswith('.tsv') or uploaded_data.name.lower().endswith('.txt'):
            df = pd.read_csv(uploaded_data, sep=None, engine='python', header=None, names=['text'])
        else:
            df = pd.read_csv(uploaded_data, header=None, names=['text'])
        st.success(f"Loaded {len(df)} rows from upload.")
    except Exception:
        df = pd.read_csv(uploaded_data, sep='\t', header=None, names=['text'])
else:
    # try local file
    if os.path.exists("all_sentences.tsv"):
        try:
            df = pd.read_csv("all_sentences.tsv", sep=None, engine='python', header=None, names=['text'])
            st.info(f"Loaded local all_sentences.tsv ({len(df)} rows).")
        except Exception:
            df = pd.read_csv("all_sentences.tsv", sep='\t', header=None, names=['text'])
    else:
        # fallback sample
        sample = [
            "The chairman will lead the team and he must be confident.",
            "We are looking for a nurturing candidate who can manage the home.",
            "The waitress serves customers.",
            "A strong engineer needed to work on backend systems.",
            "The businesswoman spoke with the manager.",
            "He should be assertive and dominant.",
            "She was emotional during the meeting."
        ]
        df = pd.DataFrame({"text": sample})
        st.warning("No dataset provided. Using a small sample.")

# Preprocess and compute bias info
df['clean'] = df['text'].astype(str).apply(clean_text)
df['tokens'] = df['clean'].apply(simple_tokenize_lemmatize)
df['biased_words'] = df['text'].astype(str).apply(lambda t: detect_biased_tokens(t, st.session_state.bias_dict))
df['bias_score'] = df['biased_words'].apply(len)

st.subheader("Dataset preview")
st.dataframe(df[['text','biased_words','bias_score']].head(10))

# Stats
total = len(df)
biased_count = int((df['bias_score']>0).sum())
st.metric("Total sentences", total)
st.metric("Biased sentences", f"{biased_count} ({biased_count/total*100:.1f}%)")

# Frequency
all_biased = [w for row in df['biased_words'] for w in row]
freq = Counter(all_biased)
freq_df = pd.DataFrame(freq.most_common(25), columns=['word','count'])
if not freq_df.empty:
    st.subheader("Top biased tokens")
    st.bar_chart(freq_df.set_index('word'))
# --- Insert after df has columns: 'text','clean','tokens','biased_words','bias_score' ---

import scipy.stats as stats
from statistics import mean, median, mode, StatisticsError
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

# ensure sentence length column exists
if 'sent_len' not in df.columns:
    df['sent_len'] = df['tokens'].apply(len)

# ---------- Statistical summary ----------
def descriptive_stats(series):
    vals = series.dropna().astype(float)
    stats_dict = {
        "count": int(vals.count()),
        "mean": float(vals.mean()),
        "median": float(vals.median()) if len(vals)>0 else None,
        "mode": None,
        "variance": float(vals.var(ddof=0)) if len(vals)>1 else 0.0,
        "std": float(vals.std(ddof=0)) if len(vals)>1 else 0.0,
        "min": float(vals.min()) if len(vals)>0 else None,
        "max": float(vals.max()) if len(vals)>0 else None,
        "skewness": float(stats.skew(vals)) if len(vals)>2 else 0.0,
        "kurtosis": float(stats.kurtosis(vals)) if len(vals)>2 else 0.0
    }
    try:
        stats_dict["mode"] = float(mode(vals))
    except StatisticsError:
        stats_dict["mode"] = None
    return stats_dict

st.subheader("Statistical Summary")
colA, colB = st.columns(2)

with colA:
    st.markdown("**Bias Score (overall)**")
    bias_stats = descriptive_stats(df['bias_score'])
    st.table(pd.DataFrame.from_dict(bias_stats, orient='index', columns=['Bias Score']).rename_axis('stat'))
with colB:
    st.markdown("**Sentence Length (words)**")
    len_stats = descriptive_stats(df['sent_len'])
    st.table(pd.DataFrame.from_dict(len_stats, orient='index', columns=['Sentence Length']).rename_axis('stat'))

# ---------- Group comparison: biased vs neutral ----------
df['is_biased'] = (df['bias_score'] > 0).astype(int)
biased_group = df[df['is_biased']==1]['sent_len']
neutral_group = df[df['is_biased']==0]['sent_len']

st.subheader("Group Comparison: Biased vs Neutral Sentences")
c1, c2 = st.columns(2)
with c1:
    st.markdown("**Counts**")
    st.write("Biased sentences:", int(df['is_biased'].sum()))
    st.write("Neutral sentences:", int((df['is_biased']==0).sum()))
with c2:
    st.markdown("**Average lengths**")
    st.write("Mean length (biased):", round(biased_group.mean() if len(biased_group)>0 else 0,2))
    st.write("Mean length (neutral):", round(neutral_group.mean() if len(neutral_group)>0 else 0,2))

# Mann-Whitney U test (non-parametric length comparison)
try:
    if len(biased_group)>0 and len(neutral_group)>0:
        mw_stat, mw_p = stats.mannwhitneyu(biased_group, neutral_group, alternative='two-sided')
        st.markdown(f"**Mann-Whitney U test** for sentence length difference: U = {mw_stat:.2f}, p = {mw_p:.4f}")
    else:
        st.info("Not enough data for Mann-Whitney U test.")
except Exception as e:
    st.write("Mann-Whitney U test error:", e)

# ---------- More visuals - interactive ----------
st.subheader("More Visualizations")
viz_choice = st.selectbox("Choose visualization", [
    "Bias score distribution",
    "Top biased words",
    "Sentence length vs Bias (scatter + fit)",
    "Boxplot of sentence length by bias",
    "Bias proportion",
    "Correlation heatmap"
])

# helper: top biased words
all_biased = [w for row in df['biased_words'] for w in row]  # ensure biased_words are lists
bias_counts = Counter(all_biased)
bias_freq_df = pd.DataFrame(bias_counts.most_common(30), columns=['word','count'])

if viz_choice == "Bias score distribution":
    fig, ax = plt.subplots(figsize=(7,3.5))
    sns.histplot(df['bias_score'], bins=range(0, int(df['bias_score'].max())+2), kde=True, ax=ax)
    ax.set_title("Bias Score Distribution")
    ax.set_xlabel("Bias score")
    ax.set_ylabel("Count")
    st.pyplot(fig)

elif viz_choice == "Top biased words":
    if bias_freq_df.empty:
        st.info("No biased tokens found to plot.")
    else:
        fig, ax = plt.subplots(figsize=(7,4))
        sns.barplot(data=bias_freq_df.head(20), x='count', y='word', ax=ax)
        ax.set_title("Top biased words (most common)")
        st.pyplot(fig)
        st.markdown("Download frequency table:")
        st.download_button("Download biased word frequencies (CSV)", bias_freq_df.to_csv(index=False), file_name="biased_word_freq.csv")

elif viz_choice == "Sentence length vs Bias (scatter + fit)":
    fig, ax = plt.subplots(figsize=(7,4))
    sns.scatterplot(data=df, x='sent_len', y='bias_score', alpha=0.6, ax=ax)
    # linear fit
    try:
        coeffs = np.polyfit(df['sent_len'], df['bias_score'], 1)
        xvals = np.linspace(df['sent_len'].min(), df['sent_len'].max(), 100)
        ax.plot(xvals, coeffs[0]*xvals + coeffs[1], color='red', linestyle='--', label=f'fit: y={coeffs[0]:.3f}x+{coeffs[1]:.3f}')
        ax.legend()
    except Exception:
        pass
    ax.set_title("Sentence length vs Bias score")
    ax.set_xlabel("Sentence length (words)")
    ax.set_ylabel("Bias score")
    st.pyplot(fig)
    st.markdown(f"Pearson correlation: {df['sent_len'].corr(df['bias_score']):.3f}")

elif viz_choice == "Boxplot of sentence length by bias":
    fig, ax = plt.subplots(figsize=(7,4))
    sns.boxplot(data=df, x='is_biased', y='sent_len', ax=ax)
    ax.set_xticklabels(['Neutral (0)','Biased (1)'])
    ax.set_title("Sentence length by Bias group")
    ax.set_ylabel("Sentence length (words)")
    st.pyplot(fig)

elif viz_choice == "Bias proportion":
    fig, ax = plt.subplots(figsize=(5,4))
    counts = df['is_biased'].value_counts().sort_index()
    labels = ['Neutral','Biased']
    ax.pie(counts, labels=labels, autopct='%1.1f%%', colors=['#99ff99','#ff9999'])
    ax.set_title("Proportion Biased vs Neutral")
    st.pyplot(fig)

elif viz_choice == "Correlation heatmap":
    corr_df = df[['sent_len','bias_score']].corr()
    fig, ax = plt.subplots(figsize=(4,3))
    sns.heatmap(corr_df, annot=True, fmt=".3f", cmap='coolwarm', ax=ax)
    ax.set_title("Correlation matrix")
    st.pyplot(fig)

# ---------- Extra: grouped occupations summary ----------
st.subheader("Counts by occupation (if occupation tokens present)")
# find occupation mentions by matching against bias_dict keys (some are identity mapped)
occ_keys = [k for k,v in st.session_state.bias_dict.items() if v==k or k in ["manager","engineer","developer","nurse","teacher","doctor"]]
# naive count: count occurrences of each occupation keyword in text
occ_counter = Counter()
for k in occ_keys:
    occ_counter[k] = df['text'].str.contains(r'\b' + re.escape(k) + r'\b', case=False, na=False).sum()
occ_df = pd.DataFrame.from_records(list(occ_counter.items()), columns=['occupation','count']).sort_values('count', ascending=False)
if not occ_df.empty:
    st.table(occ_df.head(20))
else:
    st.info("No occupation keywords matched (try uploading occupations file or expanding dictionary).")


# Histogram
fig, ax = plt.subplots(figsize=(6,3))
sns.histplot(df['bias_score'], bins=range(0, df['bias_score'].max()+2), ax=ax)
ax.set_xlabel("Bias score"); ax.set_ylabel("Count")
st.pyplot(fig)

# ---------------------------
# Model training (weak supervision)
# ---------------------------
st.subheader("Train classifier (weak supervision: bias_score>0)")
label = (df['bias_score']>0).astype(int)
X_text = df['clean'].fillna("")
if augment_toggle:
    aug = augment_dataset(df, 'text')
    aug['clean'] = aug['text'].astype(str).apply(clean_text)
    aug['biased_words'] = aug['text'].astype(str).apply(lambda t: detect_biased_tokens(t, st.session_state.bias_dict))
    aug['bias_score'] = aug['biased_words'].apply(len)
    aug['label'] = (aug['bias_score']>0).astype(int)
    combined = pd.concat([df[['clean']].assign(label=label), aug[['clean','label']]]).drop_duplicates().reset_index(drop=True)
    X_all = combined['clean'].fillna("")
    y_all = combined['label'].values
    st.info(f"Training set size after augmentation: {len(combined)}")
else:
    X_all = X_text
    y_all = label.values

if st.button("Train classifier"):
    if len(np.unique(y_all)) == 1:
        st.error("Need both classes to train. Provide more varied data or enable augmentation.")
    else:
        vect = TfidfVectorizer(ngram_range=(1,2), min_df=1)
        Xv = vect.fit_transform(X_all)
        X_train, X_test, y_train, y_test = train_test_split(Xv, y_all, test_size=0.2, random_state=42, stratify=y_all)
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train, y_train)
        ypred = clf.predict(X_test)
        report = classification_report(y_test, ypred, output_dict=True, zero_division=0)
        st.session_state.vectorizer = vect
        st.session_state.classifier = clf
        st.success("Classifier trained and stored in session.")
        st.json(report)
        cm = confusion_matrix(y_test, ypred)
        fig2, ax2 = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', ax=ax2, cmap='Blues')
        ax2.set_xlabel("Pred"); ax2.set_ylabel("True")
        st.pyplot(fig2)
        try:
            feat_names = np.array(vect.get_feature_names_out())
            coefs = clf.coef_[0]
            top_idx = np.argsort(coefs)[-20:][::-1]
            top_feats = feat_names[top_idx]
            top_vals = coefs[top_idx]
            st.table(pd.DataFrame({"feature":top_feats, "coef":top_vals}))
        except Exception:
            pass

# ---------------------------
# Interactive single-sentence analyzer
# ---------------------------
st.header("Interactive: Analyze & Repair a sentence")
input_text = st.text_area("Paste a sentence:", value="We are hiring a chairman — he should be assertive and not too emotional.", height=140)
show_model = st.checkbox("Show model prediction (if trained)", value=True)
if st.button("Analyze"):
    if not input_text.strip():
        st.warning("Enter a sentence first.")
    else:
        biased = detect_biased_tokens(input_text, st.session_state.bias_dict)
        if not biased:
            st.success("No dictionary matches found.")
        else:
            st.error(f"Detected {len(biased)} biased token(s): {', '.join(biased)}")
            # highlight
            def highlight_md(txt, toks):
                out = txt
                toks_sorted = sorted(toks, key=len, reverse=True)
                for t in toks_sorted:
                    out = re.sub(r'(?i)\b' + re.escape(t) + r'\b', lambda m: f"[{m.group(0)}]", out)
                return out
            st.markdown("*Highlighted sentence:*")
            st.markdown(highlight_md(input_text, biased))
            # suggestions
            suggestions = {t: st.session_state.bias_dict.get(t.lower(), "") for t in biased}
            st.markdown("*Suggested replacements:*")
            st.table(pd.DataFrame(list(suggestions.items()), columns=['token','replacement']))
            inclusive = replace_biases_in_text(input_text, st.session_state.bias_dict)
            st.markdown("*Inclusive suggestion:*")
            st.info(inclusive)
        # model prediction
        if show_model and st.session_state.classifier is not None and st.session_state.vectorizer is not None:
            clean_in = clean_text(input_text)
            Xv_input = st.session_state.vectorizer.transform([clean_in])
            pred = st.session_state.classifier.predict(Xv_input)[0]
            prob = st.session_state.classifier.predict_proba(Xv_input)[0][1]
            if pred == 1:
                st.error(f"Model predicts: BIASED (prob={prob:.2f})")
            else:
                st.success(f"Model predicts: NEUTRAL (prob={prob:.2f})")
        elif show_model:
            st.info("No trained model in session. Train the classifier above to enable predictions.")

# ---------------------------
# Export results
# ---------------------------
st.subheader("Export results")
if st.button("Prepare CSV of dataset analysis"):
    out = df[['text','biased_words','bias_score']].copy()
    out['inclusive'] = out['text'].apply(lambda t: replace_biases_in_text(t, st.session_state.bias_dict))
    csv = out.to_csv(index=False)
    st.download_button("Download CSV", csv, file_name="gender_bias_analysis.csv")

st.caption("Add more occupations or mappings in the sidebar to increase coverage. For production-grade accuracy, create a labeled dataset and fine-tune a transformer model.")