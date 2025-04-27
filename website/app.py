import streamlit as st
from transformers import pipeline

# Streamlit Page Configuration
st.set_page_config(
    page_title="Deteksi Sarkasme",
    layout="wide"
)

st.markdown("""
        <style>
            * {
                text-align: justify;
            }
               .block-container {
                    padding-top: 1.5rem;
                    padding-bottom: 0rem;
                    padding-left: 5rem;
                    padding-right: 5rem;
                }
            .stAppDeployButton {
                display:none !important;
            }
        </style>
        <style>
            section[data-testid="stSidebar"] {
                width: 3em !important; # Set the width to your desired value
            }
            div[data-testid="stSidebarCollapseButton"] {
                display:none !important;
            }
            a[data-testid="stSidebarNavLink"] {
                padding-left: 15px;
            }
            li:first-of-type a[data-testid="stSidebarNavLink"] {
                padding-left: 5px;
            }
            span[data-testid="stHeaderActionElements"] {
                display:none !important;
            }
            .stDeployButton {
                    visibility: hidden;
                }
            
        </style>
        <style>
            .reportview-container {
                margin-top: -2em;
            }
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            #stDecoration {display:none;}
        </style>
        """, unsafe_allow_html=True)

# Load models only once using session state
if "twitter_model" not in st.session_state:
    st.session_state.twitter_model = pipeline(
        "text-classification",
        model="enoubi/XLM-RoBERTa-Twitter-Indonesian-Sarcastic-Few-Shot"
    )

if "reddit_model" not in st.session_state:
    st.session_state.reddit_model = pipeline(
        "text-classification",
        model="enoubi/XLM-RoBERTa-Reddit-Indonesian-Sarcastic-Few-Shot"
    )

# Initialize session state for inputs
st.session_state.setdefault("twitter_input", "")
st.session_state.setdefault("reddit_input", "")
st.session_state.setdefault("last", "")

# Define available pages
pages = {
    "": [st.Page("home.py", title="Home", icon="")],
    "Deteksi Sarkasme": [
        st.Page("twitter.py", title="Twitter", icon=""),
        st.Page("reddit.py", title="Reddit", icon="")
    ]
}

# Navigation initialization
pg = st.navigation(pages, expanded=True)

pg.run()
