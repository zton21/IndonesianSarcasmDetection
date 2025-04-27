import streamlit as st
from PIL import Image
import pandas as pd
import altair as alt
from streamlit_javascript import st_javascript

def run(name: str, title: str):
    dark_mode = st_javascript("""function darkMode(i){return (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches)}(1)""")

    icon = Image.open(f"website/assets/{name}.png")
    col1, col2 = st.columns([1, 11], vertical_alignment="center")
    with col1:
        st.image(icon)
    with col2:
        st.title(title)

    # Input box
    def _sync_input():
        st.session_state[name + "_input"] = st.session_state[name + "_text"]

    # reset on change page
    if st.session_state["last"] != name:
        st.session_state[name + "_text"] = ""
        st.session_state["last"] = name
        _sync_input()

    st.text_area(
        "Masukkan teks untuk dianalisis:",
        key=name + "_text",
        value=st.session_state.get(name + "_input", ""),
        on_change=_sync_input,
        height=120,
    )

    # Buttons
    col1, col2 = st.columns([1, 1])
    with col1:
        analyze = st.button("Analisis", key=name + "_button", use_container_width=True)
    with col2:
        clear = st.button("Hapus", key=name + "_clear", use_container_width=True)

    if clear:
        st.session_state[name + "_input"] = ""
        st.rerun()

    # Analysis
    if analyze:
        text = st.session_state[name + "_input"].strip()
        if not text:
            st.error("Teks tidak boleh kosong. Mohon isi teks terlebih dahulu.", icon=':material/cancel:')
        else:
            with st.spinner('Sedang menganalisis teks, harap tunggu...', show_time=True):
                res = st.session_state[name + "_model"](text)[0]
                label_map = {"LABEL_0": "Bukan Sarkasme", "LABEL_1": "Sarkasme"}
                pred = label_map.get(res["label"], res["label"])
                conf_pct = res["score"] # percent

                text_color = "white" if dark_mode else "black"
                alt_color = "black" if dark_mode else "white"

                # Calculate segments
                sarcastic_pct = conf_pct if pred == "Sarkasme" else 1 - conf_pct
                nonsarcastic_pct = 1 - sarcastic_pct
                df = pd.DataFrame({
                    "Type": ["Prediction", "Prediction"],
                    "Category": ["Bukan Sarkasme", "Sarkasme"],
                    "Value": [nonsarcastic_pct, sarcastic_pct],
                    "Value-inv": [sarcastic_pct, nonsarcastic_pct],
                    "X-Text": [0.03, 0.97],
                    "X-Type": ["X", "X"]
                })
                # st.write(df)
                st.markdown("#### Hasil Prediksi")
                st.markdown(f"Teks tersebut diprediksi model sebagai **{pred}**")

                domain = ['Sarkasme', 'Bukan Sarkasme']
                range_ = ['#bacedf', '#39ace7']
                # range_ = ['#1f77b4', '#aec7e8']

                chart = alt.Chart(df).mark_bar(
                    height=30, 
                    cornerRadiusTopLeft=15,
                    cornerRadiusTopRight=15,
                    cornerRadiusBottomLeft=15,
                    cornerRadiusBottomRight=15,
                    tooltip=None
                ).encode(
                    x=alt.X('Value-inv', axis=None, sort=None),
                    y=alt.Y('Type', axis=None),
                    color=alt.Color("Category", legend=None, scale=alt.Scale(domain=domain, range=range_))
                )
                
                chart2 = alt.Chart(df).mark_text(height=30, tooltip=None).encode(
                    x=alt.X("X-Text", axis=None),
                    y=alt.Y('Type', axis=None),
                    text=alt.Text('Value-inv', format='.2%'),
                    color=alt.value(alt_color)
                ) # Bottom category labels

                chart3 = alt.Chart(pd.DataFrame({
                    "Type": ["Prediction"],
                    "Category": ["Sarkasme"],
                    "X-Text": [0]
                })).mark_text(height=30, dy = 33, align="left", tooltip=None).encode(
                    x=alt.X("X-Text", axis=None),
                    y=alt.Y('Type', axis=None),
                    text=alt.Text('Category'),
                    color=alt.value(text_color)
                ) + alt.Chart(pd.DataFrame({
                    "Type": ["Prediction"],
                    "Category": ["Bukan Sarkasme"],
                    "X-Text": [1]
                })).mark_text(height=30, dy = 33, align="right").encode(
                    x=alt.X("X-Text", axis=None),
                    y=alt.Y('Type', axis=None),
                    text=alt.Text('Category'),
                    color=alt.value(text_color)
                )
                st.altair_chart(chart + chart2 + chart3)
