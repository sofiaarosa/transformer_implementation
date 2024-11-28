import streamlit as st

app_page = st.Page("pages/application.py", title = "Tradutor PT-EN", icon=":material/translate:")
transformer_page = st.Page("pages/transformer_model.py", title = "O Modelo Transformer", icon=":material/model_training:")
pages = [app_page, transformer_page]

pg = st.navigation(pages=pages)
pg.run()