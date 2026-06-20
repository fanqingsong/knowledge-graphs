import streamlit as st

pg = st.navigation(
    [
        st.Page("streamlit_pages/home.py", title="Home", icon="🏠"),
        st.Page("streamlit_pages/upload.py", title="Upload File", icon="🗳️"),
        st.Page("streamlit_pages/chat.py", title="Chat with Graph" , icon="🦜"),
    ]
)

pg.run()
