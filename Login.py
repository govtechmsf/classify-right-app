import streamlit as st
from streamlit_extras.switch_page_button import switch_page

# Constants
ICON = "images/dataset.png"

if 'user_type' not in st.session_state:
    st.session_state['user_type'] = ""
    

def display_header():
    """Displays the header with logo and welcome message."""
    col1, col2, col3 = st.columns([2, 5, 1])
    with col2:
        st.image(ICON, use_column_width=False,width=200)
    
    col1, col2, col3 = st.columns([1, 4, 1])
    with col2:
        st.markdown("# Welcome to Classify Right App!")

def display_main_content():
    """Displays the main content and interactive buttons."""
    col1, col2, col3 = st.columns([2, 4, 1])
    with col2:
        st.caption("Ask anything about Dataset Classification!")
    
    col1, col2, col3 = st.columns([2, 6, 5])
    with col2:
        if st.button("Officer"):
            st.session_state["user_type"]="OFFICER"
            switch_page("Classify Dataset")
    with col3:
        if st.button("Admin"):
            st.session_state["user_type"]="ADMIN"
            switch_page("Learning Repository")

            
def main():
    st.set_page_config(page_title="Welcome to Classify Right App")
    display_header()
    display_main_content()

if __name__ == "__main__":
    main()