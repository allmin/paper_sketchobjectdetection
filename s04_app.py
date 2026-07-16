import streamlit as st

# Define the pages with titles and icons
page_1 = st.Page("s04_apps/annotation.py", title="Annotation", icon="📊")
page_2 = st.Page("s04_apps/prediction.py", title="Prediction", icon="🤖")

# Define the navigation structure
nav = st.navigation(
    {
        "Tools": [page_1,page_2],
    }
)
# Run the selected page
nav.run()