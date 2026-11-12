import streamlit as st

# Define the pages with titles and icons
page_1 = st.Page("apps/annotation.py", title="Annotation", icon="📊")
page_2 = st.Page("apps/prediction.py", title="Prediction", icon="🤖")
page_3 = st.Page("apps/calibration_annotation.py", title="Calibration annotation", icon="🏷️")

# Define the navigation structure
nav = st.navigation(
    {
        "Tools": [page_1, page_2],
        "Threshold calibration (s03)": [page_3],
    }
)
# Run the selected page
nav.run()