import streamlit as st
import pandas as pd
import os
import random

random.seed(123)


def load_data():
    if os.path.exists('03_evaluation/similarity_cache_sampled_annotated.pickle'):
        df_sampled = pd.read_pickle('03_evaluation/similarity_cache_sampled_annotated.pickle')
    else:
        df_full = pd.read_excel('03_evaluation/similarity_cache.xlsx')
        df_full = df_full[df_full.similarity > 0]
        df_sampled = df_full.sample(500, random_state=123)
        df_sampled['annotation'] = False
        print("creating Default annotations")
        print(df_sampled['annotation'].unique())
        df_sampled.to_pickle('03_evaluation/similarity_cache_sampled_annotated.pickle')
    return df_sampled

def save_data(df):
    df.to_pickle('03_evaluation/similarity_cache_sampled_annotated.pickle')


def show():
    st.title("App 1: Annotate for similarity")
    st.write("Please say if the two pieces of text refer to the same or similar entity.")

    if 'df' not in st.session_state:
        st.session_state.df = load_data()
    
    df = st.session_state.df

    # Track current index
    if "index" not in st.session_state:
        st.session_state.index = 0

    ind = st.session_state.index

    # Bounds check
    if ind < 0:
        st.session_state.index = 0
        ind = 0
    if ind >= len(df):
        st.success("🎉 All items have been annotated")
        return

    row = df.iloc[ind]
    text1 = row["text1"]
    text2 = row["text2"]
    booleanize = {"Similar":True, "Not similar":False}
    similar_similarity = df[df.annotation == True].similarity
    dissimilar_similarity = df[df.annotation == False].similarity


    st.subheader(f"Pair {ind + 1} of {len(df)}")



    st.write("### Text 1")
    st.info(text1)

    st.write("### Text 2")
    st.info(text2)

    # Pre-fill existing annotation if present
    current_annotation = row["annotation"]
    st.write(current_annotation)

    choice = st.radio(
        """Are these text similar? 
        Tips: 
        - Moronyms, Hyponyms, Co-hyponyms of similar appearances => Similar |
        - Too broad Hyponyms => Dissimilar
        - Objects""",
        ["Similar", "Not similar"],
        index=0 if current_annotation == True else 1 if current_annotation == False else 0,
        key=f"choice_{ind}"
    )
    booleanized_choice = booleanize[choice]
    st.write(booleanized_choice)

    # Navigation buttons
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("⬅️ Previous"):
            save_data(df)
            if st.session_state.index > 0:
                st.session_state.index -= 1
            else:
                st.session_state.index = len(df) - 1
            st.rerun()

    with col2:
        if st.button("💾 Save"):
            df.at[df.index[ind], "annotation"] = booleanized_choice
            save_data(df)
            st.success("Saved!")

    with col3:
        if st.button("➡️ Next"):
            df.at[df.index[ind], "annotation"] = booleanized_choice
            save_data(df)
            if (st.session_state.index + 1) < len(df):
                st.session_state.index += 1
            else:
                st.session_state.index = 0
            st.rerun()
    
    col1, col2, col3 = st.columns(3)
    with col2:
        next_page = st.text_input(label="Go to:")
        if next_page:
            st.session_state.index = int(next_page)
            st.rerun()
    


    # Similar stats
    sim_stats = {
        "Count": len(similar_similarity),
        "Min": similar_similarity.min(),
        "Mean": similar_similarity.mean(),
        "Median": similar_similarity.median(),
        "P25": similar_similarity.quantile(0.25),
        "P75": similar_similarity.quantile(0.75),
        "Max": similar_similarity.max(),
    }

    # Dissimilar stats
    dissim_stats = {
        "Count": len(dissimilar_similarity),
        "Min": dissimilar_similarity.min(),
        "Mean": dissimilar_similarity.mean(),
        "Median": dissimilar_similarity.median(),
        "P25": dissimilar_similarity.quantile(0.25),
        "P75": dissimilar_similarity.quantile(0.75),
        "Max": dissimilar_similarity.max(),
    }


    st.subheader("Similar Stats")
    cols = st.columns(len(sim_stats))
    for col, (label, value) in zip(cols, sim_stats.items()):
        col.metric(label, f"{value:0.3f}")

    st.subheader("Dissimilar Stats")
    cols = st.columns(len(dissim_stats))
    for col, (label, value) in zip(cols, dissim_stats.items()):
        col.metric(label, f"{value:0.2f}")

    similar_df = df[df.annotation==True][['text1','text2','similarity']]
    st.write("Similar Items:", len(similar_df))
    st.write(similar_df)


if __name__ == "__main__":
    show()
