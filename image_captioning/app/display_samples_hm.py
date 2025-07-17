import streamlit as st
import pandas as pd
import os
from PIL import Image
from streamlit_gsheets import GSheetsConnection
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from streamlit_extras.stylable_container import stylable_container
from dotenv import load_dotenv

st.set_page_config(page_title="Qualitative Analysis", layout="wide")

st.title("Caption results")

st.write(
    """<style>
    [data-testid="stHorizontalBlock"]:has(img):first-of-type {
        align-items: center;
    }
    [data-testid="stHorizontalBlock"]:has(img):first-of-type {
        align-items: center;
    }
    h1 { style="text-align"}
    /* Media query for widths greater than 1080px */
    @media (min-width: 1080px) {
        div[data-testid="column"]:nth-of-type(1),
        div[data-testid="column"]:nth-of-type(2),
        div[data-testid="column"]:nth-of-type(3) {
            max-width: 1080px;
            width: 100%;
            display: flex;
            justify-content: center;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


load_dotenv()

HM_PATH_PROCESSED = os.getenv("HM_PROCESSED")
HM_TEST_FILE = os.path.join(HM_PATH_PROCESSED, f"hm_test.csv")

hm_data = pd.read_csv("HM_TEST_FILE", index_col=False)
scores = pd.read_csv("./metrics_results_50.csv")

hm_results = scores[scores["dataset"] == "hm"]
hm_ids = list(hm_results["image_id"].unique())

if "idx" not in st.session_state:
    st.session_state.idx = 0

if "image_id" not in st.session_state:
    st.session_state.image_id = hm_ids[st.session_state.idx]

if "dataset" not in st.session_state:
    st.session_state.dataset = "hm"

conn = st.connection("gsheets", type=GSheetsConnection)
notes = conn.read(worksheet="data", usecols=[0, 1, 2])

def get_image_row():
    row = hm_data[hm_data["img_path"].str.endswith(st.session_state.image_id)].iloc[0]
    return row


def get_metrics_row():
    row = hm_results[hm_results["image_id"] == st.session_state.image_id]
    return row


def next_id():
    id = (st.session_state.idx + 1) % len(hm_ids)
    st.session_state.idx = id
    st.session_state.image_id = hm_ids[id]


def previous_id():
    id = (st.session_state.idx - 1) % len(hm_ids)
    st.session_state.idx = id
    st.session_state.image_id = hm_ids[id]


def update_sheet(df):
    df = df.dropna()
    df = conn.update(worksheet="data", data=df)
    st.cache_data.clear()
    st.experimental_rerun()


def img_col(col):
    with col:
        image_row = get_image_row()
        image = Image.open(image_row["img_path"])
        st.image(image, caption=st.session_state.image_id, width=300)
        st.write(f"**Sample number:** {st.session_state.idx + 1}")
        image_row = image_row.T
        st.dataframe(image_row)


def show_metrics(col):
    with col:
        metrics_row = get_metrics_row()
        show_cols_metrics = [
            "model",
            "pred_caption",
            "Bleu_1",
            "Bleu_2",
            "Bleu_3",
            "Bleu_4",
            "METEOR",
            "ROUGE_L",
            "SPICE",
        ]
        gt_caption = metrics_row["gt_caption"].iloc[0]
        st.write(f"**Ground truth caption:** {gt_caption}")
        st.dataframe(metrics_row[show_cols_metrics], hide_index=True)


def show_attrs(col):
    with col:
        show_cols_attrs = ["model", "true_positives", "attrs_selected", "precision", "recall"]
        values = scores[scores["image_id"] == st.session_state.image_id]
        gt_attrs = set(values['gt_attrs'].iloc[0].split(" "))
        st.write(f"**Ground truth attributes ({len(gt_attrs)} attrs):** {gt_attrs}")
        st.dataframe(values[show_cols_attrs], hide_index=True)


def show_buttons(notes):
    col11, col21, col31, col41 = st.columns([0.55, 0.15, 0.1, 0.20])
    with col21:
        if st.session_state.idx > 0:
            st.button("Previous", on_click=previous_id)
    with col31:
        if st.session_state.idx < len(hm_ids) - 1:
            st.button("Next", on_click=next_id)
    with col41:
        if st.button("Update"):
            update_sheet(notes)

def show_class(col):
    with col:
        show_cols_cats = ["model", "pred_cat", "gt_cat"]
        values = scores[scores["image_id"] == st.session_state.image_id]
        st.dataframe(values[show_cols_cats], hide_index=True, height=450)


left, right = st.columns([0.30, 0.70])


def display_info(notes):
    img_col(left)
    show_class(left)
    show_metrics(right)
    show_attrs(right)

    with right:
        st.write(f"Session Image Id: {st.session_state.image_id}")

        if st.session_state.image_id in notes["id"].values:
            note = notes[notes["id"] == st.session_state.image_id]["note"].iloc[0]
        else:
            note = ""

        user_input = st.text_area("Notes", note)

        if user_input:
            # Check if the id is already in the DataFrame
            if st.session_state.image_id in notes["id"].values:
                # Update the note for the existing id
                notes.loc[notes["id"] == st.session_state.image_id, "note"] = user_input
            else:
                # Add a new entry with the id and dataset
                new_entry = pd.DataFrame(
                    {
                        "id": [st.session_state.image_id],
                        "note": [user_input],
                        "dataset": [st.session_state.dataset],
                    }
                )
                notes = pd.concat([notes, new_entry], ignore_index=True)
        show_buttons(notes)
        st.dataframe(notes, width=3000, hide_index=True)


def main():
    if st.session_state.image_id:
        try:
            display_info(notes)
        except IndexError:
            st.error("Image not found for the provided id.")


if __name__ == "__main__":
    main()
