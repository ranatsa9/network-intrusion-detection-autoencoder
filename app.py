"""Streamlit interface for the network intrusion autoencoder."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.inference import (
    categorical_options,
    expected_columns,
    load_artifacts,
    predict_connection,
    predict_connections,
)


st.set_page_config(
    page_title="Network Intrusion Detector",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ Network Intrusion Detection")
st.caption("Deep autoencoder trained on normal KDD Cup 1999 connections")

try:
    model, _, _, metadata = load_artifacts()
except Exception as exc:
    st.error(f"The saved model artifacts could not be loaded: {exc}")
    st.stop()

st.info(
    "The autoencoder reconstructs each connection. A reconstruction error "
    f"above **{metadata['threshold']:.6f}** is classified as an attack."
)

single_tab, batch_tab, about_tab = st.tabs(
    ["Single connection", "Batch CSV", "How it works"]
)

with single_tab:
    st.subheader("Enter one network connection")
    options = categorical_options()

    categorical_values = {}
    categorical_columns = st.columns(3)
    for position, column in enumerate(metadata["cat_cols"]):
        values = options[column]
        preferred = {"protocol_type": "tcp", "service": "http", "flag": "SF"}.get(column)
        default_index = values.index(preferred) if preferred in values else 0
        categorical_values[column] = categorical_columns[position].selectbox(
            column,
            values,
            index=default_index,
        )

    st.markdown("#### Numeric features")
    numeric_values = {}
    numeric_columns = st.columns(3)
    for position, column in enumerate(metadata["num_cols"]):
        numeric_values[column] = numeric_columns[position % 3].number_input(
            column,
            value=0.0,
            format="%.6f",
        )

    if st.button("Analyze connection", type="primary", use_container_width=True):
        connection = {**numeric_values, **categorical_values}
        result = predict_connection(connection)
        if result["prediction"] == "ATTACK":
            st.error("⚠️ ATTACK detected")
        else:
            st.success("✅ Connection appears NORMAL")

        metric_columns = st.columns(3)
        metric_columns[0].metric("Anomaly score", f"{result['anomaly_score']:.6f}")
        metric_columns[1].metric("Threshold", f"{result['threshold']:.6f}")
        metric_columns[2].metric("Score / threshold", f"{result['score_ratio']:.2f}×")

with batch_tab:
    st.subheader("Analyze a CSV file")
    st.write("Upload a CSV containing the required raw KDD connection columns.")
    with st.expander("Required columns"):
        st.code(", ".join(expected_columns()))

    uploaded = st.file_uploader("Connection CSV", type=["csv"])
    if uploaded is not None:
        try:
            uploaded_frame = pd.read_csv(uploaded)
            batch_results = predict_connections(uploaded_frame)
            combined = pd.concat(
                [uploaded_frame.reset_index(drop=True), batch_results.reset_index(drop=True)],
                axis=1,
            )
            attack_count = int((batch_results["prediction"] == "ATTACK").sum())
            normal_count = len(batch_results) - attack_count

            metric_columns = st.columns(3)
            metric_columns[0].metric("Connections", len(batch_results))
            metric_columns[1].metric("Attacks", attack_count)
            metric_columns[2].metric("Normal", normal_count)
            st.dataframe(combined, use_container_width=True)
            st.download_button(
                "Download predictions",
                combined.to_csv(index=False).encode("utf-8"),
                file_name="intrusion_predictions.csv",
                mime="text/csv",
            )
        except Exception as exc:
            st.error(str(exc))

with about_tab:
    st.subheader("How the detector works")
    st.markdown(
        """
1. The categorical fields are one-hot encoded.
2. Numeric fields are scaled using the saved training scaler.
3. The 72-feature connection is reconstructed by the autoencoder.
4. Mean squared reconstruction error becomes the anomaly score.
5. Scores above the validation-derived threshold are labeled **ATTACK**.

The model was trained on normal traffic only. This demonstration evaluates
connection records; it does not capture live network packets.
"""
    )
    st.caption(f"Model input: {model.input_shape[-1]} features · Bottleneck: {metadata['bottleneck']}")
