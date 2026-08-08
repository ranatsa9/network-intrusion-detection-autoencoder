"""Beginner-friendly Streamlit interface for the intrusion autoencoder."""

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
    page_title="Network Safety Checker",
    page_icon=":shield:",
    layout="wide",
)

st.markdown(
    """
<style>
    .block-container {max-width: 1180px; padding-top: 2rem; padding-bottom: 4rem;}
    .hero {
        padding: 2.2rem; border-radius: 24px; color: white;
        background: linear-gradient(135deg, #0f172a 0%, #1d4ed8 58%, #06b6d4 100%);
        box-shadow: 0 18px 50px rgba(30, 64, 175, .22); margin-bottom: 1.5rem;
    }
    .hero h1 {font-size: 2.45rem; margin: 0 0 .55rem 0;}
    .hero p {font-size: 1.08rem; max-width: 760px; margin: 0; opacity: .92;}
    .eyebrow {font-weight: 700; letter-spacing: .12em; font-size: .76rem; opacity: .8;}
    .info-card {
        border: 1px solid #dbeafe; background: #f8fbff; border-radius: 18px;
        padding: 1.15rem 1.25rem; height: 100%;
    }
    .info-card h4 {margin-top: 0; color: #1e3a8a;}
    .result-normal {background:#ecfdf5; border:1px solid #a7f3d0; padding:1.3rem; border-radius:18px;}
    .result-attack {background:#fff7ed; border:1px solid #fed7aa; padding:1.3rem; border-radius:18px;}
    .small-note {color:#475569; font-size:.9rem;}
    [data-testid="stMetric"] {background:white; border:1px solid #e2e8f0; padding:1rem; border-radius:16px;}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def artifacts():
    return load_artifacts()


try:
    model, _, _, metadata = artifacts()
except Exception as exc:
    st.error(f"The saved model files could not be loaded: {exc}")
    st.stop()


def empty_connection() -> dict[str, object]:
    row: dict[str, object] = {column: 0.0 for column in metadata["num_cols"]}
    row.update(protocol_type="tcp", service="http", flag="SF")
    return row


def normal_web_example() -> dict[str, object]:
    row = empty_connection()
    row.update(
        src_bytes=181,
        dst_bytes=5450,
        logged_in=1,
        count=8,
        srv_count=8,
        same_srv_rate=1,
        dst_host_count=9,
        dst_host_srv_count=9,
        dst_host_same_srv_rate=1,
        dst_host_same_src_port_rate=0.11,
    )
    return row


def suspicious_example() -> dict[str, object]:
    row = empty_connection()
    row.update(
        protocol_type="icmp",
        service="ecr_i",
        src_bytes=1032,
        count=511,
        srv_count=511,
        same_srv_rate=1,
        dst_host_count=255,
        dst_host_srv_count=255,
        dst_host_same_srv_rate=1,
        dst_host_same_src_port_rate=1,
    )
    return row


PRESETS = {
    "Normal web browsing": normal_web_example,
    "Suspicious high-volume traffic": suspicious_example,
    "Create a custom connection": empty_connection,
}


FRIENDLY_LABELS = {
    "duration": "Connection duration (seconds)",
    "src_bytes": "Data sent by the visitor (bytes)",
    "dst_bytes": "Data returned by the server (bytes)",
    "logged_in": "Successful login (1=yes, 0=no)",
    "count": "Recent connections to the same destination",
    "srv_count": "Recent connections to the same service",
}


st.markdown(
    """
<section class="hero">
  <div class="eyebrow">BEGINNER-FRIENDLY AI SECURITY DEMO</div>
  <h1>Network Safety Checker</h1>
  <p>This app asks one simple question: does a network connection look similar
  to normal activity, or is it unusual enough to deserve attention?</p>
</section>
""",
    unsafe_allow_html=True,
)

intro_columns = st.columns(3)
intro_columns[0].markdown(
    '<div class="info-card"><h4>1. A connection</h4><p>A short summary of one device communicating with another device or service.</p></div>',
    unsafe_allow_html=True,
)
intro_columns[1].markdown(
    '<div class="info-card"><h4>2. The AI compares</h4><p>The model learned normal traffic and measures how different a new connection looks.</p></div>',
    unsafe_allow_html=True,
)
intro_columns[2].markdown(
    '<div class="info-card"><h4>3. A warning</h4><p>An unusual result is a reason to investigate—not automatic proof of an attack.</p></div>',
    unsafe_allow_html=True,
)

st.write("")
detector_tab, batch_tab, learn_tab = st.tabs(
    ["Try the detector", "Analyze a CSV", "Learn how it works"]
)

with detector_tab:
    st.subheader("Try a guided example")
    st.write("Start with a prepared example, or choose custom to edit the technical details.")

    selected_preset = st.selectbox(
        "Choose a connection example",
        list(PRESETS),
        help="The first two choices use realistic example values from the KDD-style data.",
    )
    connection = PRESETS[selected_preset]()

    if selected_preset == "Normal web browsing":
        st.info("A visitor opens a web page, sends a small request, and receives a normal-sized response.")
    elif selected_preset == "Suspicious high-volume traffic":
        st.warning("One source rapidly repeats similar ICMP traffic—a pattern associated with flooding activity in this dataset.")
    else:
        st.info("Edit the common fields below. The remaining research fields are available under Advanced details.")

    st.markdown("#### Connection type")
    options = categorical_options()
    type_columns = st.columns(3)
    for position, column in enumerate(metadata["cat_cols"]):
        values = options[column]
        current = str(connection[column])
        connection[column] = type_columns[position].selectbox(
            column.replace("_", " ").title(),
            values,
            index=values.index(current) if current in values else 0,
            disabled=selected_preset != "Create a custom connection",
            help={
                "protocol_type": "The communication method, such as TCP, UDP, or ICMP.",
                "service": "The destination service, such as a website (HTTP) or email (SMTP).",
                "flag": "How the connection ended. SF usually means a normal completed connection.",
            }[column],
        )

    st.markdown("#### Easy-to-understand details")
    common_columns = st.columns(3)
    common_fields = list(FRIENDLY_LABELS)
    for position, column in enumerate(common_fields):
        connection[column] = common_columns[position % 3].number_input(
            FRIENDLY_LABELS[column],
            min_value=0.0,
            value=float(connection[column]),
            disabled=selected_preset != "Create a custom connection",
            help="This value is one of the measurements used by the trained model.",
        )

    with st.expander("Advanced details (research features)"):
        st.caption("Most beginners can leave these values unchanged.")
        advanced_fields = [field for field in metadata["num_cols"] if field not in common_fields]
        advanced_columns = st.columns(3)
        for position, column in enumerate(advanced_fields):
            connection[column] = advanced_columns[position % 3].number_input(
                column.replace("_", " ").title(),
                min_value=0.0,
                value=float(connection[column]),
                disabled=selected_preset != "Create a custom connection",
                key=f"advanced_{column}",
            )

    if st.button("Check this connection", type="primary", use_container_width=True):
        result = predict_connection(connection)
        ratio = result["score_ratio"]
        if result["prediction"] == "ATTACK":
            st.markdown(
                """<div class="result-attack"><h3>Unusual connection — investigate</h3>
                <p>The model could not reproduce this pattern accurately. It looks different
                from the normal traffic used during training. This is a warning, not proof
                that someone attacked the network.</p></div>""",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """<div class="result-normal"><h3>Connection looks normal</h3>
                <p>The model reconstructed this pattern accurately, which means it resembles
                the normal traffic learned during training.</p></div>""",
                unsafe_allow_html=True,
            )

        st.write("")
        result_columns = st.columns(3)
        result_columns[0].metric("Model decision", result["prediction"])
        result_columns[1].metric("Difference score", f"{result['anomaly_score']:.6f}")
        result_columns[2].metric("Warning threshold", f"{result['threshold']:.6f}")
        st.progress(min(float(ratio) / 4.0, 1.0), text=f"The score is {ratio:.2f}× the warning threshold")
        st.caption("Higher difference score = the connection was harder for the autoencoder to reconstruct.")

with batch_tab:
    st.subheader("Check many connections from a CSV file")
    st.write("Each row should describe one network connection. Download the example first to see the expected format.")

    example_frame = pd.DataFrame([normal_web_example(), suspicious_example()])
    st.download_button(
        "Download example CSV",
        example_frame.to_csv(index=False).encode("utf-8"),
        file_name="network_connections_example.csv",
        mime="text/csv",
    )
    uploaded = st.file_uploader("Upload connection CSV", type=["csv"])
    with st.expander("Show required columns"):
        st.code(", ".join(expected_columns()))

    if uploaded is not None:
        try:
            uploaded_frame = pd.read_csv(uploaded)
            batch_results = predict_connections(uploaded_frame)
            combined = pd.concat(
                [uploaded_frame.reset_index(drop=True), batch_results.reset_index(drop=True)],
                axis=1,
            )
            attacks = int((batch_results["prediction"] == "ATTACK").sum())
            summary_columns = st.columns(3)
            summary_columns[0].metric("Connections checked", len(batch_results))
            summary_columns[1].metric("Needs investigation", attacks)
            summary_columns[2].metric("Looks normal", len(batch_results) - attacks)
            st.dataframe(
                combined[["prediction", "anomaly_score", "threshold", "score_ratio"]],
                use_container_width=True,
            )
            st.download_button(
                "Download full results",
                combined.to_csv(index=False).encode("utf-8"),
                file_name="intrusion_predictions.csv",
                mime="text/csv",
            )
        except Exception as exc:
            st.error(f"The CSV could not be analyzed: {exc}")

with learn_tab:
    st.subheader("What did we build?")
    st.markdown(
        """
### The dataset
The project uses the **KDD Cup 1999** network-intrusion dataset. Each row is a
summary of one network connection, with measurements such as connection length,
data volume, service type, and recent connection frequency.

### The model: a deep autoencoder
An autoencoder is a neural network trained to copy its input. Ours was trained
only on **normal connections**:

`72 input features → 64 → 32 → 8-number bottleneck → 32 → 64 → 72 reconstructed features`

The small bottleneck forces the network to learn a compressed description of
normal behavior. Normal connections are usually reconstructed well. Unfamiliar
connections produce a larger reconstruction error.

### How a decision is made
1. Text categories such as protocol and service are converted into numbers.
2. All features are scaled into the range used during training.
3. The autoencoder reconstructs the 72-number connection.
4. Mean squared error measures the difference between input and reconstruction.
5. Error above **0.015422** is labeled `ATTACK`; otherwise it is `NORMAL`.

### Why the threshold exists
The threshold is the 99th percentile of reconstruction errors on normal
validation traffic. In simple terms, the system accepts that about 1% of normal
validation connections may still look unusual.
"""
    )

    explanation_columns = st.columns(2)
    with explanation_columns[0]:
        st.markdown(
            """
### What `NORMAL` means
- The connection resembles learned normal behavior.
- The model reconstructed it with low error.
- It does **not** guarantee that the connection is safe.
"""
        )
    with explanation_columns[1]:
        st.markdown(
            """
### What `ATTACK` means
- The connection looks unusual to the model.
- A security analyst should investigate it.
- It does **not** prove malicious intent by itself.
"""
        )

    st.warning(
        "Important limitation: KDD Cup 1999 represents traffic from 1998. "
        "This application is an educational demonstration, not a modern live security product."
    )
    with st.expander("Mini glossary"):
        st.markdown(
            """
- **Network connection:** communication between two computers or services.
- **Protocol:** the communication rules used, such as TCP, UDP, or ICMP.
- **Service:** the destination function, such as HTTP for websites.
- **Feature:** one measurement supplied to the model.
- **Autoencoder:** a neural network that learns to reconstruct its input.
- **Anomaly:** something sufficiently different from the learned normal pattern.
- **Threshold:** the score boundary between normal and unusual.
- **False alarm:** normal activity incorrectly flagged as unusual.
"""
        )

st.divider()
st.caption(
    f"Educational project · Model input: {model.input_shape[-1]} encoded features · "
    f"Bottleneck: {metadata['bottleneck']} · Threshold: {metadata['threshold']:.6f}"
)
