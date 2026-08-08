"""Beginner-friendly Streamlit interface for the intrusion autoencoder."""

from __future__ import annotations

import streamlit as st

from src.inference import (
    categorical_options,
    load_artifacts,
    predict_connection,
)


st.set_page_config(
    page_title="Network Safety Checker",
    page_icon=":shield:",
    layout="wide",
)

st.markdown(
    """
<style>
    :root {--navy:#071044; --coral:#ff5364; --teal:#1ea7a8; --cream:#fff8f5;}
    .stApp {background: var(--cream); color: var(--navy);}
    .block-container {max-width: 1180px; padding-top: 2rem; padding-bottom: 4rem;}
    .hero {
        padding: 2.2rem; border-radius: 24px; color: var(--navy);
        background: linear-gradient(135deg, #fff 0%, #fff1ee 68%, #dff7f5 100%);
        border: 1px solid #ffd6d9; box-shadow: 0 18px 50px rgba(7, 16, 68, .08); margin-bottom: 1.5rem;
        position: relative; overflow: hidden;
    }
    .hero > *:not(.shape) {position:relative; z-index:2;}
    .shape {position:absolute; z-index:1; opacity:.85; pointer-events:none;}
    .shape-circle {width:150px; height:150px; border-radius:50%; background:#ffd9dd; right:65px; top:-45px;}
    .shape-ring {width:90px; height:90px; border:14px solid #28aaa9; border-radius:50%; right:20px; bottom:-38px; opacity:.35;}
    .shape-dots {right:185px; bottom:15px; color:#ff5364; font-size:32px; letter-spacing:8px; opacity:.5;}
    .hero h1 {font-size: 2.45rem; margin: 0 0 .55rem 0;}
    .hero p {font-size: 1.08rem; max-width: 760px; margin: 0; opacity: .92;}
    .hero .accent {color:var(--coral);}
    .eyebrow {font-weight: 700; letter-spacing: .12em; font-size: .76rem; color:var(--teal);}
    .info-card {
        border: 1px solid #ffdfe1; background: white; border-radius: 18px;
        padding: 1.15rem 1.25rem; height: 100%;
    }
    .info-card h4 {margin-top: 0; color: var(--navy);}
    .demo-guide {background:white; border-left:5px solid var(--teal); border-radius:14px; padding:1rem 1.2rem;}
    .demo-guide strong {color:var(--coral);}
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
  <div class="shape shape-circle"></div>
  <div class="shape shape-ring"></div>
  <div class="shape shape-dots">•••</div>
  <div class="eyebrow">ANOMALY DETECTION SYSTEM</div>
  <h1>Learning <span class="accent">normal</span> to detect the abnormal.</h1>
  <p>A beginner-friendly demonstration of network intrusion detection with a
  deep autoencoder. Enter a connection, let the model reconstruct it, and compare
  its reconstruction error with the warning threshold.</p>
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
detector_tab, learn_tab = st.tabs(["Try the detector", "Learn how it works"])

with detector_tab:
    st.subheader("Enter one network connection")
    st.write("Type the connection measurements below, then ask the model whether the pattern looks normal or unusual.")
    connection = empty_connection()

    with st.expander("Need values for the presentation? Open this example"):
        st.code(
            "Normal example\nProtocol: tcp | Service: http | Flag: SF\n"
            "Duration: 0 | Bytes sent: 181 | Bytes returned: 5450\n"
            "Successful login: 1 | Recent destination connections: 8 | "
            "Recent service connections: 8"
        )

    st.markdown("#### Step 1 - Describe the connection")
    options = categorical_options()
    type_columns = st.columns(3)
    for position, column in enumerate(metadata["cat_cols"]):
        values = options[column]
        current = str(connection[column])
        connection[column] = type_columns[position].selectbox(
            column.replace("_", " ").title(),
            values,
            index=values.index(current) if current in values else 0,
            help={
                "protocol_type": "The communication method, such as TCP, UDP, or ICMP.",
                "service": "The destination service, such as a website (HTTP) or email (SMTP).",
                "flag": "How the connection ended. SF usually means a normal completed connection.",
            }[column],
        )

    st.markdown("#### Step 2 - Review the main measurements")
    common_columns = st.columns(3)
    common_fields = list(FRIENDLY_LABELS)
    for position, column in enumerate(common_fields):
        connection[column] = common_columns[position % 3].number_input(
            FRIENDLY_LABELS[column],
            min_value=0.0,
            value=float(connection[column]),
            help="This value is one of the measurements used by the trained model.",
        )

    with st.expander("Optional: see all research measurements"):
        st.caption("Most beginners can leave these values unchanged.")
        advanced_fields = [field for field in metadata["num_cols"] if field not in common_fields]
        advanced_columns = st.columns(3)
        for position, column in enumerate(advanced_fields):
            connection[column] = advanced_columns[position % 3].number_input(
                column.replace("_", " ").title(),
                min_value=0.0,
                value=float(connection[column]),
                key=f"advanced_{column}",
            )

    st.markdown("#### Step 3 - Ask the model")
    if st.button("Analyze this connection", type="primary", use_container_width=True):
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
