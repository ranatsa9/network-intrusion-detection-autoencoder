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
    :root {--navy:#071044; --coral:#f34f61; --teal:#168f91; --cream:#c8ccd7;}
    .stApp {
        color:var(--navy);
        background:
          radial-gradient(circle at 10% 8%, rgba(33,170,172,.34), transparent 30%),
          radial-gradient(circle at 88% 18%, rgba(243,79,97,.25), transparent 27%),
          radial-gradient(circle at 52% 92%, rgba(54,72,139,.28), transparent 35%),
          linear-gradient(135deg,#aeb9c8 0%,#d3c7ca 52%,#aebfc2 100%);
        background-attachment:fixed;
    }
    [data-testid="stHeader"] {background:rgba(205,210,220,.72); backdrop-filter:blur(16px);}
    [data-testid="stAppViewContainer"] {background:transparent;}
    .block-container {max-width: 1180px; padding-top: 2rem; padding-bottom: 4rem;}
    .hero {
        padding: 2.5rem 2.4rem; border-radius: 26px; color: white;
        background:#10184c;
        border:1px solid rgba(255,255,255,.16); box-shadow:0 22px 55px rgba(7,16,68,.22); margin-bottom:1.7rem;
        position: relative; overflow: hidden;
    }
    .hero > *:not(.shape) {position:relative; z-index:2;}
    .shape {position:absolute; z-index:1; opacity:.85; pointer-events:none;}
    .shape-circle {width:170px; height:170px; border-radius:50%; background:#ff5364; right:65px; top:-62px; opacity:.75;}
    .shape-ring {width:105px; height:105px; border:15px solid #28c4c0; border-radius:50%; right:18px; bottom:-48px; opacity:.55;}
    .shape-dots {right:185px; bottom:15px; color:#ff5364; font-size:32px; letter-spacing:8px; opacity:.5;}
    .shape-circle {animation:float-one 7s ease-in-out infinite;}
    .shape-ring {animation:float-two 8s ease-in-out infinite;}
    .shape-dots {animation:pulse 4s ease-in-out infinite;}
    .hero h1 {font-size: 2.45rem; margin: 0 0 .55rem 0;}
    .hero p {font-size: 1.08rem; max-width: 760px; margin: 0; opacity: .92;}
    .hero .accent {color:#ff6b78;}
    .eyebrow {font-weight: 700; letter-spacing: .12em; font-size:.76rem; color:#50d4d0;}
    .process {display:grid; grid-template-columns:1fr 70px 1fr 70px 1fr; align-items:center; margin:1.6rem 0 1.9rem;}
    .process-card {position:relative; min-height:168px; padding:1.3rem 1.35rem; background:rgba(255,255,255,.48); border:1px solid rgba(255,255,255,.68); border-radius:20px; box-shadow:0 12px 30px rgba(7,16,68,.12); backdrop-filter:blur(18px); transition:transform .25s ease,border-color .25s ease,box-shadow .25s ease;}
    .process-card:hover {transform:translateY(-6px); border-color:rgba(255,255,255,.95); box-shadow:0 20px 42px rgba(7,16,68,.18);}
    .process-number {position:static; width:48px; height:48px; display:grid; place-items:center; margin-bottom:.9rem; border-radius:12px; background:var(--coral); color:#fff; font-weight:800; font-size:1.05rem; box-shadow:none;}
    .process-card.teal .process-number {background:var(--teal); box-shadow:none;}
    .process-card h4 {margin:.1rem 0 .45rem; color:var(--navy); font-size:1.08rem;}
    .process-card p {margin:0; color:#4e5877; line-height:1.55; font-size:.92rem;}
    .process-arrow {height:3px; background:linear-gradient(90deg,var(--teal),var(--coral),var(--teal)); background-size:200% 100%; position:relative; margin:0 14px; animation:flow 2.8s linear infinite;}
    .process-arrow:after {content:''; position:absolute; right:-1px; top:-6px; width:0; height:0; border-top:7px solid transparent; border-bottom:7px solid transparent; border-left:11px solid var(--coral);}
    .section-step {display:flex; align-items:center; gap:.85rem; margin:1.7rem 0 .8rem; padding:.75rem .95rem; background:rgba(255,255,255,.45); border:1px solid rgba(255,255,255,.7); border-left:5px solid var(--coral); border-radius:12px; box-shadow:0 8px 22px rgba(7,16,68,.08); backdrop-filter:blur(14px);}
    .section-step span {width:34px; height:34px; display:grid; place-items:center; border-radius:10px; background:var(--navy); color:#fff; font-size:.78rem; font-weight:800;}
    .section-step strong {color:var(--navy); font-size:1.05rem;}
    div.stButton > button {border-radius:14px; min-height:3.25rem; font-weight:750; background:linear-gradient(110deg,#f34f61,#e53e62); border:1px solid rgba(255,255,255,.45); box-shadow:0 10px 25px rgba(190,46,70,.22); transition:transform .2s ease,box-shadow .2s ease;}
    div.stButton > button:hover {background:linear-gradient(110deg,#ff6070,#ed4769); border-color:white; transform:translateY(-2px); box-shadow:0 15px 30px rgba(190,46,70,.30);}
    .demo-guide {background:white; border-left:5px solid var(--teal); border-radius:14px; padding:1rem 1.2rem;}
    .demo-guide strong {color:var(--coral);}
    .result-normal {background:#ecfdf5; border:1px solid #a7f3d0; padding:1.3rem; border-radius:18px;}
    .result-attack {background:#fff7ed; border:1px solid #fed7aa; padding:1.3rem; border-radius:18px;}
    .small-note {color:#475569; font-size:.9rem;}
    [data-testid="stMetric"] {background:rgba(255,255,255,.5); border:1px solid rgba(255,255,255,.72); padding:1rem; border-radius:14px; backdrop-filter:blur(14px);}
    [data-testid="stTabs"] [role="tablist"] {border-bottom:1px solid #cdbbbb;}
    [data-testid="stTabs"] button[aria-selected="true"] {color:var(--coral);}
    [data-testid="stExpander"] {background:rgba(255,255,255,.48); border:1px solid rgba(255,255,255,.75); border-radius:16px; box-shadow:0 10px 26px rgba(7,16,68,.09); backdrop-filter:blur(16px); overflow:hidden;}
    [data-testid="stExpander"] summary {font-weight:700; color:var(--navy); padding:.25rem .35rem;}
    [data-testid="stCheckbox"] {background:rgba(7,16,68,.82); border:1px solid rgba(255,255,255,.3); border-radius:15px; padding:.8rem 1rem; margin-top:.8rem;}
    [data-testid="stCheckbox"] label, [data-testid="stCheckbox"] p {color:white !important; font-weight:700;}
    .sample-box {display:grid; grid-template-columns:1fr 1fr; gap:12px; margin:.35rem 0 .65rem;}
    .sample {background:rgba(7,16,68,.88); color:#f8fafc; border-radius:13px; padding:1rem 1.1rem; line-height:1.65;}
    .sample.attack {background:rgba(123,36,55,.9);}
    .sample strong {display:block; color:#52d4d0; margin-bottom:.25rem;}
    .sample.attack strong {color:#ffc1c8;}
    @keyframes float-one {0%,100%{transform:translate(0,0)}50%{transform:translate(-12px,12px)}}
    @keyframes float-two {0%,100%{transform:translate(0,0) rotate(0)}50%{transform:translate(-10px,-9px) rotate(8deg)}}
    @keyframes pulse {0%,100%{opacity:.35;transform:scale(.96)}50%{opacity:.75;transform:scale(1.05)}}
    @keyframes flow {to{background-position:-200% 0}}
    @media(prefers-reduced-motion:reduce){.shape,.process-arrow,.process-card,div.stButton>button{animation:none!important;transition:none!important}}
    @media(max-width:800px){.process{grid-template-columns:1fr;gap:12px}.process-arrow{width:3px;height:28px;margin:auto}.process-arrow:after{right:-4px;top:18px;border-left:6px solid transparent;border-right:6px solid transparent;border-top:10px solid var(--coral);border-bottom:0}.process-card{min-height:auto}.sample-box{grid-template-columns:1fr}}
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

st.markdown(
    """
<section class="process">
  <div class="process-card"><div class="process-number">01</div><h4>Enter a connection</h4><p>Describe one device communicating with a service using a few network measurements.</p></div>
  <div class="process-arrow"></div>
  <div class="process-card teal"><div class="process-number">02</div><h4>The model reconstructs it</h4><p>The autoencoder compares the connection with the normal patterns it learned.</p></div>
  <div class="process-arrow"></div>
  <div class="process-card"><div class="process-number">03</div><h4>Receive a clear result</h4><p>A high reconstruction error creates a warning for investigation, not automatic proof of an attack.</p></div>
</section>
""",
    unsafe_allow_html=True,
)

st.write("")
detector_tab, learn_tab = st.tabs(["Try the detector", "Learn how it works"])

with detector_tab:
    st.subheader("Enter one network connection")
    st.write("Type the connection measurements below, then ask the model whether the pattern looks normal or unusual.")
    connection = empty_connection()

    with st.expander("Presentation example values", expanded=False):
        st.markdown(
            """
<div class="sample-box">
  <div class="sample"><strong>NORMAL WEB CONNECTION</strong>
  Protocol: tcp &nbsp;|&nbsp; Service: http &nbsp;|&nbsp; Flag: SF<br>
  Duration: 0 &nbsp;|&nbsp; Sent: 181 &nbsp;|&nbsp; Returned: 5450<br>
  Login: 1 &nbsp;|&nbsp; Destination count: 8 &nbsp;|&nbsp; Service count: 8
  </div>
  <div class="sample attack"><strong>SUSPICIOUS ICMP CONNECTION</strong>
  Protocol: icmp &nbsp;|&nbsp; Service: ecr_i &nbsp;|&nbsp; Flag: SF<br>
  Duration: 0 &nbsp;|&nbsp; Sent: 1032 &nbsp;|&nbsp; Returned: 0<br>
  Login: 0 &nbsp;|&nbsp; Destination count: 511 &nbsp;|&nbsp; Service count: 511
  </div>
</div>
<small>For the suspicious example, open the optional research measurements and set:
Same Service Rate = 1, Destination Host Count = 255, Destination Host Service Count = 255,
Destination Host Same Service Rate = 1, and Destination Host Same Source Port Rate = 1.</small>
""",
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-step"><span>01</span><strong>Choose the connection type</strong></div>', unsafe_allow_html=True)
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

    st.markdown('<div class="section-step"><span>02</span><strong>Enter the main measurements</strong></div>', unsafe_allow_html=True)
    common_columns = st.columns(3)
    common_fields = list(FRIENDLY_LABELS)
    for position, column in enumerate(common_fields):
        connection[column] = common_columns[position % 3].number_input(
            FRIENDLY_LABELS[column],
            min_value=0.0,
            value=float(connection[column]),
            help="This value is one of the measurements used by the trained model.",
        )

    show_advanced = st.toggle(
        "Advanced measurements",
        value=False,
        help="Turn this on only when you want to edit the full research feature set.",
    )
    if show_advanced:
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

    st.markdown('<div class="section-step"><span>03</span><strong>Run the anomaly detector</strong></div>', unsafe_allow_html=True)
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
