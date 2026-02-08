"""Streamlit UI for NG12 Cancer Risk Assessor."""

import os
import uuid

import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="NG12 Cancer Risk Assessor", layout="wide")
st.title("NG12 Cancer Risk Assessor")
st.markdown("Clinical Decision Support powered by NG12 guidelines and Google Gemini")
st.divider()

try:
    requests.get(f"{API_URL}/health", timeout=5)
except Exception:
    st.error(f"Could not connect to API at {API_URL}. Is the FastAPI server running?")
    st.stop()

tab_assess, tab_chat = st.tabs(["Risk Assessment", "NG12 Chat"])


# -- Risk Assessment Tab --

with tab_assess:
    try:
        response = requests.get(f"{API_URL}/patients", timeout=5)
        patient_ids = response.json()["patient_ids"]
    except Exception:
        st.error("Could not fetch patient list.")
        patient_ids = []

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Select Patient")
        selected_id = st.selectbox("Patient ID", patient_ids)

        if selected_id:
            try:
                patient_resp = requests.get(f"{API_URL}/patients/{selected_id}", timeout=5)
                patient = patient_resp.json()
                st.markdown(f"**Name:** {patient['name']}")
                st.markdown(f"**Age:** {patient['age']} | **Gender:** {patient['gender']}")
                st.markdown(f"**Smoking:** {patient['smoking_history']}")
                st.markdown(f"**Symptoms:** {', '.join(patient['symptoms'])}")
                st.markdown(f"**Duration:** {patient['symptom_duration_days']} days")
            except Exception:
                st.warning("Could not fetch patient details.")

        assess_button = st.button("Assess Cancer Risk", type="primary", use_container_width=True)

    with col2:
        st.subheader("Risk Assessment")

        if assess_button and selected_id:
            with st.spinner("Analyzing patient data against NG12 guidelines..."):
                try:
                    resp = requests.post(
                        f"{API_URL}/assess",
                        json={"patient_id": selected_id},
                        timeout=60,
                    )

                    if resp.status_code == 200:
                        result = resp.json()
                        risk = result["risk_level"]

                        if "Urgent Referral" in risk:
                            st.error(f"**Risk Level: {risk}**")
                        elif "Urgent Investigation" in risk:
                            st.warning(f"**Risk Level: {risk}**")
                        elif "Non-Urgent" in risk:
                            st.info(f"**Risk Level: {risk}**")
                        else:
                            st.success(f"**Risk Level: {risk}**")

                        st.markdown("### Clinical Assessment")
                        st.markdown(result["assessment"])

                        if result.get("citations"):
                            st.markdown("### NG12 Guideline Citations")
                            for i, citation in enumerate(result["citations"], 1):
                                with st.expander(f"Citation {i}"):
                                    st.markdown(f"**Source:** {citation['source']}")
                                    st.markdown(f"**Page:** {citation['page']}")
                                    st.markdown("**Excerpt:**")
                                    st.text(citation["excerpt"])
                    else:
                        st.error(f"API error: {resp.status_code} - {resp.text}")

                except requests.exceptions.Timeout:
                    st.error("Request timed out. The assessment may take longer for complex cases.")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        elif not assess_button:
            st.info("Select a patient and click 'Assess Cancer Risk' to begin.")


# -- NG12 Chat Tab --

with tab_chat:
    st.subheader("Chat with NG12 Guidelines")
    st.markdown(
        "Ask questions about the NICE NG12 cancer referral guidelines. "
        "Answers are grounded in the guideline text with citations."
    )

    if "chat_session_id" not in st.session_state:
        st.session_state.chat_session_id = str(uuid.uuid4())
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    if st.button("New Chat", key="new_chat"):
        st.session_state.chat_session_id = str(uuid.uuid4())
        st.session_state.chat_messages = []
        st.rerun()

    chat_container = st.container()
    user_input = st.chat_input("Ask about NG12 guidelines...")

    with chat_container:
        for msg in st.session_state.chat_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg["role"] == "assistant" and msg.get("citations"):
                    for i, cit in enumerate(msg["citations"], 1):
                        with st.expander(f"[NG12 p.{cit['page']}] {cit['chunk_id']}"):
                            st.text(cit["excerpt"])

        if user_input:
            with st.chat_message("user"):
                st.markdown(user_input)

            with st.chat_message("assistant"):
                with st.spinner("Searching NG12 guidelines..."):
                    try:
                        resp = requests.post(
                            f"{API_URL}/chat",
                            json={
                                "session_id": st.session_state.chat_session_id,
                                "message": user_input,
                                "top_k": 5,
                            },
                            timeout=60,
                        )

                        if resp.status_code == 200:
                            data = resp.json()
                            answer = data["answer"]
                            citations = data.get("citations", [])

                            st.markdown(answer)

                            if citations:
                                for i, cit in enumerate(citations, 1):
                                    with st.expander(f"[NG12 p.{cit['page']}] {cit['chunk_id']}"):
                                        st.text(cit["excerpt"])

                            st.session_state.chat_messages.append(
                                {"role": "user", "content": user_input, "citations": []}
                            )
                            st.session_state.chat_messages.append(
                                {"role": "assistant", "content": answer, "citations": citations}
                            )
                        else:
                            st.error(f"API error: {resp.status_code} - {resp.text}")

                    except requests.exceptions.Timeout:
                        st.error("Request timed out. Try a simpler question.")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
