# paste this into chatbot.py replacing the previous render_chat_widget function

import streamlit as st
from datetime import datetime
from typing import Callable, Optional

def render_chat_widget(
    call_api: Callable[[str, Optional[str]], dict],
    extract_fn: Callable[[dict], str],
    prefix: str = "langflow",
    show_compact_first: bool = True,
    placeholder_first: str = "Ask me anything...",
    placeholder_followup: str = "Type a follow-up and press Enter..."
):
    """
    Compact-first-then-full chat widget that only shows the follow-up input
    after the bot's reply has been appended (avoids the input appearing above the bot reply).
    """
    # session keys (unique per prefix)
    hist_key = f"{prefix}_hist"
    shown_key = f"{prefix}_shown"
    proc_key = f"{prefix}_proc"
    first_key = f"{prefix}_first_val"
    first_sent_key = f"{prefix}_first_sent"
    await_key = f"{prefix}_awaiting_bot"   # NEW: indicates we are waiting for bot reply

    # defaults
    if hist_key not in st.session_state:
        st.session_state[hist_key] = []  # list[(role, text, ts)]
    if shown_key not in st.session_state:
        st.session_state[shown_key] = False
    if proc_key not in st.session_state:
        st.session_state[proc_key] = False
    if first_key not in st.session_state:
        st.session_state[first_key] = ""
    if await_key not in st.session_state:
        st.session_state[await_key] = False

    # header (optional)
    st.markdown("### AI ASSISTANTssistant" )

    # ---------- COMPACT TOP INPUT (only until first message) ----------
    if show_compact_first and not st.session_state[shown_key]:
        col_in, col_btn = st.columns([9, 1])
        with col_in:
            first_val = st.text_input("", key=first_key, placeholder=placeholder_first)
        with col_btn:
            send_click = st.button("Send", key=f"{prefix}_first_send", use_container_width=True)

        first_triggered = (first_val.strip() != "" and not st.session_state.get(first_sent_key, False)) or send_click

        if first_triggered:
            if st.session_state[proc_key]:
                st.warning("Please wait — processing previous request...")
            else:
                # Begin processing: hide follow-up input until bot reply appended
                st.session_state[proc_key] = True
                st.session_state[await_key] = True
                try:
                    user_msg = first_val.strip()
                    if user_msg == "":
                        st.session_state[proc_key] = False
                        st.session_state[await_key] = False
                    else:
                        ts = datetime.now().strftime("%H:%M:%S")
                        # append user message immediately
                        st.session_state[hist_key].append(("You", user_msg, ts))
                        st.session_state[first_sent_key] = True
                        # Build a short context safely (optional)
                        context = ""
                        if st.session_state.get("results_df") is not None:
                            try:
                                df = st.session_state["results_df"]
                                context = f"Analysis summary: total_providers={len(df)}\n\n"
                            except Exception:
                                context = ""
                        prompt = context + "User question: " + user_msg

                        # Call model (synchronous)
                        with st.spinner("🤖 Thinking..."):
                            try:
                                api_res = call_api(prompt, st.session_state.get("session_id"))
                                if api_res and isinstance(api_res, dict) and api_res.get("success"):
                                    bot_text = extract_fn(api_res.get("response"))
                                elif api_res and isinstance(api_res, dict):
                                    bot_text = api_res.get("error", "⚠️ Model returned an error.")
                                else:
                                    bot_text = "⚠️ Invalid response from model."
                            except Exception as e:
                                bot_text = f"⚠️ Error calling model: {e}"

                        # append bot reply, then allow follow-up input to show
                        st.session_state[hist_key].append(("Bot", bot_text, ts))
                        st.session_state[shown_key] = True
                finally:
                    st.session_state[proc_key] = False
                    st.session_state[await_key] = False  # IMPORTANT: follow-up input will be rendered now

        st.markdown("---")

    # ---------- FULL CHAT AREA ----------
    if (not show_compact_first) or st.session_state[shown_key]:
        # render chat history
        for role, text, ts in st.session_state[hist_key][-200:]:
            if role == "You":
                st.markdown(
                    f"<div style='background:#e6f7ff;padding:8px;border-radius:8px;margin:6px 0;'><strong>You</strong> <span style='color:#666;font-size:12px;'> {ts}</span><div style='margin-top:6px'>{text}</div></div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"<div style='background:#f3f3f3;padding:8px;border-radius:8px;margin:6px 0;'><strong>Assistant</strong> <span style='color:#666;font-size:12px;'> {ts}</span><div style='margin-top:6px'>{text}</div></div>",
                    unsafe_allow_html=True,
                )

        st.markdown("---")

        # ONLY show the follow-up input if we're NOT currently awaiting bot reply
        if not st.session_state[await_key]:
            followup = st.chat_input(placeholder=placeholder_followup)
            if followup:
                if st.session_state[proc_key]:
                    st.warning("Please wait — processing previous request...")
                else:
                    st.session_state[proc_key] = True
                    st.session_state[await_key] = True
                    try:
                        ts = datetime.now().strftime("%H:%M:%S")
                        st.session_state[hist_key].append(("You", followup, ts))

                        context = ""
                        if st.session_state.get("results_df") is not None:
                            try:
                                df = st.session_state["results_df"]
                                context = f"Analysis summary: total_providers={len(df)}\n\n"
                            except Exception:
                                context = ""
                        prompt = context + "User question: " + followup

                        with st.spinner("🤖 Thinking..."):
                            try:
                                api_res = call_api(prompt, st.session_state.get("session_id"))
                                if api_res and isinstance(api_res, dict) and api_res.get("success"):
                                    bot_text = extract_fn(api_res.get("response"))
                                elif api_res and isinstance(api_res, dict):
                                    bot_text = api_res.get("error", "⚠️ Model returned an error.")
                                else:
                                    bot_text = "⚠️ Invalid response from model."
                            except Exception as e:
                                bot_text = f"⚠️ Error calling model: {e}"

                        # append bot reply and clear waiting flag to allow next input to appear beneath it
                        st.session_state[hist_key].append(("Bot", bot_text, ts))
                    finally:
                        st.session_state[proc_key] = False
                        st.session_state[await_key] = False

        else:
            # Optional: show a subtle "waiting for reply" indicator where the follow-up input would be
            st.markdown("<div style='color:#666;font-size:13px;margin-top:6px;'>Waiting for assistant's reply...</div>", unsafe_allow_html=True)

        # Controls (clear)
        c1, c2 = st.columns([1, 5])
        with c1:
            if st.button("Clear", key=f"{prefix}_clear"):
                st.session_state[hist_key] = []
                st.session_state[shown_key] = False
                st.session_state[first_key] = ""
                if first_sent_key in st.session_state:
                    del st.session_state[first_sent_key]
                st.experimental_rerun()
        with c2:
            st.markdown("<div style='color:#666;font-size:12px;'>Tip: press Enter to send.</div>", unsafe_allow_html=True)
