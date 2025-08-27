"""
login.py

Improved Streamlit login/signup single-card component:
- Card is centered both vertically and horizontally.
- Inputs and buttons are contained inside the white card.
- Page background uses a sky-blue gradient (light at top).
- Card size constrained to medium (approx 420px wide).
- Uses a form so inputs stay within the card.
- Compatibility safe_rerun() included.

Security: demo only. Use proper backends and secure hashing for production.
"""

import streamlit as st
import os
import json
import hashlib
import datetime
from typing import Tuple

CRED_FILE = "credentials.json"
# Sky-blue palette: light at top, deeper at bottom
PRIMARY_LIGHT = "#DFF6FF"  # light sky blue (top)
PRIMARY_DARK = "#007ACC"   # deeper sky blue (bottom)
CARD_BG = "#ffffff"

def safe_rerun() -> None:
    try:
        if hasattr(st, "experimental_rerun"):
            st.experimental_rerun()
            return
    except Exception:
        pass
    try:
        if hasattr(st, "rerun"):
            st.rerun()
            return
    except Exception:
        pass
    st.session_state["__rerun_marker"] = not st.session_state.get("__rerun_marker", False)
    st.stop()

def _ensure_cred_file() -> None:
    if not os.path.exists(CRED_FILE):
        with open(CRED_FILE, "w") as f:
            json.dump({"users": {}}, f, indent=2)

def load_credentials() -> dict:
    _ensure_cred_file()
    with open(CRED_FILE, "r") as f:
        return json.load(f)

def save_credentials(data: dict) -> None:
    with open(CRED_FILE, "w") as f:
        json.dump(data, f, indent=2, default=str)

def hash_password(username: str, password: str) -> str:
    salted = (username + "::" + password).encode("utf-8")
    return hashlib.sha256(salted).hexdigest()

def register_user(username: str, password: str) -> Tuple[bool, str]:
    username = username.strip()
    if not username or not password:
        return False, "Username and password required."
    data = load_credentials()
    if username in data.get("users", {}):
        return False, "User already exists."
    pwd_hash = hash_password(username, password)
    data.setdefault("users", {})[username] = {
        "password_hash": pwd_hash,
        "created_at": datetime.datetime.utcnow().isoformat() + "Z",
    }
    save_credentials(data)
    return True, "User registered successfully."

def authenticate_user(username: str, password: str) -> Tuple[bool, str]:
    username = username.strip()
    if not username or not password:
        return False, "Username and password required."
    data = load_credentials()
    user = data.get("users", {}).get(username)
    if not user:
        return False, "User not found."
    pwd_hash = hash_password(username, password)
    if pwd_hash == user.get("password_hash"):
        data["last_login"] = {"user": username, "at": datetime.datetime.utcnow().isoformat() + "Z"}
        save_credentials(data)
        return True, "Authenticated"
    return False, "Incorrect password."

def logout() -> None:
    st.session_state["authenticated"] = False
    st.session_state["user"] = None
    safe_rerun()

def login_component() -> None:
    """
    Renders a centered login/signup card. Uses Streamlit columns so widgets
    render inside the centered column (no raw open <div> wrapper that would break widget nesting).
    """
    # CSS: not an f-string to avoid brace escaping mistakes. Use .format().
    css = """
    <style>
    .stApp {{
        background: linear-gradient(180deg, {PRIMARY_LIGHT} 0%, {PRIMARY_DARK} 100%);
        min-height: 100vh;
        padding: 24px;
    }}


    .login-header {{
        display:flex; align-items:center; gap:12px; margin-bottom:10px;
    }}
    .login-badge {{
        width:46px; height:46px; border-radius:10px; display:flex; align-items:center; justify-content:center;
        color:white; font-weight:800; font-size:18px; background: linear-gradient(90deg, {PRIMARY_DARK}, {PRIMARY_LIGHT});
    }}
    .login-title {{ font-size:18px; font-weight:800; color:{PRIMARY_DARK}; margin:0; }}
    .login-sub {{ font-size:13px; color:#5a6b78; margin-top:2px; }}

    /* Make inputs look compact */
    .stTextInput>div>div>input, .stTextArea>div>div>textarea {{
        padding:10px 12px !important;
        border-radius:10px !important;
        border:1px solid #edf7ff !important;
        font-size:14px !important;
    }}

    .stButton>button {{
    border-radius: 10px;
    padding: 10px 30px;
    font-weight: 450;
    box-shadow: none;
    white-space: nowrap;     /* Prevents breaking into two lines */
    min-width: 100px;        /* Ensures enough width for "Submit" */
    }}


    .small-link {{ font-size:13px; color:{PRIMARY_DARK}; text-decoration:underline; cursor:pointer; }}
    </style>
    """.format(PRIMARY_LIGHT=PRIMARY_LIGHT, PRIMARY_DARK=PRIMARY_DARK, CARD_BG=CARD_BG)

    st.markdown(css, unsafe_allow_html=True)

    # Initialize session_state defaults
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False
    if "user" not in st.session_state:
        st.session_state["user"] = None
    if "__rerun_marker" not in st.session_state:
        st.session_state["__rerun_marker"] = False

    # Use three columns to center the middle column. The middle column holds the card.
    col_left, col_mid, col_right = st.columns([1, 0.7, 1])

    with col_mid:
        # Draw a visual "card" container using Markdown (this is only visual; widgets that follow are actually
        # placed in this same column so they appear inside/under the card visually).
        st.markdown('<div class="login-card">', unsafe_allow_html=True)

        # Header (HTML inside the card)
        st.markdown(
            """
            <div class="login-header">
                <div class="login-badge">F</div>
                <div>
                    <div class="login-title">Fanatic</div>
                    <div class="login-sub">AI-powered Fraud Risk Insights</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Now create the actual Streamlit form — because we're inside col_mid, widgets render inside that column
        with st.form("auth_form"):
            mode = st.radio(
                "Select mode",  # non-empty label
                ("Sign in", "Sign up"),
                index=0,
                horizontal=True,
                label_visibility="collapsed"  # hides the label visually
            )

            remembered = load_credentials().get("remembered_user")
            username_default = remembered if remembered else ""
            username = st.text_input("Username", value=username_default, key="login_username", placeholder="your.username")
            password = st.text_input("Password", key="login_password", type="password", placeholder="••••••••")
            col1, col2 = st.columns([3, 1])
            with col1:
                remember = st.checkbox("Remember me", key="login_remember")
            with col2:
                submit = st.form_submit_button("Submit")

            if submit:
                if not username or not password:
                    st.error("Please provide username and password.")
                else:
                    if mode == "Sign up":
                        ok, msg = register_user(username, password)
                        if ok:
                            st.success(msg + " — you can now sign in.")
                        else:
                            st.error(msg)
                    else:
                        ok, msg = authenticate_user(username, password)
                        if ok:
                            st.success("Welcome back, " + username + "!")
                            st.session_state["authenticated"] = True
                            st.session_state["user"] = username
                            if remember:
                                data = load_credentials()
                                data["remembered_user"] = username
                                save_credentials(data)
                            safe_rerun()
                        else:
                            st.error(msg)


# Exports
__all__ = ["login_component", "authenticate_user", "register_user", "logout", "load_credentials", "save_credentials"]
