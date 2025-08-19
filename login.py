import streamlit as st
import pyrebase
import os
import json

# Firebase configuration
firebaseConfig = {
    "apiKey": "AIzaSyAG_OYVtCUP-GGYacmw8YVBsXSQ1ayuZjM",
    "authDomain": "ai-summarizer-15092.firebaseapp.com",
    "projectId": "ai-summarizer-15092",
    "storageBucket": "ai-summarizer-15092.firebasestorage.app",
    "messagingSenderId": "390463874678",
    "appId": "1:390463874678:web:e56e51ac448a50f5d1a157",
    "measurementId": "G-7EYDWKYEX0",
    "databaseURL": "https://ai-summarizer-15092-default-rtdb.firebaseio.com/"
}

firebase = pyrebase.initialize_app(firebaseConfig)
auth = firebase.auth()

# Session state
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user_email" not in st.session_state:
    st.session_state.user_email = None

st.title("YouTube Summarizer Login")

menu = st.selectbox("Choose Action", ["Login", "Sign Up", "Login with Google"])

email = st.text_input("Email")
password = st.text_input("Password", type="password")

# ----------------- SIGN UP -----------------
if menu == "Sign Up":
    if st.button("Create Account"):
        if email and password:
            try:
                user = auth.create_user_with_email_and_password(email, password)
                st.success("Account created successfully! Please login.")
            except Exception as e:
                try:
                    error_json = e.args[1]
                    if "EMAIL_EXISTS" in error_json:
                        st.warning("Email already exists. Please login instead.")
                    else:
                        st.error(f"Error: {e}")
                except:
                    st.error(f"Error: {e}")
        else:
            st.warning("Please enter both email and password.")

# ----------------- LOGIN -----------------
elif menu == "Login":
    if st.button("Login"):
        if email and password:
            try:
                user = auth.sign_in_with_email_and_password(email, password)
                st.success(f"Logged in as {email}")
                st.session_state.logged_in = True
                st.session_state.user_email = email

                # Redirect to app.py
                st.experimental_set_query_params(page="app")
                st.experimental_rerun()
            except Exception as e:
                st.error(f"Login failed: {e}")
        else:
            st.warning("Please enter both email and password.")

# ----------------- GOOGLE LOGIN (Placeholder) -----------------
elif menu == "Login with Google":
    st.info("Google login is not natively supported in Pyrebase. You need OAuth setup.")
    if st.button("Sign in with Google (Placeholder)"):
        st.warning("This is a placeholder. Implement Google OAuth redirect to Firebase manually.")

# ----------------- REDIRECT -----------------
query_params = st.experimental_get_query_params()
if query_params.get("page") == ["app"] and st.session_state.logged_in:
    st.success(f"Redirecting {st.session_state.user_email} to app...")
    os.system("streamlit run app.py")
