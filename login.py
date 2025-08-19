import streamlit as st
import pyrebase
import os

firebaseConfig = st.secrets["firebase"]

firebase = pyrebase.initialize_app(firebaseConfig)
auth = firebase.auth()

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user_email" not in st.session_state:
    st.session_state.user_email = None

st.title("YouTube Summarizer Login")

menu = st.selectbox("Choose Action", ["Login", "Sign Up", "Login with Google"])

email = st.text_input("Email")
password = st.text_input("Password", type="password")

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

elif menu == "Login":
    if st.button("Login"):
        if email and password:
            try:
                user = auth.sign_in_with_email_and_password(email, password)
                st.success(f"Logged in as {email}")
                st.session_state.logged_in = True
                st.session_state.user_email = email
                st.experimental_set_query_params(page="app")
                st.experimental_rerun()
            except Exception as e:
                st.error(f"Login failed: {e}")
        else:
            st.warning("Please enter both email and password.")

elif menu == "Login with Google":
    st.info("Google login requires custom OAuth redirect setup.")
    if st.button("Sign in with Google (Placeholder)"):
        st.warning("This is just a placeholder. Implement Google OAuth manually.")

query_params = st.experimental_get_query_params()
if query_params.get("page") == ["app"] and st.session_state.logged_in:
    st.success(f"Redirecting {st.session_state.user_email} to app...")
