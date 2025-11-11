import os
import numpy as np
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

from database import SessionLocal, engine, Base
from models import User, PatientProfile, SensorStream, Prediction, TwinModel
from util.auth import authenticate
from audit import log_action
from report import generate_report
from data_ingestion import parse_and_store

st.set_page_config(page_title="Digital-Twin Recovery Companion", layout="wide")

# Ensure DB tables exist
Base.metadata.create_all(bind=engine)

# ---------------------- SAFE DEMO SEED ----------------------
def seed_demo():
    """Seed base users only once (safe, portable hash)."""
    db = SessionLocal()
    try:
        if not db.query(User).filter(User.email == "admin@example.com").first():
            from passlib.context import CryptContext
            # ✅ Use pure-Python hash (no bcrypt backend required)
            pwd = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")

            for email, role, name in [
                ("admin@example.com", "admin", "Admin User"),
                ("clinician@example.com", "clinician", "Clinician One"),
                ("patient@example.com", "patient", "Patient One"),
            ]:
                u = User(
                    email=email,
                    hashed_password=pwd.hash("changeme"),
                    role=role,
                    full_name=name
                )
                db.add(u)
                db.commit()
                db.refresh(u)
                if role == "patient":
                    p = PatientProfile(
                        user_id=u.id,
                        demographics={"age": 45},
                        medical_history="post-op knee"
                    )
                    db.add(p)
                    db.commit()
            print("✅ Seeded demo users successfully.")
    except Exception as e:
        print(f"⚠️ Seeding skipped or failed: {e}")
    finally:
        db.close()

# Run seeding manually only if environment variable is set
if os.getenv("SEED_ON_STARTUP", "0") == "1":
    seed_demo()
# (On Streamlit Cloud, run `python seed.py` or set SEED_ON_STARTUP=1 manually)

# ---------------------- SESSION ----------------------
if "user" not in st.session_state:
    st.session_state.user = None
if "role" not in st.session_state:
    st.session_state.role = None

# ---------------------- SIDEBAR ----------------------
with st.sidebar:
    st.markdown("## 🏥 Digital-Twin")
    judge = st.toggle("🎯 Judge Mode (auto-demo)", value=True)
    st.caption("Auto-login and preload demo data for instant walkthrough.")

    if judge and not st.session_state.user:
        db = SessionLocal()
        user = db.query(User).filter(User.email == "patient@example.com").first()
        if user:
            st.session_state.user = user
            st.session_state.role = "patient"
            st.success("Auto-logged in as patient@example.com")
        db.close()

    if st.session_state.user:
        st.success(f"Logged in: {st.session_state.user.email} ({st.session_state.role})")
        if st.button("Logout", use_container_width=True):
            st.session_state.user = None
            st.session_state.role = None
            st.rerun()
    else:
        st.subheader("Login")
        email = st.text_input("Email", value="patient@example.com")
        password = st.text_input("Password", value="changeme", type="password")
        role_pick = st.selectbox("Role", options=["patient", "clinician", "admin"])
        if st.button("Sign in", use_container_width=True):
            db = SessionLocal()
            user = authenticate(db, email, password)
            if user and (role_pick == user.role or role_pick == "admin"):
                st.session_state.user = user
                st.session_state.role = user.role
                log_action(db, user.id, "login", {"role": user.role})
                st.rerun()
            else:
                st.error("Invalid credentials or role mismatch")
            db.close()

st.title("Digital-Twin Recovery Companion")

if not st.session_state.user:
    st.info("Please log in from the sidebar to continue.")
    st.stop()

role = st.session_state.user.role if hasattr(st.session_state.user, "role") else "patient"

# ---------------------- TABS ----------------------
if role == "patient":
    tabs = st.tabs(["🏠 Overview", "📥 Data Ingestion"])
elif role == "clinician":
    tabs = st.tabs(["👩‍⚕️ Clinician", "📥 Data Ingestion"])
else:
    tabs = st.tabs(["👩‍⚕️ Clinician", "📥 Data Ingestion", "🛠️ Admin"])

# -------- Overview / Clinician Tab --------
with tabs[0]:
    col_info, col_sim = st.columns([2, 1], gap="large")

    with col_info:
        st.subheader("Recovery Progress")

        days = list(range(0, 30))
        values = [0.35 + i * 0.02 + np.sin(i / 3) * 0.01 for i in days]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=days, y=values, mode="lines+markers", name="Recovery Index"))
        fig.update_layout(
            margin=dict(l=10, r=10, t=40, b=10),
            height=320,
            title="30-Day Recovery Trajectory",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Mood & Pain Heatmap")
        timeslots = ["08:00", "12:00", "16:00", "20:00"]
        days_labels = [
            (datetime.today() - timedelta(days=i)).strftime("%a %d") for i in range(6, -1, -1)
        ]
        pain = np.clip(np.random.normal(4, 1.5, size=(7, 4)), 0, 10)
        heat = go.Figure(
            data=go.Heatmap(
                z=pain,
                x=timeslots,
                y=days_labels,
                zmin=0,
                zmax=10,
                colorbar=dict(title="Pain"),
            )
        )
        heat.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=260)
        st.plotly_chart(heat, use_container_width=True)

        st.markdown("#### Digital Twin (Animated)")
        frames = []
        base_x = np.array([0, 0, -0.3, 0, 0.3, 0, 0, -0.2, 0, 0.2])
        base_y = np.array([1.8, 1.4, 1.1, 1.4, 1.1, 1.4, 0.8, 0.2, 0.8, 0.2])
        for t in range(20):
            tilt = 0.05 * np.sin(t / 3)
            xs = base_x + np.array([0, 0, tilt, 0, -tilt, 0, 0, tilt, 0, -tilt])
            frames.append(go.Frame(data=[go.Scatter3d(x=xs, y=base_y, z=[0] * 10, mode="lines")]))
        twin = go.Figure(
            data=[go.Scatter3d(x=base_x, y=base_y, z=[0] * 10, mode="lines")],
            frames=frames,
        )
        twin.update_layout(
            scene=dict(
                xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)
            ),
            margin=dict(l=10, r=10, t=10, b=10),
            height=320,
            updatemenus=[
                {
                    "type": "buttons",
                    "buttons": [{"label": "Play", "method": "animate", "args": [None]}],
                }
            ],
        )
        st.plotly_chart(twin, use_container_width=True)

    with col_sim:
        st.subheader("What-if Simulation")
        extra = st.slider("Extra balance training (min/day)", 0, 60, 10)
        conf = st.select_slider("Confidence", options=["Low", "Medium", "High"], value="Medium")
        if st.button("Run Simulation", use_container_width=True):
            model = TwinModel()
            pred = model.predict(patient_id=1, scenario={"extra_minutes_balance": extra})
            log_action(
                SessionLocal(), st.session_state.user.id, "prediction", {"extra_minutes": extra, "conf": conf}
            )
            st.metric("Predicted gait speed Δ", f"{pred['gait_speed_change_pct']} %")
            st.metric("Adherence score", f"{pred['adherence_score']}")
        st.markdown("---")
        st.markdown("#### Generate PDF Report")
        patient_name = st.text_input("Patient Name", value="Patient One")
        if st.button("Download Report", use_container_width=True):
            metrics = {
                "Gait Speed Change %": 12.5,
                "Adherence Score": 85,
                "Next Step": "Add 5 min balance training",
            }
            pdf_bytes = generate_report(patient_name or "Unknown", metrics)
            st.download_button(
                "Download PDF",
                data=pdf_bytes,
                file_name="recovery_report.pdf",
                mime="application/pdf",
            )

# -------- Data Ingestion Tab --------
with tabs[1]:
    st.subheader("📥 Ingest Wearable CSV")
    st.write(
        "Upload CSV with columns: `timestamp, accel_x, accel_y, accel_z, emg, spo2, hr, step_count`."
    )
    uploaded = st.file_uploader("Choose CSV file", type=["csv"])
    feats_state_key = "latest_feats"
    if uploaded is not None:
        db = SessionLocal()
        patient_profile = db.query(PatientProfile).filter(
            PatientProfile.user_id == st.session_state.user.id
        ).first()
        if not patient_profile:
            patient_profile = PatientProfile(user_id=st.session_state.user.id, demographics={}, medical_history="")
            db.add(patient_profile)
            db.commit()
            db.refresh(patient_profile)
        try:
            head_df, feats = parse_and_store(uploaded.read(), patient_profile.id, db)
            st.session_state[feats_state_key] = feats
            st.success("Data ingested! Preview & features below.")
            log_action(db, st.session_state.user.id, "csv_upload", {"rows": len(head_df)})
            st.dataframe(head_df, use_container_width=True)
            st.json(feats)
        except Exception as e:
            st.error(f"Failed to parse CSV: {e}")
        finally:
            db.close()

    st.markdown("#### Predict from Uploaded Features")
    extra2 = st.slider("Extra balance training (min/day)", 0, 60, 15, key="extra2")
    if st.button("Predict from Features", use_container_width=True):
        feats = st.session_state.get(feats_state_key)
        if not feats:
            st.warning("Please upload CSV first to compute features.")
        else:
            model = TwinModel()
            res = model.predict(patient_id=1, scenario={"extra_minutes_balance": extra2}, feats=feats)
            log_action(SessionLocal(), st.session_state.user.id, "prediction", {"extra_minutes": extra2})
            st.metric("Predicted gait speed Δ", f"{res['gait_speed_change_pct']} %")
            st.metric("Adherence score", f"{res['adherence_score']}")

# -------- Admin Tab --------
if role == "admin":
    with tabs[2]:
        st.subheader("🛠️ Admin")
        db = SessionLocal()
        st.markdown("Create Clinician/Patient Users")
        full_name = st.text_input("Full name")
        email_new = st.text_input("Email")
        pw_new = st.text_input("Password", type="password")
        role_new = st.selectbox("Role", ["patient", "clinician"])

        if st.button("Create User", use_container_width=True):
            if not email_new or not pw_new:
                st.error("Email and password required")
            else:
                exists = db.query(User).filter(User.email == email_new).first()
                if exists:
                    st.error("Email already exists")
                else:
                    from passlib.context import CryptContext
                    # ✅ Changed to pbkdf2_sha256
                    pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")
                    u = User(email=email_new, hashed_password=pwd_context.hash(pw_new),
                             role=role_new, full_name=full_name)
                    db.add(u)
                    db.commit()
                    db.refresh(u)
                    if role_new == "patient":
                        p = PatientProfile(user_id=u.id, demographics={}, medical_history="")
                        db.add(p)
                        db.commit()
                    st.success(f"Created {role_new}: {email_new}")
        st.markdown("---")
        st.markdown("### System Info")
        st.code(f"DB = {os.getenv('DATABASE_URL', 'sqlite:///./data/app.db')}")
        db.close()
