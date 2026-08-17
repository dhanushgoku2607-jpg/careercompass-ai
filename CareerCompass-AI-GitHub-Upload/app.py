from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "model" / "placement_model.joblib"
METRICS_PATH = ROOT / "model" / "model_metrics.joblib"
DATA_PATH = ROOT / "data" / "placementdata.csv"
st.set_page_config(page_title="CareerCompass AI", page_icon="🎓", layout="wide")

st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700;800&family=Lora:ital,wght@0,400;0,500;0,600;1,400&display=swap');
:root{--parchment:#f3ead9;--ink:#2c2418;--maroon:#6e1f2b;--maroon-dark:#4f151f;--gold:#b6892f;--gold-light:#d9b264;--forest:#3c5a45;--shadow:0 6px 18px rgba(78,58,25,.15)}
html,body,[class*="css"]{font-family:'Lora',Georgia,serif}.stApp{background:radial-gradient(circle at 15% 10%,rgba(182,137,47,.08),transparent 40%),radial-gradient(circle at 85% 90%,rgba(110,31,43,.07),transparent 45%),var(--parchment);color:var(--ink)}
h1,h2,h3,.hero h1{font-family:'Playfair Display',Georgia,serif!important;color:var(--maroon-dark)!important;letter-spacing:.02em}.stApp h2,.stApp h3{border-bottom:1px solid var(--gold-light);padding-bottom:.35rem}
.stApp label,.stApp [data-testid="stWidgetLabel"] p,.stApp .stMarkdown,.stApp .stMarkdown p,.stApp span,.stApp p{color:var(--ink)!important}
.hero{padding:1.6rem 2rem;margin-bottom:1.2rem;background:linear-gradient(135deg,var(--maroon),var(--maroon-dark));border:1px solid var(--gold);border-radius:10px;box-shadow:var(--shadow)}.hero h1{color:#f6ecd8!important;margin:0 0 .3rem}.hero p{color:#e9d9b8!important;margin:0;font-style:italic}
section[data-testid="stSidebar"]{background:linear-gradient(180deg,#2c2418,#3a2f1e);border-right:2px solid var(--gold)}section[data-testid="stSidebar"] *{color:#f0e6cf!important}
.classic-card{background:#fbf6ea;border:1px solid var(--gold-light);border-left:5px solid var(--maroon);border-radius:8px;padding:1.1rem 1.4rem;margin-bottom:1rem;box-shadow:var(--shadow)}.classic-card h4{font-family:'Playfair Display',serif;color:var(--maroon-dark);margin:0 0 .5rem}.tip-item{padding:.55rem .8rem;margin-bottom:.5rem;background:#fffaf0;border-left:3px solid var(--gold);border-radius:4px}
div[data-testid="stMetric"]{background:#fbf6ea;border:1px solid var(--gold-light);border-radius:8px;padding:.8rem 1rem;box-shadow:var(--shadow)}div[data-testid="stMetricLabel"]{color:var(--maroon-dark)!important;font-weight:600}div[data-testid="stMetricValue"]{color:var(--ink)!important;font-family:'Playfair Display',serif}
.stButton>button{background:linear-gradient(135deg,var(--maroon),var(--maroon-dark));color:#f6ecd8;border:1px solid var(--gold);border-radius:6px;font-family:'Playfair Display',serif;padding:.6rem 1rem}.stButton>button:hover{color:#fff;border-color:var(--gold-light)}div[data-testid="stProgress"]>div>div,div[data-testid="stSlider"] [role="slider"]{background-color:var(--maroon)!important}.ornament{text-align:center;color:var(--gold);margin:.4rem 0 1.2rem;letter-spacing:.4em}
</style>""", unsafe_allow_html=True)

@st.cache_resource
def load_assets():
    if not (MODEL_PATH.exists() and METRICS_PATH.exists() and DATA_PATH.exists()):
        return None, None, None
    return joblib.load(MODEL_PATH), joblib.load(METRICS_PATH), pd.read_csv(DATA_PATH)

def recommendations(v):
    checks = [(v["CGPA"] < 7, "Improve core-subject performance and aim for a stronger CGPA."), (v["AptitudeTestScore"] < 65, "Practice aptitude questions and take timed mock tests each week."), (v["SoftSkillsRating"] < 3.5, "Develop communication through mock interviews and group discussions."), (v["Projects"] < 2, "Build and publish two end-to-end portfolio projects on GitHub."), (v["Internships"] == 0, "Apply for internships, virtual experience programmes, or faculty-led projects."), (v["Workshops/Certifications"] < 2, "Complete a relevant certificate in Python, cloud, or data analytics."), (v["PlacementTraining"] == "No", "Attend placement-training sessions and practise HR interview questions.")]
    return [tip for condition, tip in checks if condition] or ["Excellent profile. Focus on mock interviews, networking, and tailoring your resume to each role."]

def profile_radar(v):
    categories = ["CGPA", "Aptitude", "Soft skills", "Projects", "Experience"]
    scores = [min(v["CGPA"] / 10 * 100, 100), v["AptitudeTestScore"], v["SoftSkillsRating"] / 5 * 100, min(v["Projects"] / 5 * 100, 100), min((v["Internships"] + v["Workshops/Certifications"]) / 6 * 100, 100)]
    chart = go.Figure(go.Scatterpolar(r=scores + scores[:1], theta=categories + categories[:1], fill="toself", line=dict(color="#6e1f2b", width=2), fillcolor="rgba(110,31,43,.25)"))
    chart.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,100], gridcolor="#d9b264", tickfont=dict(color="#2c2418")), angularaxis=dict(tickfont=dict(color="#2c2418")), bgcolor="rgba(0,0,0,0)"), paper_bgcolor="rgba(0,0,0,0)", showlegend=False, margin=dict(t=20,b=20,l=30,r=30), height=360)
    return chart

CHART_STYLE = dict(font=dict(family="Lora, Georgia, serif", color="#2c2418"), title_font=dict(family="Playfair Display, Georgia, serif", color="#4f151f", size=18), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(251,246,234,.6)", legend=dict(bgcolor="rgba(0,0,0,0)"))
model, metrics, data = load_assets()
if model is None:
    st.error("Model files are missing. Run `python train_model.py` once, then refresh this page.")
    st.stop()
st.markdown("<div class='hero'><h1>🎓 CareerCompass AI</h1><p>Placement prediction and personalised skill-gap analysis for students.</p></div>", unsafe_allow_html=True)
page = st.sidebar.radio("Navigate", ["Predict placement", "Placement dashboard", "About the project"])
st.sidebar.markdown("<hr/>", unsafe_allow_html=True); st.sidebar.caption("Dataset: Kaggle Placement Prediction Dataset • Source recorded in README")

if page == "Predict placement":
    st.subheader("Student profile"); st.markdown("<div class='ornament'>— ✦ —</div>", unsafe_allow_html=True)
    left, right = st.columns(2)
    with left:
        cgpa=st.slider("CGPA",4.0,10.0,7.0,.1); aptitude=st.slider("Aptitude test score",0,100,65); soft_skills=st.slider("Soft-skills rating",1.0,5.0,3.5,.1); ssc=st.slider("SSC / 10th marks (%)",35,100,75); hsc=st.slider("HSC / 12th marks (%)",35,100,75)
    with right:
        internships=st.number_input("Internships",0,5,0); projects=st.number_input("Completed projects",0,10,2); certificates=st.number_input("Workshops / certifications",0,10,2); extracurricular=st.selectbox("Extracurricular activities",["No","Yes"]); training=st.selectbox("Placement training completed",["No","Yes"])
    values={"CGPA":cgpa,"Internships":internships,"Projects":projects,"Workshops/Certifications":certificates,"AptitudeTestScore":aptitude,"SoftSkillsRating":soft_skills,"ExtracurricularActivities":extracurricular,"PlacementTraining":training,"SSC_Marks":ssc,"HSC_Marks":hsc}
    if st.button("Analyze profile",type="primary",use_container_width=True):
        probability=model.predict_proba(pd.DataFrame([values]))[0][1]; strength="Strong" if probability>=.70 else "Developing" if probability>=.45 else "Needs focus"
        a,b,c=st.columns(3); a.metric("Placement likelihood",f"{probability:.0%}"); b.metric("Profile strength",strength); c.metric("Model test accuracy",f"{metrics['accuracy']:.0%}"); st.progress(int(probability*100))
        chart_col,tips_col=st.columns([1,1.1])
        with chart_col:
            st.markdown("<div class='classic-card'><h4>Readiness profile</h4>",unsafe_allow_html=True); st.plotly_chart(profile_radar(values),use_container_width=True,config={"displayModeBar":False}); st.markdown("</div>",unsafe_allow_html=True)
        with tips_col:
            st.markdown("<div class='classic-card'><h4>Your recommended next steps</h4>",unsafe_allow_html=True)
            for tip in recommendations(values): st.markdown(f"<div class='tip-item'>✦ {tip}</div>",unsafe_allow_html=True)
            st.markdown("</div>",unsafe_allow_html=True)
elif page == "Placement dashboard":
    st.subheader("Dataset overview"); st.markdown("<div class='ornament'>— ✦ —</div>",unsafe_allow_html=True)
    a,b,c=st.columns(3); a.metric("Students analysed",len(data)); b.metric("Placement rate",f"{(data.PlacementStatus=='Placed').mean():.0%}"); c.metric("Model test accuracy",f"{metrics['accuracy']:.0%}")
    c1,c2=st.columns(2)
    with c1:
        fig=px.histogram(data,x="CGPA",color="PlacementStatus",barmode="overlay",title="CGPA distribution by placement outcome",color_discrete_map={"Placed":"#3c5a45","NotPlaced":"#6e1f2b"});fig.update_layout(**CHART_STYLE);st.plotly_chart(fig,use_container_width=True)
    with c2:
        rates=data.assign(score_band=pd.cut(data.AptitudeTestScore,[0,40,60,80,100])).groupby("score_band",observed=False).PlacementStatus.apply(lambda x:(x=="Placed").mean()).reset_index(name="placement_rate"); rates["score_band"]=rates["score_band"].astype(str)
        fig=px.bar(rates,x="score_band",y="placement_rate",title="Placement rate by aptitude-score band",labels={"score_band":"Aptitude score","placement_rate":"Placement rate"},color_discrete_sequence=["#b6892f"]);fig.update_layout(**CHART_STYLE);st.plotly_chart(fig,use_container_width=True)
    d1,d2=st.columns(2)
    with d1:
        fig=px.box(data,x="PlacementStatus",y="SoftSkillsRating",color="PlacementStatus",title="Soft-skills rating by placement outcome",color_discrete_map={"Placed":"#3c5a45","NotPlaced":"#6e1f2b"});fig.update_layout(**CHART_STYLE,showlegend=False);st.plotly_chart(fig,use_container_width=True)
    with d2:
        rates=data.groupby("Projects").PlacementStatus.apply(lambda x:(x=="Placed").mean()).reset_index(name="placement_rate");fig=px.line(rates,x="Projects",y="placement_rate",markers=True,title="Placement rate by number of projects completed",color_discrete_sequence=["#6e1f2b"]);fig.update_layout(**CHART_STYLE);st.plotly_chart(fig,use_container_width=True)
    with st.expander("View raw dataset sample"): st.dataframe(data.head(50),use_container_width=True)
else:
    st.subheader("Project summary");st.markdown("<div class='ornament'>— ✦ —</div>",unsafe_allow_html=True)
    st.markdown("<div class='classic-card'><h4>What CareerCompass AI does</h4>CareerCompass AI predicts placement likelihood using academic performance, aptitude, experience, and placement-readiness indicators. It provides personalised, action-oriented skill-gap suggestions to help students strengthen weak areas before campus placements.</div><div class='classic-card'><h4>Machine-learning workflow</h4>Data cleaning → train/test split → scaling and categorical encoding → Random Forest classification → evaluation → Streamlit deployment.</div>",unsafe_allow_html=True)
