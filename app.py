import streamlit as st
import pandas as pd
import numpy as np
import os

st.set_page_config(
    page_title="DataSense Advisor",
    layout="wide",
    page_icon="📊",
    initial_sidebar_state="expanded"
)

# ── Session state defaults ────────────────────────────────────────────
for k, v in {
    "df": None, "cleaned_df": None,
    "dataset_name": "", "target_col": None, "theme": "dark",
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Theme colours ─────────────────────────────────────────────────────
DARK = {
    "bg":          "#0a0d14",
    "card":        "#161d2e",
    "side":        "#0d1117",
    "border":      "#1e293b",
    "text":        "#f1f5f9",
    "muted":       "#94a3b8",
    "accent":      "#00e5a0",
    "dim":         "#00b37a",
    "inp":         "#1a2235",
    "exp":         "#111827",
    "tbl_bg":      "#161d2e",
    "tbl_hdr":     "#1e293b",
    "tbl_hdr_txt": "#f1f5f9",
    "tbl_txt":     "#f1f5f9",
}
LIGHT = {
    "bg": "#f8fafc", "card": "#ffffff", "side": "#ffffff",
    "border": "#e2e8f0", "text": "#0f172a", "muted": "#64748b",
    "accent": "#059669", "dim": "#047857", "inp": "#f8fafc", "exp": "#f1f5f9",
    "tbl_bg": "#ffffff", "tbl_hdr": "#f1f5f9", "tbl_hdr_txt": "#0f172a", "tbl_txt": "#0f172a",
}
T = DARK if st.session_state.theme == "dark" else LIGHT

# ── CSS ───────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, .stApp {{
    background-color: {T['bg']} !important;
    font-family: 'DM Sans', sans-serif !important;
    color: {T['text']} !important;
}}
header[data-testid="stHeader"], [data-testid="stToolbar"],
[data-testid="stDecoration"], [data-testid="stStatusWidget"],
[data-testid="stBaseButton-headerNoPadding"],
.stDeployButton, #MainMenu {{ display: none !important; }}

[data-testid="stSidebar"] {{
    background-color: {T['side']} !important;
    border-right: 1px solid {T['border']} !important;
    min-width: 230px !important;
    max-width: 250px !important;
    transition: none !important;
    transform: translateX(0) !important;
    visibility: visible !important;
    opacity: 1 !important;
}}
[data-testid="stSidebar"] > div {{
    min-width: 230px !important;
    max-width: 250px !important;
    transition: none !important;
}}
[data-testid="stSidebar"] * {{
    font-family: 'DM Sans', sans-serif !important;
    color: {T['text']} !important;
}}
[data-testid="stSidebarCollapseButton"] button {{
    background-color: {T['card']} !important;
    border: 1px solid {T['border']} !important;
    border-radius: 6px !important;
    transition: none !important;
}}
section[data-testid="stSidebar"],
section[data-testid="stSidebar"]:hover,
section[data-testid="stSidebar"]:focus,
section[data-testid="stSidebar"]:focus-within {{
    transition: none !important;
    animation: none !important;
    transform: translateX(0) !important;
    visibility: visible !important;
    opacity: 1 !important;
}}

.main .block-container {{
    background-color: {T['bg']} !important;
    padding-top: 1.5rem !important;
    padding-left: 2rem !important; padding-right: 2rem !important;
    max-width: 1200px !important;
}}

h1 {{
    color: {T['accent']} !important; font-size: 1.6rem !important; font-weight: 700 !important;
    border-bottom: 2px solid {T['accent']}; padding-bottom: 0.3rem; margin-bottom: 0.2rem !important;
}}
h2 {{ color: {T['accent']} !important; font-size: 1.05rem !important; font-weight: 600 !important; }}
h3 {{ color: {T['accent']} !important; font-size: 0.92rem !important; font-weight: 600 !important; }}
p, div, span, label {{ font-size: 14px !important; color: {T['text']} !important; }}

/* Metric cards — no truncation */
[data-testid="stMetric"] {{
    background-color: {T['card']} !important;
    border: 1px solid {T['border']} !important;
    border-radius: 10px !important; padding: 0.85rem 1rem !important;
    overflow: visible !important;
}}
[data-testid="stMetricLabel"] {{
    color: {T['muted']} !important; font-size: 0.6rem !important;
    text-transform: uppercase !important; letter-spacing: 0.7px !important;
    white-space: normal !important; word-break: break-word !important;
}}
[data-testid="stMetricValue"] {{
    color: {T['accent']} !important; font-size: 1.25rem !important;
    font-weight: 700 !important; font-family: 'JetBrains Mono', monospace !important;
    white-space: normal !important; word-break: break-all !important;
    overflow: visible !important;
}}

/* DataFrame — column headers always visible */
.stDataFrame {{
    background-color: {T['tbl_bg']} !important;
    border: 1px solid {T['border']} !important;
    border-radius: 10px !important; overflow: hidden;
}}
.stDataFrame thead tr th,
.stDataFrame [data-testid="glideDataEditor"] .headerRow,
.stDataFrame .dvn-scroller .column-header,
[data-testid="stDataFrame"] th {{
    background-color: {T['tbl_hdr']} !important;
    color: {T['tbl_hdr_txt']} !important;
    font-weight: 700 !important; font-size: 12px !important;
    border-bottom: 1px solid {T['border']} !important;
}}
[data-testid="stDataFrame"] td,
.stDataFrame td {{
    color: {T['tbl_txt']} !important; font-size: 13px !important;
    background-color: {T['tbl_bg']} !important;
}}

/* Buttons */
.stButton > button {{
    background: linear-gradient(135deg, {T['accent']}, {T['dim']}) !important;
    color: #0a0d14 !important; font-weight: 700 !important; border: none !important;
    border-radius: 8px !important; padding: 0.4rem 1.1rem !important;
    font-size: 13px !important; transition: all 0.18s !important;
}}
.stButton > button:hover {{
    transform: translateY(-1px);
    box-shadow: 0 4px 14px rgba(0,200,140,.25) !important;
}}

.stSelectbox > div > div {{
    background-color: {T['inp']} !important; border: 1px solid {T['border']} !important;
    border-radius: 8px !important; color: {T['text']} !important; font-size: 13px !important;
}}
.stRadio > div {{ background: transparent !important; border: none !important; padding: 0 !important; }}
.stRadio > label {{ display: none !important; }}
.stRadio span {{ color: {T['text']} !important; }}

.stTabs [data-baseweb="tab-list"] {{
    background-color: {T['card']} !important; border-radius: 10px !important;
    padding: 3px !important; gap: 3px; border: 1px solid {T['border']};
}}
.stTabs [data-baseweb="tab"] {{
    background-color: transparent !important; color: {T['muted']} !important;
    border-radius: 8px !important; font-weight: 500 !important;
    font-size: 13px !important; padding: 0.35rem 0.85rem !important;
}}
.stTabs [aria-selected="true"] {{
    background-color: {T['accent']} !important; color: #0a0d14 !important; font-weight: 700 !important;
}}

.stSuccess {{ background-color: rgba(16,185,129,.1) !important; border: 1px solid rgba(16,185,129,.3) !important; border-radius: 8px !important; }}
.stWarning {{ background-color: rgba(245,158,11,.1)  !important; border: 1px solid rgba(245,158,11,.3)  !important; border-radius: 8px !important; }}
.stError   {{ background-color: rgba(239,68,68,.1)   !important; border: 1px solid rgba(239,68,68,.3)   !important; border-radius: 8px !important; }}
.stInfo    {{ background-color: rgba(59,130,246,.1)  !important; border: 1px solid rgba(59,130,246,.3)  !important; border-radius: 8px !important; }}

.streamlit-expanderHeader {{
    background-color: {T['card']} !important; border: 1px solid {T['border']} !important;
    border-radius: 8px !important; font-weight: 600 !important; font-size: 13px !important;
    color: {T['text']} !important;
}}
.streamlit-expanderContent {{
    background-color: {T['exp']} !important; border: 1px solid {T['border']} !important;
    border-top: none !important;
}}
[data-testid="stFileUploader"] {{
    background-color: {T['card']} !important; border: 2px dashed {T['border']} !important;
    border-radius: 12px !important; padding: 0.75rem !important;
}}
[data-testid="stFileUploader"]:hover {{ border-color: {T['accent']} !important; }}

hr {{ border-color: {T['border']} !important; margin: 0.85rem 0 !important; }}
code {{
    font-family: 'JetBrains Mono', monospace !important;
    background-color: {T['card']} !important; color: {T['accent']} !important;
    padding: 2px 5px; border-radius: 4px; font-size: 12px !important;
}}
::-webkit-scrollbar {{ width: 5px; height: 5px; }}
::-webkit-scrollbar-track {{ background: {T['bg']}; }}
::-webkit-scrollbar-thumb {{ background: {T['border']}; border-radius: 3px; }}
::-webkit-scrollbar-thumb:hover {{ background: {T['accent']}; }}

.slabel {{
    color: {T['muted']} !important; font-size: 0.6rem !important;
    text-transform: uppercase !important; letter-spacing: 1.2px !important;
    font-weight: 700 !important; display: block !important; margin-bottom: 4px !important;
}}

/* ── Selectbox: selected value text ── */
.stSelectbox [data-baseweb="select"] > div {{
    background-color: {T['inp']} !important;
    border-color: {T['border']} !important;
    color: {T['text']} !important;
}}
.stSelectbox [data-baseweb="select"] span,
.stSelectbox [data-baseweb="select"] div {{
    color: {T['text']} !important;
    background-color: transparent !important;
}}

/* ── Selectbox / Multiselect dropdown popup ── */
/* Target every possible BaseUI popup container */
div[data-baseweb="popover"] > div,
div[data-baseweb="popover"] ul,
div[data-baseweb="popover"] li,
[data-baseweb="popover"],
[data-baseweb="menu"],
[role="listbox"],
ul[role="listbox"] {{
    background-color: {T['inp']} !important;
    border: 1px solid {T['accent']} !important;
    border-radius: 8px !important;
    color: {T['text']} !important;
}}

/* Every option item */
[role="option"],
[data-baseweb="option"],
[data-baseweb="menu"] li,
ul[role="listbox"] li {{
    background-color: {T['inp']} !important;
    color: {T['text']} !important;
    font-size: 13px !important;
    padding: 8px 12px !important;
}}

/* Hover state */
[role="option"]:hover,
[data-baseweb="option"]:hover,
[role="option"][aria-selected="true"] {{
    background-color: {T['border']} !important;
    color: {T['accent']} !important;
}}

/* Force all text inside popup to be visible */
div[data-baseweb="popover"] *,
div[data-baseweb="popover"] span,
div[data-baseweb="popover"] p,
div[data-baseweb="popover"] div {{
    background-color: transparent !important;
    color: {T['text']} !important;
}}

/* Override the white default that Safari/Chrome applies */
div[class*="menu"],
div[class*="dropdown"],
div[class*="Listbox"],
div[class*="popover"] {{
    background-color: {T['inp']} !important;
    color: {T['text']} !important;
}}

/* ── Multiselect ── */
.stMultiSelect [data-baseweb="select"] > div {{
    background-color: {T['inp']} !important;
    border-color: {T['border']} !important;
}}
.stMultiSelect [data-baseweb="tag"] {{
    background-color: {T['accent']} !important;
    color: #0a0d14 !important;
}}
.stMultiSelect span {{
    color: {T['text']} !important;
}}

/* ── Radio button labels ── */
.stRadio [data-testid="stWidgetLabel"] p,
.stRadio label p,
.stRadio div[role="radiogroup"] label span p,
.stRadio div[role="radiogroup"] label {{
    color: {T['text']} !important;
    font-size: 13px !important;
}}

/* ── Text input ── */
.stTextInput input {{
    background-color: {T['inp']} !important;
    border: 1px solid {T['border']} !important;
    border-radius: 8px !important;
    color: {T['text']} !important;
    font-size: 13px !important;
}}
.stTextInput input::placeholder {{ color: {T['muted']} !important; }}

/* ── Number input ── */
.stNumberInput input {{
    background-color: {T['inp']} !important;
    border: 1px solid {T['border']} !important;
    color: {T['text']} !important;
}}

/* ── Expander header text ── */
details summary p,
.streamlit-expanderHeader p,
.streamlit-expanderHeader span {{
    color: {T['text']} !important;
}}

/* ── st.code block ── */
.stCodeBlock, pre, pre code {{
    background-color: {T['card']} !important;
    color: {T['accent']} !important;
    border: 1px solid {T['border']} !important;
    border-radius: 8px !important;
}}

/* ── Caption / small text ── */
.stCaption, small {{
    color: {T['muted']} !important;
    font-size: 12px !important;
}}

/* ── Download button ── */
.stDownloadButton > button {{
    background: transparent !important;
    color: {T['accent']} !important;
    border: 1px solid {T['accent']} !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 13px !important;
}}
.stDownloadButton > button:hover {{
    background: {T['accent']} !important;
    color: #0a0d14 !important;
}}

/* ── Checkbox ── */
.stCheckbox label span {{ color: {T['text']} !important; }}
</style>
""", unsafe_allow_html=True)

# ── Force dropdown styles via JS (portaled elements CSS fix) ─────────
st.markdown('''
<script>
(function injectDropdownStyles() {
    var id = 'ds-dropdown-fix';
    if (document.getElementById(id)) return;
    var s = document.createElement('style');
    s.id = id;
    s.innerHTML = `
        div[data-baseweb="popover"] {
            background-color: #1a2235 !important;
            border: 1px solid #00e5a0 !important;
            border-radius: 8px !important;
        }
        div[data-baseweb="popover"] * {
            background-color: #1a2235 !important;
            color: #f1f5f9 !important;
        }
        [role="option"] {
            background-color: #1a2235 !important;
            color: #f1f5f9 !important;
            padding: 8px 12px !important;
        }
        [role="option"]:hover,
        [role="option"][aria-selected="true"] {
            background-color: #1e293b !important;
            color: #00e5a0 !important;
        }
        [data-baseweb="select"] > div {
            background-color: #1a2235 !important;
            border-color: #1e293b !important;
        }
        [data-baseweb="select"] span {
            color: #f1f5f9 !important;
        }
    `;
    document.head.appendChild(s);
    // Also watch for dynamically added popover elements
    var obs = new MutationObserver(function() {
        var els = document.querySelectorAll('[data-baseweb="popover"] *');
        els.forEach(function(el) {
            el.style.setProperty('background-color', '#1a2235', 'important');
            el.style.setProperty('color', '#f1f5f9', 'important');
        });
    });
    obs.observe(document.body, {childList: true, subtree: true});
})()
</script>
''', unsafe_allow_html=True)

# ── JS: Force dropdown popup styles (portaled elements bypass CSS) ────
_dd_bg  = T['inp']
_dd_txt = T['text']
_dd_acc = T['accent']
_dd_hov = T['border']
st.markdown(f"""
<script>
(function(){{
    var styleId = 'ds-dd-fix';
    var existing = document.getElementById(styleId);
    if (existing) existing.remove();
    var s = document.createElement('style');
    s.id = styleId;
    s.textContent = `
        div[data-baseweb="popover"],
        div[data-baseweb="popover"] > div,
        [data-baseweb="menu"] {{
            background-color: {_dd_bg} !important;
            border: 1px solid {_dd_acc} !important;
            border-radius: 8px !important;
        }}
        [role="option"] {{
            background-color: {_dd_bg} !important;
            color: {_dd_txt} !important;
            font-size: 13px !important;
        }}
        [role="option"]:hover,
        [role="option"][aria-selected="true"],
        [role="option"][data-highlighted="true"] {{
            background-color: {_dd_hov} !important;
            color: {_dd_acc} !important;
        }}
        div[data-baseweb="popover"] span,
        div[data-baseweb="popover"] p {{
            color: {_dd_txt} !important;
        }}
        [data-baseweb="select"] > div {{
            background-color: {_dd_bg} !important;
        }}
        [data-baseweb="select"] span {{
            color: {_dd_txt} !important;
        }}
    `;
    document.head.appendChild(s);
    // MutationObserver ensures styles apply to dynamically injected portal elements
    var obs = new MutationObserver(function(muts) {{
        muts.forEach(function(m) {{
            m.addedNodes.forEach(function(node) {{
                if (node.nodeType === 1) {{
                    var pop = node.querySelector
                        ? node.querySelector('[data-baseweb="popover"]')
                        : null;
                    if (pop || (node.dataset && node.dataset.baseweb === 'popover')) {{
                        var items = (pop || node).querySelectorAll('[role="option"]');
                        items.forEach(function(item) {{
                            item.style.setProperty('background-color', '{_dd_bg}', 'important');
                            item.style.setProperty('color', '{_dd_txt}', 'important');
                        }});
                    }}
                }}
            }});
        }});
    }});
    obs.observe(document.body, {{childList: true, subtree: true}});
}})();
</script>
""", unsafe_allow_html=True)

# ── Imports ───────────────────────────────────────────────────────────
from SRC.enhancement_module   import show_future_enhancement
from SRC.model_recommendation import show_model_recommendation
from SRC.eda_module            import basic_overview, feature_analysis, advanced_analysis
from SRC.data_cleaner_module   import show_data_cleaner

# ── Demo data ─────────────────────────────────────────────────────────
def create_demo_data():
    np.random.seed(42)
    clf = os.path.join("demo_data","classification")
    reg = os.path.join("demo_data","regression")
    os.makedirs(clf, exist_ok=True); os.makedirs(reg, exist_ok=True)
    if not os.path.exists(os.path.join(clf,"iris_demo.csv")):
        n=150
        pd.DataFrame({"sepal_length":np.round(np.random.normal(5.8,.8,n),1),
            "sepal_width":np.round(np.random.normal(3.0,.4,n),1),
            "petal_length":np.round(np.random.normal(3.7,1.7,n),1),
            "petal_width":np.round(np.random.normal(1.2,.7,n),1),
            "species":np.random.choice(["setosa","versicolor","virginica"],n)
        }).to_csv(os.path.join(clf,"iris_demo.csv"),index=False)
    if not os.path.exists(os.path.join(clf,"titanic_demo.csv")):
        n=200
        pd.DataFrame({"age":np.round(np.random.normal(30,12,n).clip(1,80),0),
            "fare":np.round(np.random.exponential(30,n),2),
            "pclass":np.random.choice([1,2,3],n),"sex":np.random.choice(["male","female"],n),
            "siblings":np.random.choice([0,1,2,3],n,p=[.6,.25,.1,.05]),
            "survived":np.random.choice([0,1],n,p=[.6,.4])
        }).to_csv(os.path.join(clf,"titanic_demo.csv"),index=False)
    if not os.path.exists(os.path.join(reg,"housing_demo.csv")):
        n=200; sz=np.random.randint(600,3500,n); rm=np.random.randint(1,6,n); ag=np.random.randint(1,50,n)
        pr=np.round(sz*120+rm*8000-ag*500+np.random.normal(0,15000,n),0)
        pd.DataFrame({"size_sqft":sz,"num_rooms":rm,"house_age":ag,
            "neighborhood":np.random.choice(["urban","suburban","rural"],n),
            "price":pr.clip(50000,800000)}).to_csv(os.path.join(reg,"housing_demo.csv"),index=False)
    if not os.path.exists(os.path.join(reg,"salary_demo.csv")):
        n=150; exp=np.random.randint(0,20,n); edu=np.random.choice(["bachelor","master","phd"],n)
        bonus={"bachelor":0,"master":10000,"phd":20000}
        sal=np.round(35000+exp*3000+np.array([bonus[e] for e in edu])+np.random.normal(0,5000,n),0)
        pd.DataFrame({"years_experience":exp,"education":edu,
            "department":np.random.choice(["engineering","marketing","finance","hr"],n),
            "age":(22+exp+np.random.randint(0,5,n)).clip(22,60),
            "salary":sal.clip(25000,150000)}).to_csv(os.path.join(reg,"salary_demo.csv"),index=False)

create_demo_data()

# ── Helpers ───────────────────────────────────────────────────────────
def resolve_target(df):
    t = st.session_state.get("target_col")
    if t and t in df.columns: return t
    fb = df.columns[-1]; st.session_state.target_col = fb; return fb

def get_active_df():
    c = st.session_state.get("cleaned_df")
    return c if c is not None else st.session_state.df

# ── SIDEBAR ───────────────────────────────────────────────────────────
with st.sidebar:

    st.markdown(
        f"<div style='margin-bottom:0.75rem;'>"
        f"<div style='font-size:1rem;font-weight:700;color:{T['accent']};'>DataSense Advisor</div>"
        f"<div style='font-size:0.7rem;color:{T['muted']};'>ML Dataset Intelligence</div>"
        f"</div>",
        unsafe_allow_html=True
    )

    # Theme toggle
    is_dark = st.session_state.theme == "dark"
    if st.button(
        ("☀️  Light Mode" if is_dark else "🌙  Dark Mode"),
        key="theme_toggle", use_container_width=True
    ):
        st.session_state.theme = "light" if is_dark else "dark"
        st.rerun()

    st.markdown("---")

    # Navigation — always show all 5 pages, no resetting
    st.markdown("<span class='slabel'>Navigation</span>", unsafe_allow_html=True)
    df = st.session_state.df
    ALL_PAGES = ["Load Dataset", "EDA", "Data Cleaner", "Models", "Enhancements"]
    page_sel  = st.radio("nav", ALL_PAGES, key="nav_radio", label_visibility="collapsed")
    if df is None and page_sel != "Load Dataset":
        st.warning("Please load a dataset first.")

    # Dataset info section
    if df is not None:
        st.markdown("---")
        st.markdown("<span class='slabel'>Dataset</span>", unsafe_allow_html=True)
        cleaned = st.session_state.cleaned_df is not None
        status  = "Cleaned" if cleaned else "Raw"
        name    = st.session_state.dataset_name
        st.markdown(
            f"<div style='color:{T['accent']};font-size:0.78rem;font-weight:600;margin-bottom:2px;'>{status}</div>"
            f"<div style='color:{T['muted']};font-size:0.72rem;overflow:hidden;text-overflow:ellipsis;"
            f"white-space:nowrap;max-width:190px;margin-bottom:6px;' title='{name}'>{name}</div>",
            unsafe_allow_html=True
        )
        active = get_active_df()
        c1, c2 = st.columns(2)
        c1.metric("Rows", f"{active.shape[0]:,}")
        c2.metric("Cols", active.shape[1])
        mp = round(active.isnull().sum().sum()/(active.shape[0]*active.shape[1])*100, 1)
        st.caption(f"Missing: {mp}%")

        st.markdown("---")
        st.markdown("<span class='slabel'>Target Column</span>", unsafe_allow_html=True)
        current = resolve_target(df)
        cols    = list(df.columns)
        idx     = cols.index(current) if current in cols else len(cols)-1
        new_t   = st.selectbox("Target", df.columns, index=idx,
                               key="sidebar_target", label_visibility="collapsed")
        if new_t != current:
            st.session_state.target_col = new_t
            st.rerun()

        st.markdown("---")
        if cleaned:
            if st.button("Reset to Raw Data", use_container_width=True, key="reset_raw"):
                st.session_state.cleaned_df = None
                st.rerun()
        if st.button("Load New Dataset", use_container_width=True, key="load_new"):
            st.session_state.df=None; st.session_state.cleaned_df=None
            st.session_state.dataset_name=""; st.session_state.target_col=None
            st.rerun()

# ── Refresh locals ────────────────────────────────────────────────────
df       = st.session_state.df
page_sel = st.session_state.get("nav_radio", "Load Dataset")
PAGE_MAP = {
    "Load Dataset": "upload", "EDA": "eda",
    "Data Cleaner": "cleaner", "Models": "models", "Enhancements": "enhance",
}
page = PAGE_MAP.get(page_sel, "upload")
if df is None and page != "upload":
    page = "upload"

# ── PAGES ─────────────────────────────────────────────────────────────
if page == "upload":
    st.title("DataSense Advisor")
    st.markdown(
        f"<p style='color:{T['muted']};font-size:0.9rem;margin-top:-0.3rem;'>"
        "Analyze your dataset and get machine learning model recommendations.</p>",
        unsafe_allow_html=True
    )
    st.markdown("---")
    col_l, col_r = st.columns(2, gap="large")
    with col_l:
        st.subheader("Upload CSV")
        f = st.file_uploader("CSV file", type=["csv"], label_visibility="collapsed")
        if f is not None:
            new_df = pd.read_csv(f)
            st.session_state.df=new_df; st.session_state.cleaned_df=None
            st.session_state.dataset_name=f.name; st.session_state.target_col=new_df.columns[-1]
            st.success(f"{f.name} loaded — {new_df.shape[0]:,} rows x {new_df.shape[1]} columns")
            st.rerun()
    with col_r:
        st.subheader("Demo Dataset")
        dtype = st.selectbox("Type", ["classification","regression"], key="demo_type")
        fpath = os.path.join("demo_data", dtype)
        if os.path.exists(fpath):
            files = [x for x in os.listdir(fpath) if x.endswith(".csv")]
            if files:
                chosen = st.selectbox("File", files, key="demo_pick")
                if st.button("Load Demo", key="load_demo"):
                    new_df = pd.read_csv(os.path.join(fpath,chosen))
                    st.session_state.df=new_df; st.session_state.cleaned_df=None
                    st.session_state.dataset_name=chosen; st.session_state.target_col=new_df.columns[-1]
                    st.rerun()

elif page == "eda":
    target=resolve_target(df); active_df=get_active_df()
    if st.session_state.cleaned_df is not None:
        st.info("Viewing cleaned dataset.")
    st.header("Exploratory Data Analysis")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Rows",    f"{active_df.shape[0]:,}")
    c2.metric("Columns", active_df.shape[1])
    c3.metric("Missing", f"{round(active_df.isnull().sum().sum()/(active_df.shape[0]*active_df.shape[1])*100,1)}%")
    c4.metric("Target",  target)
    st.markdown("---")
    try:
        t1,t2,t3 = st.tabs(["Basic Overview","Feature Analysis","Advanced Analysis"])
        with t1: basic_overview(active_df)
        with t2: feature_analysis(active_df)
        with t3: advanced_analysis(active_df, target)
    except Exception as e: st.error(f"EDA error: {e}"); st.exception(e)

elif page == "cleaner":
    try: show_data_cleaner()
    except Exception as e: st.error(f"Data Cleaner error: {e}"); st.exception(e)

elif page == "models":
    active_df=get_active_df(); target=resolve_target(df)
    if st.session_state.cleaned_df is not None: st.info("Using cleaned dataset.")
    try: show_model_recommendation(active_df, target)
    except Exception as e: st.error(f"Models error: {e}"); st.exception(e)

elif page == "enhance":
    active_df=get_active_df(); target=resolve_target(df)
    try: show_future_enhancement(active_df, target)
    except Exception as e: st.error(f"Enhancements error: {e}"); st.exception(e)


