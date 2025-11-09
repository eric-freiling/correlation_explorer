# app_min.py
from __future__ import annotations
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
#import statsmodels.api as sm

# =========================
# Config (replace these)
# =========================

DEFAULT_LABS = "snapshots/labs.csv"
DEFAULT_TRACKERS1 = "snapshots/trackers1.csv"
DEFAULT_TRACKERS2 = "snapshots/trackers2.csv"
DEFAULT_SURVEYS = "snapshots/surveys.csv"
DEFAULT_DEFINITIONS = "snapshots/defs.csv"

# ---- Safe defaults to avoid NameError on first render ----
parent_a = None
parent_b = None
src_a = None
src_b = None

# sensible defaults for step 3 controls (will be overwritten when sidebar renders)
reducer = "mean"
min_weeks = 1
min_subjects = 3
run_corr = False

# =========================
# Page
# =========================
st.session_state.setdefault("corr_ready", False)   # True only after a successful Run
st.session_state.setdefault("corr_result", False)  # you asked for False when dirty
st.session_state.setdefault("corr_cache", None)    # holds (heat, Aw, Bw, meta)

st.set_page_config(
    page_title="Correlation Explorer",
    page_icon="images/icon.png",
    layout="wide"
)
st.title("Correlation Explorer")
st.logo(
    "images/biocanic-logo.png", 
    
)
# =========================
# Utilities
# =========================

def _mark_corr_dirty():
    st.session_state["corr_ready"]  = False
    st.session_state["corr_result"] = False
    st.session_state["corr_cache"]  = None

def get_all_orgs():
    all_orgs = []
    df = pd.read_csv(DEFAULT_LABS)
    all_orgs = list(df['org_id'].unique())
    all_orgs.sort()
    # all_orgs = [
    #     '5f776b6e87607d459f367393',
    #     '637580f088c992578aefd098'
    # ]
    return all_orgs

@st.cache_data(show_spinner=True)
def read_csv_org(path: str, org_id: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df['org_id'] == org_id].reset_index(drop=True)
    return df

def _first_non_null(s):
    """Return first non-null value in series, or None if all null."""
    return s.dropna().iloc[0] if s.dropna().size > 0 else None

# -------------------------------
# Lab definitions (pandas version)
# -------------------------------
def enrich_and_filter_labs(labs_df: pd.DataFrame, defs_df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only labs whose (lab_id, measurement) exists in defs.
    Attach lab_name, test_name, valuetype.
    Log-transform 'value' when valuetype == 'scientific' and value > 0.
    Adds 'value_raw' column (pre-transform).
    """
    required = {"lab_id", "measurement"}
    if not required.issubset(labs_df.columns):
        # create empty columns if missing to keep shape predictable
        for c in required - set(labs_df.columns):
            labs_df[c] = pd.Series(pd.NA, index=labs_df.index, dtype="string")

    # Ensure join keys are strings (consistent casing/trim)
    left = labs_df.copy()
    left["lab_id"] = left["lab_id"].astype("string").str.strip()
    left["measurement"] = left["measurement"].astype("string").str.strip()

    right = defs_df.copy()
    right["lab_def_id"] = right["lab_def_id"].astype("string").str.strip()
    right["measurementname"] = right["measurementname"].astype("string").str.strip()

    merged = left.merge(
        right,
        left_on=["lab_id", "measurement"],
        right_on=["lab_def_id", "measurementname"],
        how="inner",
        suffixes=("", "_def"),
    )
    
    v = pd.to_numeric(merged.get("value"), errors="coerce")
    merged["value_raw"] = v
    
    vt = merged.get("valuetype")
    vt = vt.astype("string").str.lower().str.strip() if vt is not None else pd.Series("", index=merged.index, dtype="string")
    need_log = (vt == "scientific") & (v > 0)
    merged["value"] = np.where(need_log, np.log10(v), v)
    
    labs_cols = [c for c in labs_df.columns if c != "value_type"]
    keep_cols = labs_cols + ["lab_name", "test_name", "valuetype", "value_raw", "value"]
    keep_cols = [c for c in keep_cols if c in merged.columns]
    keep_cols = list(set(keep_cols))
    out = merged[keep_cols].copy()

    return out

@st.cache_data(show_spinner=True)
def load_lab_definitions_csv(path: str) -> pd.DataFrame:
    """
    Read definitions parquet and normalize to:
      - lab_def_id
      - measurementname
      - lab_name
      - test_name
      - valuetype
    Handles camelCase and snake_case variants.
    """
    df = pd.read_csv(path)

    # lowercase columns for easier matching
    df.columns = [c.lower() for c in df.columns]

    # Map possible variants → canonical
    rename = {
        "id": "lab_def_id",
        "measurementname": "measurementname",
        "name": "lab_name",
        "testname": "test_name",
        "valuetype": "valuetype",
    }
    # also accept camelCase sources that survived lowercasing
    # (already lowercased so no special handling; above covers them)

    # If some expected columns aren't present but variants are, fix them
    for src, dst in list(rename.items()):
        if src not in df.columns:
            # nothing to do; if a variant existed it would already be lower == src
            pass

    df = df.rename(columns=rename)

    needed = ["lab_def_id", "measurementname", "lab_name", "test_name", "valuetype"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Definitions parquet missing columns: {missing}")

    return df[needed].drop_duplicates()

def filter_labs_with_definitions_pandas(
    labs_df: pd.DataFrame,
    defs_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Keep only labs where (lab_id, measurement) pair exists in definitions.
    Adds 'lab_name' and 'test_name' columns.
    Also (optional) logs:
      - unmatched lab_ids
      - invalid (lab_id, measurement) pairs for known lab_ids
    """
    # Expect labs_df to be RAW labs (lowercased) with columns: lab_id, measurement, ...
    if not {"lab_id", "measurement"}.issubset(set(labs_df.columns)):
        raise ValueError("labs_df must include 'lab_id' and 'measurement' columns (raw labs).")

    defs_sel = defs_df[["lab_def_id", "measurementname", "lab_name", "test_name"]].drop_duplicates()

    # Inner join on (lab_id, measurement) -> valid rows
    valid = labs_df.merge(
        defs_sel,
        left_on=["lab_id", "measurement"],
        right_on=["lab_def_id", "measurementname"],
        how="inner",
        suffixes=("", "_def"),
    )

    # Final selection: original labs columns + names from defs
    out_cols = list(labs_df.columns) + ["lab_name", "test_name"]
    return valid[out_cols]

def adapt_labs(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure we have Series for these columns
    meas = df["measurement"] if "measurement" in df.columns else pd.Series("", index=df.index, dtype="string")
    cat  = df["category"] if "category" in df.columns else pd.Series("", index=df.index, dtype="string")

    meas = meas.astype("string").fillna("").str.strip()
    cat  = cat.astype("string").fillna("").str.strip()

    metric = np.where(cat != "", "lab:" + cat + ":" + meas, "lab:" + meas)

    out = pd.DataFrame(index=df.index)
    out["org_id"]       = df.get("org_id")
    out["subject_id"]   = df.get("user_id")
    out["metric_id"]    = metric                     # direct assign 1-D array
    out["value"]        = pd.to_numeric(df.get("value"), errors="coerce")
    out["observed_at"]  = pd.to_datetime(df.get("date"), errors="coerce", utc=True)
    out["source_id"]    = "labs"
    out["display_name"] = meas
    out["lab_id"]       = df.get("lab_id")
    out["lab_name"]     = df.get("lab_name")
    out["test_name"]    = df.get("test_name")
    out["value_type"]    = df.get("valuetype")
    out["value_raw"]    = pd.to_numeric(df.get("value_raw"), errors="coerce") if "value_raw" in df else pd.NA
    return out

def adapt_surveys(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    out["org_id"]      = df.get("org_id")
    out["subject_id"]  = df.get("user_id")
    qid                = df.get("question_id").astype("string")
    out["metric_id"]   = "survey:" + qid.fillna("").str.strip()
    out["value"]       = pd.to_numeric(df.get("answer_value"), errors="coerce")
    out["observed_at"] = pd.to_datetime(df.get("date_submitted"), errors="coerce", utc=True)\
                           .fillna(pd.to_datetime(df.get("created_at"), errors="coerce", utc=True))
    out["source_id"]   = "surveys"
    out["display_name"]= df.get("question_text")
    out["survey_id"]   = df.get("survey_id")
    out["group_name"]  = df.get("group_name")
    return out

def adapt_trackers(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    out["org_id"]      = df.get("org_id")
    out["subject_id"]  = df.get("user_id")
    name               = df.get("name").astype("string").fillna("")
    ttype              = df.get("tracker_type").astype("string").fillna("")
    out["tracker_type"] = ttype
    metric = np.where(ttype != "",
                      "tracker:" + ttype.str.strip() + ":" + name.str.strip(),
                      "tracker:" + name.str.strip())
    out["metric_id"]   = metric
    val                = pd.to_numeric(df.get("data_value"), errors="coerce")
    div                = pd.to_numeric(df.get("divide_by"), errors="coerce")
    out["value"]       = np.where(div.notna() & (div != 0), val / div, val)
    out["observed_at"] = pd.to_datetime(df.get("data_date"), errors="coerce", utc=True)\
                           .fillna(pd.to_datetime(df.get("created_at"), errors="coerce", utc=True))
    out["source_id"]    = "trackers"
    out["display_name"] = df.get("name")
    out["wearables_id"] = df.get("wearables_id")
    out["template_id"]  = df.get("template_id")
    return out

def non_empty_str(s: pd.Series) -> pd.Series:
    return s.astype("string").str.strip().replace("", np.nan)

def filter_trackers_minimal(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep ONLY tracker rows that have a non-empty wearables_id and a valid numeric value.
    """
    df = df.copy()
    has_wid = non_empty_str(df.get("wearables_id")).notna()
    ok_val  = df["value"].notna() & np.isfinite(df["value"])
    return df[has_wid & ok_val]

def scatter_with_fit_ci(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str = "",
    ci: float = 0.95,
    id_col: str | None = None,
    time_col: str | None = None,   # <-- NEW
):
    dcols = [x_col, y_col] + ([id_col] if id_col else []) + ([time_col] if time_col else [])
    d = df[dcols].dropna()
    if len(d) < 2:
        return go.Figure()

    # OLS y ~ x
    x = d[x_col].to_numpy(dtype=float)
    y = d[y_col].to_numpy(dtype=float)
    X = np.c_[np.ones_like(x), x]
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    xs = np.linspace(x.min(), x.max(), 200)
    yhat = beta[0] + beta[1] * xs

    resid = y - (beta[0] + beta[1]*x)
    s = np.std(resid, ddof=2)
    z = 1.96
    upper = yhat + z*s
    lower = yhat - z*s

    # customdata & hover
    custom_cols = []
    hover_lines = []
    if id_col:
        custom_cols.append(d[id_col].astype(str).to_numpy())
        hover_lines.append("Subject: %{customdata[0]}")
    if time_col:
        # render as ISO date if it's datetime-like
        tvals = pd.to_datetime(d[time_col], errors="coerce")
        tshow = np.where(tvals.notna(), tvals.dt.strftime("%Y-%m-%d").fillna(d[time_col].astype(str)), d[time_col].astype(str))
        custom_cols.append(np.array(tshow))
        idx = len(custom_cols) - 1
        hover_lines.append(f"Week: %{{customdata[{idx}]}}")

    customdata = np.stack(custom_cols, axis=-1) if custom_cols else None
    hovertemplate = ("<br>".join(hover_lines) + ("<br>" if hover_lines else "") +
                     f"{x_col}: %{{x}}<br>{y_col}: %{{y}}<extra></extra>")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=d[x_col], y=d[y_col],
        mode="markers",
        name="Points",
        opacity=0.75,
        marker=dict(size=7),
        customdata=customdata if custom_cols else None,
        hovertemplate=hovertemplate,
    ))

    fig.add_trace(go.Scatter(
        x=np.concatenate([xs, xs[::-1]]),
        y=np.concatenate([upper, lower[::-1]]),
        fill='toself',
        fillcolor='rgba(255, 0, 0, 0.12)',
        line=dict(color='rgba(255,0,0,0)'),
        hoverinfo='skip',
        name=f"{int(ci*100)}% band",
        showlegend=True,
    ))

    fig.add_trace(go.Scatter(
        x=xs, y=yhat,
        mode="lines",
        line=dict(color='red', width=2),
        name="OLS fit",
        hoverinfo='skip',
    ))

    try:
        r = np.corrcoef(x, y)[0,1]
        subtitle = f" (r = {r:.3f})"
    except Exception:
        subtitle = ""

    fig.update_layout(
        title=(title or f"Scatter: {x_col} vs {y_col}") + subtitle,
        xaxis_title=x_col,
        yaxis_title=y_col,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig

def _weekly_features_key(org_id, src, parent, reducer):
    return ("weekly", str(org_id), str(src), str(parent), str(reducer))

# =========================
# Step 1: Load by org (FORM, sentinel-safe)
# =========================
st.sidebar.header("Step 1 — Load data")

all_orgs = get_all_orgs()
if not all_orgs:
    st.error("No org_id partitions discovered under the configured S3 roots.")
    st.stop()

SENTINEL = "— Select an org —"
org_choices = [SENTINEL] + all_orgs

with st.sidebar.form("load_form", clear_on_submit=False):
    org_choice = st.selectbox(
        "Org ID",
        options=org_choices,
        index=0,  # always renders a valid default (the sentinel)
        key="org_select",
    )
    load = st.form_submit_button("📥 Load data", use_container_width=True)  # never disabled

def load_org(org: str):
    labs_raw     = read_csv_org(DEFAULT_LABS, org)
    trackers_raw1 = read_csv_org(DEFAULT_TRACKERS1, org)
    trackers_raw2 = read_csv_org(DEFAULT_TRACKERS2, org)
    trackers_raw = pd.concat([trackers_raw1, trackers_raw2])
    surveys_raw  = read_csv_org(DEFAULT_SURVEYS, org)
    for df in (labs_raw, trackers_raw, surveys_raw):
        df.columns = [c.lower() for c in df.columns]
        if "org_id" not in df:
            df["org_id"] = org
    defs_df = load_lab_definitions_csv(DEFAULT_DEFINITIONS)
    labs_enriched = enrich_and_filter_labs(labs_raw, defs_df)
    labs     = adapt_labs(labs_enriched)
    trackers = filter_trackers_minimal(adapt_trackers(trackers_raw))
    surveys  = adapt_surveys(surveys_raw)
    fact = pd.concat([labs, trackers, surveys], ignore_index=True, sort=False)
    fact = fact[(fact["subject_id"].notna()) & (fact["observed_at"].notna())]
    return labs, trackers, surveys, fact

if load:
    if org_choice == SENTINEL:
        st.sidebar.error("Please select an org first.")
    else:
        with st.spinner("Loading & normalizing…"):
            try:
                labs, trackers, surveys, fact = load_org(org_choice.strip())
                # Keep originals so we can re-filter without hitting S3 again
                st.session_state["labs_all"]     = labs.copy()
                st.session_state["trackers_all"] = trackers.copy()
                st.session_state["surveys_all"]  = surveys.copy()
                st.session_state["fact_all"]     = fact.copy()

                # Working (possibly filtered) frames that the rest of the app uses
                st.session_state["labs"]     = labs.copy()
                st.session_state["trackers"] = trackers.copy()
                st.session_state["surveys"]  = surveys.copy()
                st.session_state["fact"]     = fact.copy()
                _mark_corr_dirty()
                # st.sidebar.success("Loaded!")
            except Exception as e:
                st.sidebar.error(f"Load failed: {e}")

# ---------- Step 1.5 — Filters (applies to working data) ----------
if "fact_all" in st.session_state:
    st.sidebar.markdown("---")
    st.sidebar.header("Step 2 — Filters")

    def _apply_filters_df(df, user_ids, start_ts, end_ts):
        if df is None or df.empty:
            return df
        out = df
        if user_ids:  # user_ids is a list of strings
            out = out[out["subject_id"].astype(str).isin(user_ids)]
        if start_ts is not None and end_ts is not None and "observed_at" in out.columns:
            ts = pd.to_datetime(out["observed_at"], utc=True, errors="coerce")
            out = out[(ts >= start_ts) & (ts <= end_ts)]
        return out

    fact_all = st.session_state["fact_all"]

    # Build subject choices from ALL data
    all_subjects = sorted(fact_all["subject_id"].dropna().astype(str).unique())

    user_ids = st.sidebar.multiselect(
        "Users (subject_id)",
        options=all_subjects,
        default=[],
        help="Leave empty to keep all users.",
        on_change=_mark_corr_dirty
    )

    enable_date = st.sidebar.checkbox("Filter by date", value=False)
    start_ts = end_ts = None
    if enable_date:
        min_dt = pd.to_datetime(fact_all["observed_at"], utc=True).min()
        max_dt = pd.to_datetime(fact_all["observed_at"], utc=True).max()
        # Fallbacks if empty
        if pd.isna(min_dt): min_dt = pd.Timestamp("2000-01-01", tz="UTC")
        if pd.isna(max_dt): max_dt = pd.Timestamp.utcnow()
        picked = st.sidebar.date_input(
            "Date range (inclusive)",
            value=(min_dt.date(), max_dt.date()),
            on_change=_mark_corr_dirty
        )
        if isinstance(picked, (list, tuple)) and len(picked) == 2:
            start_date, end_date = picked
        else:
            start_date = picked
            end_date = picked
        start_ts = pd.Timestamp(start_date).tz_localize("UTC")
        end_ts = (pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)).tz_localize("UTC")

    c1, c2 = st.sidebar.columns(2)
    apply_filters = c1.button("Apply filters", use_container_width=True)
    clear_filters = c2.button("Clear", use_container_width=True)

    if clear_filters:
        # restore working copies to all data
        st.session_state["labs"]     = st.session_state["labs_all"].copy()
        st.session_state["trackers"] = st.session_state["trackers_all"].copy()
        st.session_state["surveys"]  = st.session_state["surveys_all"].copy()
        st.session_state["fact"]     = st.session_state["fact_all"].copy()
        st.sidebar.success("Filters cleared.")

    if apply_filters:
        labs_all     = st.session_state["labs_all"]
        trackers_all = st.session_state["trackers_all"]
        surveys_all  = st.session_state["surveys_all"]
        # Apply to each, then recompose fact for downstream needs
        labs_f     = _apply_filters_df(labs_all, user_ids, start_ts, end_ts)
        trackers_f = _apply_filters_df(trackers_all, user_ids, start_ts, end_ts)
        surveys_f  = _apply_filters_df(surveys_all, user_ids, start_ts, end_ts)
        fact_f     = pd.concat([labs_f, trackers_f, surveys_f], ignore_index=True, sort=False)

        st.session_state["labs"]     = labs_f
        st.session_state["trackers"] = trackers_f
        st.session_state["surveys"]  = surveys_f
        st.session_state["fact"]     = fact_f
        st.sidebar.success("Filters applied.")

# Show a tiny preview
if "fact" in st.session_state:
    fact = st.session_state["fact"]
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Labs",     len(st.session_state["labs"]))
    with c2: st.metric("Trackers", len(st.session_state["trackers"]))
    with c3: st.metric("Surveys",  len(st.session_state["surveys"]))
    with c4: st.metric("Total",    f"{len(fact):,}")
    #st.dataframe(fact.head(10), use_container_width=True)

# =========================
# Step 2: Catalogs + picks
# =========================
def _cat_labs(df: pd.DataFrame) -> pd.DataFrame:
    base = (
        df.groupby(["lab_id", "lab_name", "test_name"], dropna=False)
          .agg(n_subjects=("subject_id", "nunique"))
          .reset_index()
    )
    def _mk_label(row):
        parts = [str(row.get("lab_name") or "").strip(), str(row.get("test_name") or "").strip()]
        parts = [p for p in parts if p]
        return " – ".join(parts) if parts else f"lab_id:{row['lab_id']}"
    base["label"] = base.apply(_mk_label, axis=1)
    out = base[["lab_id", "label", "n_subjects"]]
    out["label"] = out["label"].astype(str) + " · n=" + out["n_subjects"].astype(str)
    out = out.sort_values(by="label").reset_index(drop=True)
    return out

def _cat_surveys(df: pd.DataFrame) -> pd.DataFrame:
    # one row per survey_id, with a simple label that is the ID itself
    out = (
        df.groupby("survey_id", dropna=False)
          .agg(n_subjects=("subject_id", "nunique"))
          .reset_index()
    )
    # what appears in the survey selectbox
    out["label_id"] = out["survey_id"].astype(str) + " · n=" + out["n_subjects"].astype(str)
    return out[["survey_id", "label_id", "n_subjects"]]

def _cat_tracker_types(df: pd.DataFrame) -> pd.DataFrame:
    t = df.copy()
    #st.dataframe(t.head())
    t["tracker_type"] = t["tracker_type"].astype("string").str.strip()
    out = (
        t.groupby("tracker_type", dropna=False)
         .agg(n_subjects=("subject_id","nunique"))
         .reset_index()
    )
    label = np.where(out["tracker_type"].notna() & (out["tracker_type"] != ""),
                     out["tracker_type"], "(unknown type)")
    out["label"] = label + " · n=" + out["n_subjects"].astype(str)
    out = out.sort_values("label").reset_index(drop=True)
    return out[["tracker_type","label","n_subjects"]]


def _corr_from_overlapping_pairs_allpoints(
    Aw: pd.DataFrame, Bw: pd.DataFrame, feat_a: str, feat_b: str,
    min_points: int = 10, min_subjects: int = 3
) -> float:
    """
    Pearson r across *all* overlapping subject-weeks for (feat_a, feat_b).
    Requires at least `min_points` rows total and `min_subjects` unique subjects.
    """
    a = Aw[Aw["feature_label"].astype(str) == str(feat_a)][["subject_id","time_key","value"]].rename(columns={"value":"A"})
    b = Bw[Bw["feature_label"].astype(str) == str(feat_b)][["subject_id","time_key","value"]].rename(columns={"value":"B"})
    pairs = pd.merge(a, b, on=["subject_id","time_key"], how="inner").dropna()
    if pairs.empty:
        return np.nan
    if pairs["subject_id"].nunique() < min_subjects or len(pairs) < min_points:
        return np.nan
    return pairs[["A","B"]].corr(method="pearson").iloc[0,1]

def _heatmap_overlapping_weeks(Aw: pd.DataFrame, Bw: pd.DataFrame,
                               min_points: int = 10, min_subjects: int = 3) -> pd.DataFrame:
    A_feats = sorted(Aw["feature_label"].dropna().astype(str).unique())
    B_feats = sorted(Bw["feature_label"].dropna().astype(str).unique())
    if not A_feats or not B_feats:
        return pd.DataFrame()
    data = []
    for fa in A_feats:
        row = []
        for fb in B_feats:
            r = _corr_from_overlapping_pairs_allpoints(Aw, Bw, fa, fb,
                                                       min_points=min_points,
                                                       min_subjects=min_subjects)
            row.append(r)
        data.append(row)
    return pd.DataFrame(data, index=A_feats, columns=B_feats)

def _features_df_weekly(df: pd.DataFrame, src: str, parent: str, reducer: str) -> pd.DataFrame:
    """
    Returns tidy weekly df with columns:
      subject_id, time_key, feature_key, feature_label, value
    - value is aggregated within subject×time_key×feature (reducer).
    """
    agg = {"mean":"mean","median":"median","max":"max","min":"min"}[reducer]

    if src == "labs":
        cur = df[df["lab_id"].astype(str) == str(parent)].copy()
        cur["feature_key"]   = cur["display_name"]
        cur["feature_label"] = cur["display_name"]

    elif src == "surveys":
        cur = df[df["survey_id"].astype(str) == str(parent)].copy()
        qlbl = cur["display_name"]
        cur["feature_key"]   = cur["metric_id"]
        cur["feature_label"] = np.where(
            qlbl.notna() & (qlbl.astype(str).str.strip() != ""), qlbl, cur["feature_key"]
        )

    else:  # trackers: parent is tracker_type; measurement = display_name within that type
        t = df.copy()
        t["tracker_type"] = t["tracker_type"].astype("string").str.strip()
        cur = t[t["tracker_type"] == (parent or "")].copy()
        cur["feature_key"]   = cur["display_name"]
        cur["feature_label"] = cur["display_name"]

    if cur.empty:
        return pd.DataFrame(columns=["subject_id","time_key","feature_key","feature_label","value"])

    cur = cur.dropna(subset=["subject_id","observed_at","feature_label"])
    cur["time_key"] = pd.to_datetime(cur["observed_at"], utc=True).dt.to_period("W").dt.to_timestamp()

    out = (cur.groupby(["subject_id","time_key","feature_key","feature_label"], as_index=False)
              .agg(value=("value", agg)))
    return out

def _raw_feature_rows(df: pd.DataFrame, src: str, parent: str, feat_label: str) -> pd.DataFrame:
    """
    Return RAW rows (no reducer) for a given selected feature label.
    Columns out: subject_id, observed_at, value
    Matching rules:
      - labs:        parent = lab_id,        feature_label == display_name
      - surveys:     parent = survey_id,     feature_label is display_name if present else metric_id
      - trackers:    parent = tracker_type,  feature_label == display_name
    """
    if df.empty or parent is None or feat_label is None:
        return pd.DataFrame(columns=["subject_id","observed_at","value"])

    if src == "labs":
        cur = df[(df["lab_id"].astype(str) == str(parent)) &
                 (df["display_name"].astype(str) == str(feat_label))]

    elif src == "surveys":
        # When building feature_label for surveys we used display_name if present, otherwise metric_id.
        disp = df["display_name"].astype("string")
        is_disp = disp.notna() & (disp.str.strip() != "")
        cur = df[df["survey_id"].astype(str) == str(parent)].copy()
        cur = cur[
            ((is_disp) & (cur["display_name"].astype(str) == str(feat_label))) |
            ((~is_disp) & (cur["metric_id"].astype(str) == str(feat_label)))
        ]

    else:  # trackers: parent is tracker_type; label = display_name
        t = df.copy()
        t["tracker_type"] = t["tracker_type"].astype("string").str.strip()
        cur = t[(t["tracker_type"] == (parent or "")) &
                (t["display_name"].astype(str) == str(feat_label))]

    cur = cur[["subject_id","observed_at","value"]].dropna(subset=["subject_id","observed_at"])
    # ensure datetime + sort
    cur["observed_at"] = pd.to_datetime(cur["observed_at"], utc=True, errors="coerce")
    cur = cur.dropna(subset=["observed_at"]).sort_values("observed_at")
    return cur

if "fact" in st.session_state:
    st.sidebar.markdown("---")
    st.sidebar.header("Step 3 — Choose Sources")

    labs     = st.session_state["labs"]
    trackers = st.session_state["trackers"]
    surveys  = st.session_state["surveys"]

    # catalogs
    cat_labs          = _cat_labs(labs)                 # lab_id
    cat_surveys       = _cat_surveys(surveys)           # survey_id
    cat_tracker_types = _cat_tracker_types(trackers)    # tracker_type

    lab_options     = cat_labs[["lab_id","label"]].to_dict("records")
    survey_options  = cat_surveys[["survey_id","label_id"]].to_dict("records")
    trktype_options = cat_tracker_types[["tracker_type","label"]].to_dict("records")

    SOURCE_LABELS = {"labs":"Labs", "surveys":"Surveys", "trackers":"Trackers"}

    # -------- Source A --------
    st.sidebar.markdown("### Source A")
    src_a = st.sidebar.radio(
        "Pick source A",
        ["labs","surveys","trackers"],
        format_func=lambda x: SOURCE_LABELS[x],
        key="src_a_radio",
        horizontal=True,
        on_change=_mark_corr_dirty
    )
    if src_a == "labs":
        rec_a = st.sidebar.selectbox(
            "Lab (lab_name – test_name)",
            lab_options, index=None, placeholder="Pick a lab",
            format_func=lambda o: o["label"] if o else "",
            key="select_A_labs",
            on_change=_mark_corr_dirty
        )
        parent_a = (rec_a or {}).get("lab_id")
    elif src_a == "surveys":
        rec_a = st.sidebar.selectbox(
            "Survey template",
            survey_options, index=None, placeholder="Pick a survey",
            format_func=lambda o: o["label_id"] if o else "",
            key="select_A_surveys",
            on_change=_mark_corr_dirty
        )
        parent_a = (rec_a or {}).get("survey_id")
    else:
        rec_a = st.sidebar.selectbox(
            "Tracker type",
            trktype_options, index=None, placeholder="Pick a tracker type",
            format_func=lambda o: o["label"] if o else "",
            key="select_A_trackertype",
            on_change=_mark_corr_dirty
        )
        parent_a = (rec_a or {}).get("tracker_type")

    # -------- Source B --------
    st.sidebar.markdown("### Source B")
    src_b = st.sidebar.radio(
        "Pick source B",
        ["labs","surveys","trackers"],
        index=1,
        format_func=lambda x: SOURCE_LABELS[x],
        key="src_b_radio",
        horizontal=True,
        on_change=_mark_corr_dirty
    )
    if src_b == "labs":
        rec_b = st.sidebar.selectbox(
            "Lab (lab_name – test_name)",
            lab_options, index=None, placeholder="Pick a lab",
            format_func=lambda o: o["label"] if o else "",
            key="select_B_labs",
            on_change=_mark_corr_dirty
        )
        parent_b = (rec_b or {}).get("lab_id")
    elif src_b == "surveys":
        rec_b = st.sidebar.selectbox(
            "Survey template",
            survey_options, index=None, placeholder="Pick a survey",
            format_func=lambda o: o["label_id"] if o else "",
            key="select_B_surveys",
            on_change=_mark_corr_dirty
        )
        parent_b = (rec_b or {}).get("survey_id")
    else:
        rec_b = st.sidebar.selectbox(
            "Tracker type",
            trktype_options, index=None, placeholder="Pick a tracker type",
            format_func=lambda o: o["label"] if o else "",
            key="select_B_trackertype",
            on_change=_mark_corr_dirty
        )
        parent_b = (rec_b or {}).get("tracker_type")

    # -------- reducer (per subject×measurement) + overlap settings + Run --------
    st.sidebar.markdown("---")
    st.sidebar.header("Step 4 — Aggregation & Correlation Settings")

    reducer = st.sidebar.selectbox(
        "Reducer for multiple rows per subject & feature",
        ["mean","median","max","min"], index=0, key="reducer_select",
        on_change=_mark_corr_dirty
    )

    min_weeks = st.sidebar.number_input(
        "Min overlapping weeks per subject",
        min_value=1, max_value=52, value=1,
        help="A subject must have at least this many overlapping weeks between A and B for a pair.",
        on_change=_mark_corr_dirty
    )

    min_subjects = st.sidebar.number_input(
        "Min subjects to report correlation",
        min_value=2, max_value=1000, value=3,
        help="Only compute/display a correlation if at least this many subjects contribute.",
        on_change=_mark_corr_dirty
    )
    # exclude_zeros = st.sidebar.checkbox(
    #     "Exclude (0,0) overlapping points from correlation",
    #     value=False,
    #     help="If enabled, overlapping A/B pairs where both values are exactly 0 are dropped before computing r and contributor counts."
    # )

    # Button to trigger computation
    run_corr = st.sidebar.button(
        "▶️ Run correlations",
        use_container_width=True,
        disabled=not ((parent_a is not None) and (parent_b is not None))
    )


    # ===== Correlation compute orchestration =====
if "fact" in st.session_state:
    org_id = st.session_state.get("org_id")

# Build but only when Run is pressed
if run_corr:
    src_map = {"labs": labs, "surveys": surveys, "trackers": trackers}

    # Cache weekly features per side in session_state so changing overlap thresholds doesn't rebuild them
    key_Aw = _weekly_features_key(org_id, src_a, parent_a, reducer)
    key_Bw = _weekly_features_key(org_id, src_b, parent_b, reducer)

    if key_Aw not in st.session_state:
        st.session_state[key_Aw] = _features_df_weekly(src_map[src_a], src_a, parent_a, reducer)
    if key_Bw not in st.session_state:
        st.session_state[key_Bw] = _features_df_weekly(src_map[src_b], src_b, parent_b, reducer)

    Aw = st.session_state[key_Aw]
    Bw = st.session_state[key_Bw]

    # Build heatmap under the current overlap/subject thresholds
    heat = _heatmap_overlapping_weeks(Aw, Bw, min_weeks)

    # Optionally mask cells with too-few contributing subjects
    # We’ll count contributors per (fa, fb) the same way we computed r:
    def _contributors_count(Aw, Bw, fa, fb, min_weeks_per_subject):
        a = Aw[Aw["feature_label"].astype(str) == str(fa)][["subject_id","time_key","value"]].rename(columns={"value":"A"})
        b = Bw[Bw["feature_label"].astype(str) == str(fb)][["subject_id","time_key","value"]].rename(columns={"value":"B"})
        if a.empty or b.empty:
            return 0
        pairs = pd.merge(a, b, on=["subject_id","time_key"], how="inner")
        if pairs.empty:
            return 0
        per_subj = (pairs.groupby("subject_id", as_index=False)
                          .agg(n_weeks=("A","size")))
        return int((per_subj["n_weeks"] >= min_weeks_per_subject).sum())

    if not heat.empty:
        contrib = pd.DataFrame(
            [[_contributors_count(Aw, Bw, fa, fb, min_weeks) for fb in heat.columns] for fa in heat.index],
            index=heat.index, columns=heat.columns
        )
        heat = heat.where(contrib >= int(min_subjects))

    # Save a single object containing everything needed to render quickly
    st.session_state["corr_cache"] = {
        "heat": heat,
        "Aw": Aw,
        "Bw": Bw,
        "params": {
            "src_a": src_a, "parent_a": parent_a,
            "src_b": src_b, "parent_b": parent_b,
            "reducer": reducer,
            "min_weeks": int(min_weeks),        # use locals, not session_state lookups
            "min_subjects": int(min_subjects),
        }
    }

    st.session_state["corr_ready"]  = True
    st.session_state["corr_result"] = True

# ===== Render (fast): only read from session_state; DO NOT recompute =====
if st.session_state.get("corr_ready") and st.session_state.get("corr_cache"):
    heat = st.session_state["corr_cache"]["heat"]
    Aw   = st.session_state["corr_cache"]["Aw"]
    Bw   = st.session_state["corr_cache"]["Bw"]

    st.subheader("Cross-source correlation heatmap (aligned on overlapping weeks)")
    st.caption(
        f"Rows = measurements from Source A ({src_a}); Columns = measurements from Source B ({src_b}). "
        f"Subjects must have ≥ {min_weeks} overlapping weeks for a pair. Cells require ≥ {min_subjects} subjects."
    )

    if heat.empty or not heat.notna().any().any():
        st.info("No valid cells under the current thresholds. Adjust Step 3 and press Run.")
    else:
        import plotly.express as px
        fig = px.imshow(heat, color_continuous_scale="RdBu", zmin=-1, zmax=1, labels=dict(color="Pearson r"))
        fig.update_layout(height=600, margin=dict(l=0,r=0,t=40,b=0))
        st.plotly_chart(fig, use_container_width=True)

        # ---- Leaderboard (Top ±N), de-dup + drop diagonal when A & B are the same lab ----
        N = st.selectbox("Top N", [3, 5, 10, 20], index=1)

        # Stack heatmap to long
        stacked = (
            heat.stack(dropna=True)
                .reset_index()
                .rename(columns={"level_0": "A_feat", "level_1": "B_feat", 0: "r"})
        )

        # If both sides reference the SAME source & SAME parent (e.g., same lab), filter out:
        same_domain = (src_a == src_b) and (parent_a == parent_b)
        if same_domain:
            # 1) remove diagonal A==B
            stacked = stacked[stacked["A_feat"] != stacked["B_feat"]].copy()
            # 2) remove symmetric duplicates by canonical pair key
            stacked["pair_key"] = stacked.apply(
                lambda row: tuple(sorted((str(row["A_feat"]), str(row["B_feat"])))),
                axis=1,
            )
            stacked = stacked.drop_duplicates(subset=["pair_key"]).drop(columns=["pair_key"])

        # Build pos/neg leaderboards
        top_pos = (stacked.sort_values("r", ascending=False).head(N))[["A_feat", "B_feat", "r"]]
        top_neg = (stacked.sort_values("r", ascending=True).head(N))[["A_feat", "B_feat", "r"]]

        st.markdown("### Leaderboard")
        c_pos, c_neg = st.columns(2)
        with c_pos:
            st.markdown("**Top positive**")
            st.dataframe(top_pos.reset_index(drop=True), use_container_width=True, hide_index=True)
        with c_neg:
            st.markdown("**Top negative**")
            st.dataframe(top_neg.reset_index(drop=True), use_container_width=True, hide_index=True)

        # ---- Pairwise scatter (fast: derive from Aw/Bw; no recompute of heatmap) ----
        strongest = (heat.abs().stack().idxmax()) if heat.notna().any().any() else (None, None)

        col1, col2 = st.columns(2)
        with col1:
            feat_a = st.selectbox(
                "Measurement (row, Source A)", list(heat.index),
                index=(list(heat.index).index(strongest[0]) if strongest[0] else 0),
                key="pick_scatter_feat_a"
            )
        with col2:
            feat_b = st.selectbox(
                "Measurement (col, Source B)", list(heat.columns),
                index=(list(heat.columns).index(strongest[1]) if strongest[1] else 0),
                key="pick_scatter_feat_b"
            )

        if feat_a and feat_b:
            # Build per-subject points from overlapping weeks ONLY (fast groupby on cached Aw/Bw)
            a = Aw[Aw["feature_label"].astype(str) == str(feat_a)][["subject_id","time_key","value"]].rename(columns={"value":"A"})
            b = Bw[Bw["feature_label"].astype(str) == str(feat_b)][["subject_id","time_key","value"]].rename(columns={"value":"B"})
            pair_weeks = pd.merge(a, b, on=["subject_id","time_key"], how="inner")
            per_subj = (pair_weeks.groupby("subject_id", as_index=False)
                                   .agg(n_weeks=("A","size"),
                                        A=("A","mean"),
                                        B=("B","mean")))
            per_subj = per_subj[per_subj["n_weeks"] >= int(min_weeks)]

            if len(per_subj) >= 2:
                r = per_subj[["A","B"]].corr(method="pearson").iloc[0,1]
                st.caption(f"{len(per_subj)} subjects (≥ {min_weeks} overlapping weeks) · Pearson r = {r:.3f}")

                fig = scatter_with_fit_ci(
                    per_subj.rename(columns={"A": feat_a, "B": feat_b}),
                    x_col=feat_a, y_col=feat_b,
                    title=f"{feat_a} vs {feat_b} (per-subject means over overlapping weeks)",
                    id_col="subject_id"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Not enough subjects after overlap filter.")

        # ----- Per-user raw time series (already cached via base DataFrames) -----
        st.markdown("#### Per-user raw time series (no reducer)")
        if feat_a and feat_b:
            src_map = {"labs": labs, "surveys": surveys, "trackers": trackers}
            raw_a = _raw_feature_rows(src_map[src_a], src_a, parent_a, feat_a)
            raw_b = _raw_feature_rows(src_map[src_b], src_b, parent_b, feat_b)

            if raw_a.empty or raw_b.empty:
                st.info("No raw rows for one or both selected measurements.")
            else:
                subs_a = set(raw_a["subject_id"].astype(str))
                subs_b = set(raw_b["subject_id"].astype(str))
                common_subjects = sorted(subs_a & subs_b)
                if not common_subjects:
                    st.info("No subject has raw data for both selections.")
                else:
                    user_sel = st.selectbox(
                        "Choose a subject to inspect",
                        options=common_subjects, index=0, key="ts_user_pick"
                    )

                    a_u = raw_a[raw_a["subject_id"].astype(str) == str(user_sel)].copy()
                    b_u = raw_b[raw_b["subject_id"].astype(str) == str(user_sel)].copy()

                    from plotly.subplots import make_subplots
                    range_a = (a_u["value"].max() - a_u["value"].min()) if len(a_u) else 0.0
                    range_b = (b_u["value"].max() - b_u["value"].min()) if len(b_u) else 0.0
                    suggest_dual = bool(range_a and range_b and (max(range_a, range_b) / max(1e-12, min(range_a, range_b)) >= 10.0))
                    dual_axis = st.checkbox("Use dual y-axis", value=suggest_dual, key="dual_axis_checkbox")

                    fig_ts = make_subplots(specs=[[{"secondary_y": dual_axis}]])
                    fig_ts.add_trace(go.Scatter(x=a_u["observed_at"], y=a_u["value"], mode="lines+markers", name=feat_a), secondary_y=False)
                    if dual_axis:
                        fig_ts.add_trace(go.Scatter(x=b_u["observed_at"], y=b_u["value"], mode="lines+markers", name=feat_b), secondary_y=True)
                        fig_ts.update_yaxes(title_text=feat_a, secondary_y=False)
                        fig_ts.update_yaxes(title_text=feat_b, secondary_y=True)
                    else:
                        fig_ts.add_trace(go.Scatter(x=b_u["observed_at"], y=b_u["value"], mode="lines+markers", name=feat_b), secondary_y=False)
                        fig_ts.update_yaxes(title_text=f"{feat_a} / {feat_b}")
                    fig_ts.update_layout(
                        title=f"Raw values over time — Subject {user_sel}",
                        xaxis_title="Observed at",
                        template="plotly_white",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        margin=dict(l=40, r=20, t=60, b=40),
                    )
                    st.plotly_chart(fig_ts, use_container_width=True)

                    # Optional: show user dataframes (your earlier block)
                    show_df = st.checkbox("Show data table for this user", value=False, key="show_user_df")
                    if show_df:
                        a_tbl = a_u.rename(columns={"observed_at": "date", "value": feat_a}).copy()
                        b_tbl = b_u.rename(columns={"observed_at": "date", "value": feat_b}).copy()
                        a_tbl["subject_id"] = str(user_sel)
                        b_tbl["subject_id"] = str(user_sel)

                        a_week = a_u.assign(week=a_u["observed_at"].dt.to_period("W").dt.to_timestamp()).rename(columns={"value": feat_a})
                        b_week = b_u.assign(week=b_u["observed_at"].dt.to_period("W").dt.to_timestamp()).rename(columns={"value": feat_b})
                        joined = (
                            a_week[["week", feat_a]]
                            .merge(b_week[["week", feat_b]], on="week", how="outer")
                            .sort_values("week")
                            .reset_index(drop=True)
                        )
                        tab1, tab2, tab3 = st.tabs([f"{feat_a} (raw)", f"{feat_b} (raw)", "Joined by week"])
                        with tab1:
                            st.dataframe(a_tbl[["subject_id", "date", feat_a]].sort_values("date"), use_container_width=True)
                        with tab2:
                            st.dataframe(b_tbl[["subject_id", "date", feat_b]].sort_values("date"), use_container_width=True)
                        with tab3:
                            st.dataframe(joined, use_container_width=True)
else:
    # No correlations computed yet or you've changed selections but haven't pressed Run
    if (parent_a is not None) and (parent_b is not None):
        st.info("Set thresholds in Step 3 and press **Run correlations** to compute the heatmap.")


        