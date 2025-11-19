import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import time


st.set_page_config(page_title="Income Prediction — Income Predictor", layout="centered", page_icon="💰")

if 'page' not in st.session_state:
    st.session_state['page'] = 'landing'

def _centered_markdown(html: str):
    st.markdown(f"<div style='text-align:center'>{html}</div>", unsafe_allow_html=True)

import time

PROJECT_DIR = Path(__file__).parent
DATA_PATH = PROJECT_DIR / "income.csv"
MODEL_PATH = PROJECT_DIR / "best_xgboost_model.joblib"


@st.cache_data(ttl=3600)
def prepare_encoders_and_schema():
    # Load raw data and fit encoders exactly as notebook did so we can preprocess incoming rows
    df = pd.read_csv(DATA_PATH)
    # map target (not used for fitting encoders except occupation target-mean)
    df['income'] = df['income'].map({'<=50K': 0, '>50K': 1})

    X = df.drop('income', axis=1)
    y = df['income']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # Replace '?' placeholders with 'unknown' where used
    for col in ['workclass', 'native-country']:
        if col in X_train.columns:
            X_train[col] = X_train[col].replace('?', 'unknown')
            X_test[col] = X_test[col].replace('?', 'unknown')

    # education order
    education_order = [
        'Preschool', '1st-4th', '5th-6th', '7th-8th', '9th', '10th', '11th', '12th',
        'HS-grad', 'Some-college', 'Assoc-voc', 'Assoc-acdm', 'Bachelors', 'Masters',
        'Prof-school', 'Doctorate'
    ]

    # One-hot encoders (handle older/newer sklearn differences)
    def make_ohe(**kw):
        try:
            return OneHotEncoder(**kw)
        except TypeError:
            # fallback for older sklearn versions that use `sparse` instead of `sparse_output`
            args = kw.copy()
            if 'sparse_output' in args:
                args['sparse'] = args.pop('sparse_output')
            return OneHotEncoder(**args)

    ohe_workclass = make_ohe(handle_unknown='ignore', sparse_output=False)
    ohe_race = make_ohe(handle_unknown='ignore', sparse_output=False)
    ohe_native = make_ohe(handle_unknown='ignore', sparse_output=False)

    # Fit encoders where columns exist
    if 'workclass' in X_train.columns:
        ohe_workclass.fit(X_train[['workclass']])
    if 'race' in X_train.columns:
        ohe_race.fit(X_train[['race']])
    if 'native-country' in X_train.columns:
        ohe_native.fit(X_train[['native-country']])

    # marital simplification function (mirror notebook)
    def simplify_marital(x):
        if x in ['Married-civ-spouse', 'Married-AF-spouse']:
            return 1
        else:
            return 0

    # occupation target encoding (simple mean encoding)
    y_train_reset = y_train.reset_index(drop=True)
    tmp = pd.DataFrame({'occupation': X_train['occupation'], 'target': y_train_reset})
    global_mean = tmp['target'].mean(skipna=True)
    occ_means = tmp.groupby('occupation')['target'].mean().fillna(global_mean)

    # Apply all transforms to derive final feature schema (dummies)
    Xtr = X_train.copy().reset_index(drop=True)
    # replace unknowns done above
    # education encoding
    if 'education' in Xtr.columns:
        Xtr['education'] = pd.Categorical(Xtr['education'], categories=education_order, ordered=True)
        Xtr['education_encoded'] = Xtr['education'].cat.codes + 1
    elif 'education-num' in Xtr.columns:
        Xtr['education_encoded'] = Xtr['education-num']

    # marital
    if 'marital-status' in Xtr.columns:
        Xtr['marital_status'] = Xtr['marital-status'].apply(simplify_marital)

    # occupation encoding
    if 'occupation' in Xtr.columns:
        Xtr['occupation_encoded'] = Xtr['occupation'].map(occ_means).fillna(global_mean)

    # drop columns that were dropped in notebook
    for c in ['relationship', 'education', 'education-num', 'educational-num', 'occupation', 'marital-status']:
        if c in Xtr.columns:
            try:
                Xtr = Xtr.drop(columns=[c])
            except Exception:
                pass

    # gender mapping
    if 'gender' in Xtr.columns:
        Xtr['gender'] = Xtr['gender'].map({'Female': 0, 'Male': 1})
    elif 'sex' in Xtr.columns:
        Xtr['gender'] = Xtr['sex'].map({'Female': 0, 'Male': 1})
        if 'sex' in Xtr.columns:
            Xtr = Xtr.drop(columns=['sex'])

    # one-hot for workclass/race/native-country if present
    new_parts = []
    if 'workclass' in Xtr.columns and hasattr(ohe_workclass, 'transform'):
        wc = ohe_workclass.transform(Xtr[['workclass']])
        df_wc = pd.DataFrame(wc, columns=ohe_workclass.get_feature_names_out(['workclass']))
        new_parts.append(df_wc)
        Xtr = Xtr.drop(columns=['workclass'])

    if 'race' in Xtr.columns and hasattr(ohe_race, 'transform'):
        rc = ohe_race.transform(Xtr[['race']])
        df_rc = pd.DataFrame(rc, columns=ohe_race.get_feature_names_out(['race']))
        new_parts.append(df_rc)
        Xtr = Xtr.drop(columns=['race'])

    if 'native-country' in Xtr.columns and hasattr(ohe_native, 'transform'):
        nc = ohe_native.transform(Xtr[['native-country']])
        df_nc = pd.DataFrame(nc, columns=ohe_native.get_feature_names_out(['native-country']))
        new_parts.append(df_nc)
        Xtr = Xtr.drop(columns=['native-country'])

    if new_parts:
        Xtr = pd.concat([Xtr.reset_index(drop=True)] + [p.reset_index(drop=True) for p in new_parts], axis=1)

    # final dummies & alignment (the notebook used get_dummies then aligned train/test)
    X_train_enc = pd.get_dummies(Xtr, dummy_na=False)
    # Also apply same pipeline to X_test and align to capture final columns
    Xte = X_test.copy().reset_index(drop=True)
    # mirror transforms on Xte
    if 'workclass' in Xte.columns:
        Xte['workclass'] = Xte['workclass'].replace('?', 'unknown')
    if 'native-country' in Xte.columns:
        Xte['native-country'] = Xte['native-country'].replace('?', 'unknown')
    if 'education' in Xte.columns:
        Xte['education'] = pd.Categorical(Xte['education'], categories=education_order, ordered=True)
        Xte['education_encoded'] = Xte['education'].cat.codes + 1
    elif 'education-num' in Xte.columns:
        Xte['education_encoded'] = Xte['education-num']
    if 'marital-status' in Xte.columns:
        Xte['marital_status'] = Xte['marital-status'].apply(simplify_marital)
    if 'occupation' in Xte.columns:
        Xte['occupation_encoded'] = Xte['occupation'].map(occ_means).fillna(global_mean)
    for c in ['relationship', 'education', 'education-num', 'educational-num', 'occupation', 'marital-status']:
        if c in Xte.columns:
            try:
                Xte = Xte.drop(columns=[c])
            except Exception:
                pass
    if 'gender' in Xte.columns:
        Xte['gender'] = Xte['gender'].map({'Female': 0, 'Male': 1})
    elif 'sex' in Xte.columns:
        Xte['gender'] = Xte['sex'].map({'Female': 0, 'Male': 1})
        if 'sex' in Xte.columns:
            Xte = Xte.drop(columns=['sex'])

    parts2 = []
    if 'workclass' in Xte.columns and hasattr(ohe_workclass, 'transform'):
        wc2 = ohe_workclass.transform(Xte[['workclass']])
        parts2.append(pd.DataFrame(wc2, columns=ohe_workclass.get_feature_names_out(['workclass'])))
        Xte = Xte.drop(columns=['workclass'])
    if 'race' in Xte.columns and hasattr(ohe_race, 'transform'):
        rc2 = ohe_race.transform(Xte[['race']])
        parts2.append(pd.DataFrame(rc2, columns=ohe_race.get_feature_names_out(['race'])))
        Xte = Xte.drop(columns=['race'])
    if 'native-country' in Xte.columns and hasattr(ohe_native, 'transform'):
        nc2 = ohe_native.transform(Xte[['native-country']])
        parts2.append(pd.DataFrame(nc2, columns=ohe_native.get_feature_names_out(['native-country'])))
        Xte = Xte.drop(columns=['native-country'])
    if parts2:
        Xte = pd.concat([Xte.reset_index(drop=True)] + [p.reset_index(drop=True) for p in parts2], axis=1)

    X_test_enc = pd.get_dummies(Xte, dummy_na=False)
    X_train_enc, X_test_enc = X_train_enc.align(X_test_enc, join='outer', axis=1, fill_value=0)

    feature_columns = list(X_train_enc.columns)

    # bundle and return relevant objects
    artifacts = {
        'ohe_workclass': ohe_workclass,
        'ohe_race': ohe_race,
        'ohe_native': ohe_native,
        'education_order': education_order,
        'occ_means': occ_means,
        'global_mean': global_mean,
        'feature_columns': feature_columns,
        'X_train_columns': feature_columns,
        'sample_X_train': X_train_enc.iloc[:1].copy()
    }
    return artifacts


art = prepare_encoders_and_schema()


def preprocess_single(raw: pd.DataFrame, artifacts: dict) -> pd.DataFrame:
    df = raw.copy()
    ohe_workclass = artifacts['ohe_workclass']
    ohe_race = artifacts['ohe_race']
    ohe_native = artifacts['ohe_native']
    education_order = artifacts['education_order']
    occ_means = artifacts['occ_means']
    global_mean = artifacts['global_mean']
    feature_columns = artifacts['feature_columns']

    # normalize missing markers
    for col in ['workclass', 'native-country']:
        if col in df.columns:
            df[col] = df[col].replace('?', 'unknown')

    # education
    if 'education' in df.columns:
        df['education'] = pd.Categorical(df['education'], categories=education_order, ordered=True)
        df['education_encoded'] = df['education'].cat.codes + 1
    elif 'education-num' in df.columns:
        df['education_encoded'] = df['education-num']

    # marital
    if 'marital-status' in df.columns:
        df['marital_status'] = df['marital-status'].apply(lambda x: 1 if x in ['Married-civ-spouse', 'Married-AF-spouse'] else 0)

    # occupation
    if 'occupation' in df.columns:
        df['occupation_encoded'] = df['occupation'].map(occ_means).fillna(global_mean)
    else:
        df['occupation_encoded'] = global_mean

    # drop columns
    for c in ['relationship', 'education', 'education-num', 'educational-num', 'occupation', 'marital-status']:
        if c in df.columns:
            try:
                df = df.drop(columns=[c])
            except Exception:
                pass

    # gender
    if 'gender' in df.columns:
        df['gender'] = df['gender'].map({'Female': 0, 'Male': 1}).fillna(0)
    elif 'sex' in df.columns:
        df['gender'] = df['sex'].map({'Female': 0, 'Male': 1}).fillna(0)
        if 'sex' in df.columns:
            df = df.drop(columns=['sex'])

    # one-hot encode workclass/race/native-country where possible
    new_parts = []
    if 'workclass' in df.columns and hasattr(ohe_workclass, 'transform'):
        try:
            wc = ohe_workclass.transform(df[['workclass']])
            df_wc = pd.DataFrame(wc, columns=ohe_workclass.get_feature_names_out(['workclass']))
            new_parts.append(df_wc)
            df = df.drop(columns=['workclass'])
        except Exception:
            pass
    if 'race' in df.columns and hasattr(ohe_race, 'transform'):
        try:
            rc = ohe_race.transform(df[['race']])
            df_rc = pd.DataFrame(rc, columns=ohe_race.get_feature_names_out(['race']))
            new_parts.append(df_rc)
            df = df.drop(columns=['race'])
        except Exception:
            pass
    if 'native-country' in df.columns and hasattr(ohe_native, 'transform'):
        try:
            nc = ohe_native.transform(df[['native-country']])
            df_nc = pd.DataFrame(nc, columns=ohe_native.get_feature_names_out(['native-country']))
            new_parts.append(df_nc)
            df = df.drop(columns=['native-country'])
        except Exception:
            pass
    if new_parts:
        df = pd.concat([df.reset_index(drop=True)] + [p.reset_index(drop=True) for p in new_parts], axis=1)

    # final get_dummies and align to training features
    df_enc = pd.get_dummies(df, dummy_na=False)
    for c in feature_columns:
        if c not in df_enc.columns:
            df_enc[c] = 0
    # drop extra columns that model doesn't expect
    extra = [c for c in df_enc.columns if c not in feature_columns]
    if extra:
        df_enc = df_enc.drop(columns=extra)
    df_enc = df_enc[feature_columns]
    return df_enc.astype(float)


def load_model():
    if not MODEL_PATH.exists():
        return None
    try:
        m = joblib.load(MODEL_PATH)
        return m
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None


model = load_model()
if model is None:
    st.warning('Model file `best_xgboost_model.joblib` not found or failed to load. Train and save the model in the notebook first.')


def show_landing():

    # Loader trigger
    if "loading" not in st.session_state:
        st.session_state["loading"] = False

    # --- Custom CSS for Futuristic Neon UI ---
    st.markdown("""
    <style>

    body {
        background: radial-gradient(circle at center, #00111a, #000000);
    }

    /* Fade In Animation */
    @keyframes fadeIn {
        0%   {opacity:0; transform: translateY(20px);}
        100% {opacity:1; transform: translateY(0);}
    }

    /* Floating Glow Particles */
    .glow-circle {
        position: absolute;
        width: 180px;
        height: 180px;
        background: radial-gradient(circle, rgba(0,200,255,0.4), transparent);
        filter: blur(35px);
        border-radius: 50%;
        animation: float 6s infinite ease-in-out alternate;
    }

    .glow-circle2 {
        position: absolute;
        width: 230px;
        height: 230px;
        top: 250px;
        right: -80px;
        background: radial-gradient(circle, rgba(255,0,170,0.4), transparent);
        filter: blur(40px);
        border-radius: 50%;
        animation: float 7s infinite ease-in-out alternate-reverse;
    }

    @keyframes float {
        from {transform: translateY(0px);}
        to {transform: translateY(-30px);}
    }

    /* Neon card */
    .neon-card {
        margin-top: 70px;
        padding: 45px 55px;
        border-radius: 18px;
        border: 2px solid rgba(0,255,255,0.4);
        background: rgba(0, 10, 15, 0.45);
        backdrop-filter: blur(14px);
        animation: fadeIn 1s ease-out;
        box-shadow: 0 0 25px rgba(0,255,255,0.4);
    }

    /* Neon Title + Typing effect */
    .title {
        font-size: 52px;
        font-weight: 900;
        color: #00eaff;
        text-shadow: 0 0 20px #00eaff;
        width: 100%;
        overflow: hidden;
        border-right: .15em solid #00eaff;
        white-space: nowrap;
        animation: typing 3.5s steps(30, end), blink .75s infinite;
    }

    @keyframes typing {
        from { width: 0 }
        to { width: 100% }
    }
    @keyframes blink {
        50% { border-color: transparent }
    }

    .subtitle {
        margin-top: 10px;
        font-size: 21px;
        color: #ffffffcc;
    }

    /* Neon Button */
    .neon-btn button {
        background: rgba(0,255,255,0.2) !important;
        border: 2px solid #00eaff !important;
        color: #00eaff !important;
        font-size: 22px !important;
        padding: 12px 28px !important;
        border-radius: 12px !important;
        text-shadow: 0 0 10px #00eaff;
        transition: all 0.25s ease-in-out !important;
        box-shadow: 0 0 12px rgba(0,255,255,0.5);
    }
    .neon-btn button:hover {
        background: rgba(0,255,255,0.35) !important;
        transform: scale(1.08);
        box-shadow: 0 0 20px rgba(0,255,255,0.9);
    }

    </style>
    """, unsafe_allow_html=True)

    # Floating glow decorations
    st.markdown("""
        <div class="glow-circle"></div>
        <div class="glow-circle2"></div>
    """, unsafe_allow_html=True)

    # --- Main Neon Card ---
    st.markdown("<div class='neon-card'>", unsafe_allow_html=True)

    st.markdown("<h1 class='title'>💰 Income Predictor</h1>", unsafe_allow_html=True)

    st.markdown("""
    <p class="subtitle">
        A futuristic ML tool that predicts whether an individual earns more than $50K/year.
        <br>Powered by advanced XGBoost modeling, preprocessing, and feature engineering.
    </p>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Animated Proceed Button
    st.markdown("<div class='neon-btn' style='text-align:center;'>", unsafe_allow_html=True)

    if st.button("🔮 Proceed to Prediction"):
        st.session_state["loading"] = True
        with st.spinner("Loading the ML Prediction Form..."):
            time.sleep(2)
        st.session_state["page"] = "form"

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br><hr><br>", unsafe_allow_html=True)

    st.markdown("""
    <p style='text-align:center;color:#ffffffaa;font-size:17px;'>
        Tip: We pre-filled some example values to help you get started quickly.
    </p>
    """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

def show_form():
    st.header('Enter a single person record')
    df_ref = pd.read_csv(DATA_PATH)
    work_choices = sorted(df_ref['workclass'].fillna('unknown').unique().tolist())
    education_choices = [
        'Preschool','1st-4th','5th-6th','7th-8th','9th','10th','11th','12th',
        'HS-grad','Some-college','Assoc-voc','Assoc-acdm','Bachelors','Masters','Prof-school','Doctorate'
    ]
    marital_choices = sorted(df_ref['marital-status'].fillna('Unknown').unique().tolist())
    occupation_choices = sorted(df_ref['occupation'].fillna('Unknown').unique().tolist())
    relationship_choices = sorted(df_ref['relationship'].fillna('Unknown').unique().tolist())
    race_choices = sorted(df_ref['race'].fillna('Unknown').unique().tolist())
    gender_choices = ['Male', 'Female']
    native_choices = sorted(df_ref['native-country'].fillna('Unknown').unique().tolist())

    # prepend blank choice so selects default to empty
    work_choices = [''] + work_choices
    education_choices = [''] + education_choices
    marital_choices = [''] + marital_choices
    occupation_choices = [''] + occupation_choices
    relationship_choices = [''] + relationship_choices
    race_choices = [''] + race_choices
    gender_choices = [''] + gender_choices
    native_choices = [''] + native_choices

    with st.form('input_form'):
        cols = st.columns(2)
        with cols[0]:
            age = st.text_input('age', value='', placeholder='Enter age (numeric)')
            workclass = st.selectbox('workclass', work_choices, index=0)
            fnlwgt = st.text_input('fnlwgt', value='', placeholder='Enter fnlwgt (numeric)')
            education = st.selectbox('education', education_choices, index=0)
            education_num = st.text_input('education-num', value='', placeholder='Enter education-num (numeric)')
            marital_status = st.selectbox('marital-status', marital_choices, index=0)
        with cols[1]:
            occupation = st.selectbox('occupation', occupation_choices, index=0)
            relationship = st.selectbox('relationship', relationship_choices, index=0)
            race = st.selectbox('race', race_choices, index=0)
            gender = st.selectbox('gender', gender_choices, index=0)
            capital_gain = st.text_input('capital-gain', value='', placeholder='0')
            capital_loss = st.text_input('capital-loss', value='', placeholder='0')
            hours_per_week = st.text_input('hours-per-week', value='', placeholder='Enter hours-per-week')
            native_country = st.selectbox('native-country', native_choices, index=0)

        submitted = st.form_submit_button('Predict')

    if st.button('← Back', key='back_from_form'):
        st.session_state['page'] = 'landing'

    if submitted:
        # convert empty text inputs to numeric values (empty -> 0)
        age_val = int(pd.to_numeric(age, errors='coerce').fillna(0))
        fnlwgt_val = int(pd.to_numeric(fnlwgt, errors='coerce').fillna(0))
        education_num_val = int(pd.to_numeric(education_num, errors='coerce').fillna(0))
        capital_gain_val = float(pd.to_numeric(capital_gain, errors='coerce').fillna(0))
        capital_loss_val = float(pd.to_numeric(capital_loss, errors='coerce').fillna(0))
        hours_val = float(pd.to_numeric(hours_per_week, errors='coerce').fillna(0))

        raw = pd.DataFrame([{ 
            'age': age_val,
            'workclass': workclass,
            'fnlwgt': fnlwgt_val,
            'education': education,
            'education-num': education_num_val,
            'marital-status': marital_status,
            'occupation': occupation,
            'relationship': relationship,
            'race': race,
            'gender': gender,
            'capital-gain': capital_gain_val,
            'capital-loss': capital_loss_val,
            'hours-per-week': hours_val,
            'native-country': native_country
        }])

        with st.spinner('Predicting — loading model and running inference...'):
            progress = st.progress(0)
            for i in range(20):
                time.sleep(0.03)
                progress.progress(int((i+1)/20*100))
            processed = preprocess_single(raw, art)

        with st.expander('View processed features (model input)'):
            st.dataframe(processed)

        if model is None:
            st.error('No trained model available. Please run the notebook to generate `best_xgboost_model.joblib`.')
            return

        # Align to model features (same logic as before)
        model_features = getattr(model, 'feature_names_in_', None)
        if model_features is None:
            try:
                booster = model.get_booster()
                model_features = getattr(booster, 'feature_names', None)
            except Exception:
                model_features = None

        if model_features is not None:
            model_features = list(model_features)
            for c in model_features:
                if c not in processed.columns:
                    processed[c] = 0.0
            extra = [c for c in processed.columns if c not in model_features]
            if extra:
                processed = processed.drop(columns=extra)
            processed = processed[model_features]
        else:
            # fallback
            fallback_cols = art.get('feature_columns', None)
            if fallback_cols is not None:
                for c in fallback_cols:
                    if c not in processed.columns:
                        processed[c] = 0.0
                extra = [c for c in processed.columns if c not in fallback_cols]
                if extra:
                    processed = processed.drop(columns=extra)
                processed = processed[fallback_cols]

        # run prediction
        try:
            pred = model.predict(processed)[0]
            proba = None
            try:
                proba_arr = model.predict_proba(processed)[0]
                # probability of >50K is index 1 if classes are [0,1]
                p_gt = float(proba_arr[1]) * 100.0 if len(proba_arr) > 1 else None
            except Exception:
                p_gt = None

            label = '<=50K' if pred == 0 else '>50K'
            icon = '✅' if pred == 1 else '🔍'

            # big centered result
            _centered_markdown(f"<div style='font-size:48px;margin:6px'>{icon}</div>")
            st.success(f'Predicted: {label}')
            if p_gt is not None:
                st.metric(label='Estimated probability >50K', value=f"{p_gt:.1f}%")
            else:
                st.info('Probability not available for this model.')

            st.write('You can export the processed features or try another prediction.')
            if st.button('Make another prediction', key='another'):
                st.experimental_rerun()
        except Exception as e:
            st.error(f'Prediction failed: {e}')


# decide which page to show
if st.session_state.get('page') == 'landing':
    show_landing()
else:
    show_form()
