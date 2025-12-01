"""
college_predictor_full.py
Complete module:
 - Improved ML (Target Encoding + XGBoost)
 - Rule engine enforcing counselling rules (category, PWD, pool, quota, round, opening/closing ranks)
 - Training, saving, loading, predict_single, predict_top_n

Assumptions:
 - Config provides TRAINING_DATA_PATH, MODEL_PATH, ENCODER_PATH, TARGET_ENCODER_PATH, CAT_SAFE_PATH
 - Website input fields are EXACTLY:
     year, institute_type, round_no, quota, pool, category, closing_rank
 - Output expected by website: (institute_short, confidence) for single predictions
   and list-of-(institute_short, confidence, matched_row_info) for top-N.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from category_encoders import TargetEncoder
from xgboost import XGBClassifier
from config import Config

# ---------- Utility: normalize category strings -------------
def split_category_and_flags(cat_str: str):
    """
    Splits category string like 'OBC-NCL-PWD' -> ('OBC-NCL', {'PWD'})
    Also handles 'GEN-PWD' -> ('GEN', {'PWD'})
    """
    if pd.isna(cat_str):
        return ("UNKNOWN", set())
    s = str(cat_str).strip()
    parts = s.split('-')
    flags = set()
    base = []
    for p in parts:
        p_up = p.upper()
        if p_up in ('PWD', 'PH'):
            flags.add('PWD')
        else:
            base.append(p_up)
    base_cat = '-'.join(base) if base else 'UNKNOWN'
    return (base_cat, flags)


class CollegePredictorFull:
    """
    Full predictor with ML + JEE rules engine
    """

    def __init__(self):
        # model artifacts
        self.model = None
        self.target_encoder = None  # category_encoders TargetEncoder
        self.label_encoder_target = None  # LabelEncoder for institute_short
        self.cat_value_lists = {}  # known categories for safe fallback

        # fixed features (MUST match website input)
        self.feature_columns = [
            'year', 'institute_type', 'round_no',
            'quota', 'pool', 'category', 'closing_rank'
        ]
        self.target_column = 'institute_short'

        # optionally cache training data for rules lookup
        self._seat_df = None

    # ---------------- Data loading & cleaning ----------------
    def load_data(self, filepath):
        print("Loading data from:", filepath)
        df = pd.read_csv(filepath)
        # basic cleaning
        df = df.dropna(subset=['closing_rank', self.target_column])
        df = df[df['closing_rank'].astype(float) > 0]
        if 'is_preparatory' in df.columns:
            df = df[df['is_preparatory'] == 0]
        df.reset_index(drop=True, inplace=True)
        print(f"Loaded {len(df)} rows.")
        return df

    # ---------------- Target encoding & label encoding ----------------
    def encode_target(self, df, fit=True):
        df2 = df.copy()
        if fit:
            self.label_encoder_target = LabelEncoder()
            df2['target_encoded'] = self.label_encoder_target.fit_transform(df2[self.target_column].astype(str))
        else:
            df2['target_encoded'] = self.label_encoder_target.transform(df2[self.target_column].astype(str))
        return df2

    def fit_category_target_encoder(self, df):
        cat_cols = ['institute_type', 'quota', 'pool', 'category']
        df_cat = df[cat_cols].astype(str).copy()
        for col in cat_cols:
            self.cat_value_lists[col] = df_cat[col].unique().tolist()
        self.target_encoder = TargetEncoder(cols=cat_cols, smoothing=0.3)
        # Use numeric encoded target for target encoding
        self.target_encoder.fit(df_cat, df['target_encoded'])

    def transform_categories(self, df, fit=False):
        cat_cols = ['institute_type', 'quota', 'pool', 'category']
        df2 = df.copy()
        for col in cat_cols:
            df2[col] = df2[col].astype(str)
        if fit:
            df2[cat_cols] = self.target_encoder.transform(df2[cat_cols])
        else:
            # fallback unseen -> most common
            for col in cat_cols:
                known = self.cat_value_lists.get(col, [])
                if len(known) == 0:
                    continue
                most_common = known[0]
                df2[col] = df2[col].apply(lambda v: v if v in known else most_common)
            df2[cat_cols] = self.target_encoder.transform(df2[cat_cols])
        return df2

    def prepare_features(self, df, fit=False):
        df_copy = df.copy()
        # ensure columns exist
        for col in self.feature_columns:
            if col not in df_copy.columns:
                df_copy[col] = 0 if col in ['year', 'round_no', 'closing_rank'] else "UNKNOWN"
        if fit:
            df_copy = self.transform_categories(df_copy, fit=True)
        else:
            df_copy = self.transform_categories(df_copy, fit=False)
        # numeric conversion
        df_copy['year'] = pd.to_numeric(df_copy['year'], errors='coerce').fillna(0).astype(int)
        df_copy['round_no'] = pd.to_numeric(df_copy['round_no'], errors='coerce').fillna(0).astype(int)
        df_copy['closing_rank'] = pd.to_numeric(df_copy['closing_rank'], errors='coerce').fillna(0).astype(float)
        return df_copy[self.feature_columns]

    # ---------------- Train / Save / Load model ----------------
    def train(self, training_csv_path):
        df = self.load_data(training_csv_path)
        df_enc = self.encode_target(df, fit=True)
        self.fit_category_target_encoder(df_enc)
        X_all = self.prepare_features(df_enc, fit=True)
        y_all = df_enc['target_encoded']
        X_train, X_test, y_train, y_test = train_test_split(
            X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
        )
        self.model = XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=8,
            subsample=0.9,
            colsample_bytree=0.9,
            objective='multi:softprob',
            eval_metric='mlogloss',
            use_label_encoder=False,
            verbosity=0,
            random_state=42
        )
        print("Fitting XGBoost (may take time)...")
        self.model.fit(X_train, y_train)
        train_acc = float(self.model.score(X_train, y_train))
        test_acc = float(self.model.score(X_test, y_test))
        print(f"Train acc: {train_acc:.4f}, Test acc: {test_acc:.4f}")
        # cache seat dataframe for rules (original df)
        self._seat_df = df.copy()
        return train_acc, test_acc

    def save_model(self):
        if not hasattr(Config, 'MODEL_PATH') or not hasattr(Config, 'ENCODER_PATH') \
           or not hasattr(Config, 'TARGET_ENCODER_PATH') or not hasattr(Config, 'CAT_SAFE_PATH'):
            raise AttributeError("Config must provide MODEL_PATH, ENCODER_PATH, TARGET_ENCODER_PATH, CAT_SAFE_PATH")
        joblib.dump(self.model, Config.MODEL_PATH)
        joblib.dump(self.label_encoder_target, Config.ENCODER_PATH)
        joblib.dump(self.target_encoder, Config.TARGET_ENCODER_PATH)
        joblib.dump(self.cat_value_lists, Config.CAT_SAFE_PATH)
        print("Saved model and encoders.")

    def load_model(self):
        if not hasattr(Config, 'MODEL_PATH') or not hasattr(Config, 'ENCODER_PATH') \
           or not hasattr(Config, 'TARGET_ENCODER_PATH') or not hasattr(Config, 'CAT_SAFE_PATH'):
            raise AttributeError("Config must provide MODEL_PATH, ENCODER_PATH, TARGET_ENCODER_PATH, CAT_SAFE_PATH")
        if not os.path.exists(Config.MODEL_PATH):
            raise FileNotFoundError(f"Model not found at {Config.MODEL_PATH}")
        self.model = joblib.load(Config.MODEL_PATH)
        self.label_encoder_target = joblib.load(Config.ENCODER_PATH)
        self.target_encoder = joblib.load(Config.TARGET_ENCODER_PATH)
        self.cat_value_lists = joblib.load(Config.CAT_SAFE_PATH)
        # load training data for rule lookups if present
        if hasattr(Config, 'TRAINING_DATA_PATH') and os.path.exists(Config.TRAINING_DATA_PATH):
            self._seat_df = pd.read_csv(Config.TRAINING_DATA_PATH)
        print("Model & encoders loaded.")

    # ---------------- Rule engine helpers ----------------
    def _row_matches_pool_quota_round(self, row, user_input):
        """Check if a df row aligns with user's pool/quota/round/year if present"""
        # exact match for quota/pool/round/year if those columns exist in row
        # We treat missing columns permissively
        try:
            if 'quota' in row and pd.notna(row['quota']):
                if str(row['quota']).strip().lower() != str(user_input.get('quota', '')).strip().lower():
                    return False
        except Exception:
            pass
        try:
            if 'pool' in row and pd.notna(row['pool']):
                if str(row['pool']).strip().lower() != str(user_input.get('pool', '')).strip().lower():
                    return False
        except Exception:
            pass
        try:
            if 'round_no' in row and pd.notna(row['round_no']):
                # row round might be numeric or textual like '6' or 'Round 1' - normalize numeric
                try:
                    row_r = int(row['round_no'])
                except Exception:
                    # try to extract digits
                    import re
                    m = re.search(r'\d+', str(row['round_no']))
                    row_r = int(m.group()) if m else None
                user_r = int(user_input.get('round_no', 0))
                if row_r is not None and row_r != user_r:
                    return False
        except Exception:
            pass
        try:
            if 'year' in row and pd.notna(row['year']):
                if int(row['year']) != int(user_input.get('year', 0)):
                    return False
        except Exception:
            pass
        return True

    def _is_pwd_required(self, row_category_base):
        # returns True if the row category indicates a PWD seat
        return 'PWD' in row_category_base[1]

    def _candidate_has_pwd(self, user_category_base):
        return 'PWD' in user_category_base[1]

    def _eligible_by_category_and_rank(self, row_cat_str, row_open, row_close, user_cat_str, user_rank):
        """
        Accept if:
         - user's rank lies between row_open and row_close AND
         - category compatibility:
            * if row is GEN -> any candidate can be eligible (GEN is open)
            * if row base equals user base -> eligible
            * if user is reserved and row is GEN and user's rank in GEN range -> eligible (reserved candidates can get GEN seats)
         - PWD must match (if row requires PWD then user must have PWD)
        """
        row_base, row_flags = split_category_and_flags(row_cat_str)
        user_base, user_flags = split_category_and_flags(user_cat_str)

        # rank check
        try:
            row_open = int(row_open)
            row_close = int(row_close)
        except Exception:
            # if not numeric, fail safe: not eligible
            return False

        if not (row_open <= user_rank <= row_close):
            return False

        # PWD check
        if 'PWD' in row_flags and 'PWD' not in user_flags:
            return False

        # Category compatibility:
        # If row is GEN -> eligible for all (including reserved) if rank in range
        if row_base == 'GEN':
            return True

        # If row_base equals user_base (e.g., both SC) -> eligible
        if row_base == user_base:
            return True

        # If user is GEN and row is reserved (SC/OBC/...) -> user cannot take reserved seat
        if user_base == 'GEN' and row_base != 'GEN':
            return False

        # If user is reserved (SC/OBC/...) and row is reserved other than user's category -> NOT eligible
        # Example: SC candidate cannot take OBC seat
        if user_base != 'GEN' and row_base != 'GEN' and row_base != user_base:
            return False

        # Otherwise conservative fallback: disallow
        return False

    # ---------------- Find a valid seat for a predicted institute -------------
    def find_valid_seat_for_institute(self, institute_short, user_input):
        """
        Given institute_short (string) and user_input, search training data rows for a valid seat that obeys rules.
        Returns a dict with matched row info or None
        """
        if self._seat_df is None:
            # try to load the training data if available in Config
            if hasattr(Config, 'TRAINING_DATA_PATH') and os.path.exists(Config.TRAINING_DATA_PATH):
                self._seat_df = pd.read_csv(Config.TRAINING_DATA_PATH)
            else:
                return None

        df = self._seat_df
        # narrow by institute
        candidates = df[df['institute_short'].astype(str).str.strip().str.upper() == str(institute_short).strip().upper()].copy()
        if candidates.empty:
            return None

        # prefer same year, round, quota, pool matches → progressive relaxation
        search_orders = [
            {'year': user_input.get('year'), 'round_no': user_input.get('round_no'), 'quota': user_input.get('quota'), 'pool': user_input.get('pool')},
            {'year': user_input.get('year'), 'round_no': user_input.get('round_no')},
            {'year': user_input.get('year')},
            {}
        ]

        user_rank = int(user_input.get('closing_rank'))
        user_cat_str = str(user_input.get('category', 'GEN'))

        for criteria in search_orders:
            sub = candidates.copy()
            # apply criteria filters
            for k, v in criteria.items():
                if k in sub.columns and pd.notna(v):
                    sub = sub[sub[k].astype(str).str.strip().str.upper() == str(v).strip().upper()]
            # iterate rows to check exact eligibility
            for _, row in sub.iterrows():
                row_cat = row.get('category', '')
                row_open = row.get('opening_rank', None)
                row_close = row.get('closing_rank', None)
                # if opening/closing missing, skip
                if pd.isna(row_open) or pd.isna(row_close):
                    continue
                try:
                    row_open_i = int(row_open)
                    row_close_i = int(row_close)
                except Exception:
                    continue

                # check pool/quota/round/year match permissively
                if not self._row_matches_pool_quota_round(row, user_input):
                    continue

                # category & rank eligibility
                if self._eligible_by_category_and_rank(row_cat, row_open_i, row_close_i, user_cat_str, user_rank):
                    # build matched info
                    matched = {
                        'institute_short': row.get('institute_short'),
                        'program_name': row.get('program_name', None),
                        'degree_short': row.get('degree_short', None),
                        'institute_type': row.get('institute_type', None),
                        'quota': row.get('quota', None),
                        'pool': row.get('pool', None),
                        'category': row_cat,
                        'opening_rank': int(row_open_i),
                        'closing_rank': int(row_close_i)
                    }
                    return matched
        return None

    # ---------------- Prediction: single & top-n ----------------
    def predict_single(self, user_input):
        """
        Returns a single (institute_short, confidence).
        Enforces rules: picks top model prediction that has a valid seat per rules.
        If none valid, falls back to top model prediction (no rule).
        """
        if self.model is None:
            self.load_model()

        # prepare input features in same fixed order and transform categories
        df_input = pd.DataFrame([user_input])
        # safety: ensure columns
        for col in self.feature_columns:
            if col not in df_input.columns:
                df_input[col] = 0 if col in ['year', 'round_no', 'closing_rank'] else "UNKNOWN"
        X = self.prepare_features(df_input, fit=False)

        probs = self.model.predict_proba(X)[0]
        class_indices = np.argsort(probs)[::-1]  # descending
        # iterate predicted institutes and check rules
        for idx in class_indices:
            inst = self.label_encoder_target.inverse_transform([int(idx)])[0]
            matched = self.find_valid_seat_for_institute(inst, user_input)
            if matched is not None:
                return inst, float(probs[int(idx)])
        # fallback: return top predicted ignoring rules
        top_idx = class_indices[0]
        top_inst = self.label_encoder_target.inverse_transform([int(top_idx)])[0]
        return top_inst, float(probs[int(top_idx)])

    def predict_top_n(self, user_input, top_n=30):
        """
        Returns list of valid predictions of the form:
         [{'institute_short':..., 'confidence':..., 'matched_row': {...}}, ...]
        If not enough rule-valid results found, append top model predictions (with matched_row=None)
        to reach top_n.
        """
        if self.model is None:
            self.load_model()

        df_input = pd.DataFrame([user_input])
        for col in self.feature_columns:
            if col not in df_input.columns:
                df_input[col] = 0 if col in ['year', 'round_no', 'closing_rank'] else "UNKNOWN"
        X = self.prepare_features(df_input, fit=False)

        probs = self.model.predict_proba(X)[0]
        class_indices = np.argsort(probs)[::-1]  # descending

        results = []
        used_insts = set()

        # First pass: only rule-valid matches
        for idx in class_indices:
            inst = self.label_encoder_target.inverse_transform([int(idx)])[0]
            if inst in used_insts:
                continue
            matched = self.find_valid_seat_for_institute(inst, user_input)
            if matched is not None:
                results.append({
                    'institute_short': inst,
                    'confidence': float(probs[int(idx)]),
                    'matched_row': matched
                })
                used_insts.add(inst)
            if len(results) >= top_n:
                break

        # Second pass: if not enough, append top model predictions ignoring rules (but still unique)
        if len(results) < top_n:
            for idx in class_indices:
                inst = self.label_encoder_target.inverse_transform([int(idx)])[0]
                if inst in used_insts:
                    continue
                results.append({
                    'institute_short': inst,
                    'confidence': float(probs[int(idx)]),
                    'matched_row': None
                })
                used_insts.add(inst)
                if len(results) >= top_n:
                    break

        return results

# ----------------- CLI / main flow -----------------
def main():
    # initialize Config if needed
    if hasattr(Config, 'init_app'):
        try:
            Config.init_app(None)
        except Exception:
            pass

    if not hasattr(Config, 'TRAINING_DATA_PATH'):
        raise AttributeError("Config must define TRAINING_DATA_PATH")

    predictor = CollegePredictorFull()

    # Train if model not present, else load
    if not (hasattr(Config, 'MODEL_PATH') and os.path.exists(Config.MODEL_PATH)):
        print("No saved model found — training from scratch.")
        predictor.train(Config.TRAINING_DATA_PATH)
        predictor.save_model()
    else:
        predictor.load_model()

    # Example test: create a user input from website fields
    example_input = {
        'year': 2016,
        'institute_type': 'IIT',
        'round_no': 1,
        'quota': 'AI',               # All India
        'pool': 'Gender-Neutral',
        'category': 'SC',            # user's category
        'closing_rank': 200          # user's rank
    }

    print("Predicting top 10 valid colleges for example input...")
    top = predictor.predict_top_n(example_input, top_n=10)
    for i, item in enumerate(top, 1):
        print(f"{i}. {item['institute_short']}  (conf={item['confidence']:.3f}) matched_row={item['matched_row']}")

if __name__ == "__main__":
    main()
