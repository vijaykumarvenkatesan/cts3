import pandas as pd
import numpy as np
from typing import Dict, Tuple, Any

# Filename heuristic to assign roles
def _role(fname: str):
    f = fname.lower()
    if "beneficiary" in f: return "beneficiary"
    if "inpatient" in f: return "inpatient"
    if "outpatient" in f: return "outpatient"
    if f.startswith("test-") or "test" in f or "provider" in f: return "test"
    return "other"

def aggregate_from_user_spec(file_map: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Aggregate test CSVs exactly like the user's notebook description (provider-level features)."""
    info = {"files": list(file_map.keys())}
    bene_key = next((k for k in file_map if _role(k)=="beneficiary"), None)
    inp_key  = next((k for k in file_map if _role(k)=="inpatient"), None)
    out_key  = next((k for k in file_map if _role(k)=="outpatient"), None)
    test_key = next((k for k in file_map if _role(k)=="test"), None)

    if not (bene_key and inp_key and out_key and test_key):
        combined = pd.concat(file_map.values(), axis=0, ignore_index=True)
        info["strategy"] = "fallback_concat"
        info["shape"] = list(combined.shape)
        return combined, info

    df_test_beneficiary = file_map[bene_key].copy()
    df_test_inpatient   = file_map[inp_key].copy()
    df_test_outpatient  = file_map[out_key].copy()
    df_test             = file_map[test_key].copy()

    # Dates
    for col in ['DOB','DOD']:
        if col in df_test_beneficiary.columns:
            df_test_beneficiary[col] = pd.to_datetime(df_test_beneficiary[col], errors='coerce')
    for col in ['ClaimStartDt','ClaimEndDt','AdmissionDt','DischargeDt']:
        if col in df_test_inpatient.columns:
            df_test_inpatient[col] = pd.to_datetime(df_test_inpatient[col], errors='coerce')
        if col in df_test_outpatient.columns:
            df_test_outpatient[col] = pd.to_datetime(df_test_outpatient[col], errors='coerce')

    # Beneficiary cleanup
    chronic_cols = [c for c in df_test_beneficiary.columns if 'ChronicCond' in c]
    if 'RenalDiseaseIndicator' in df_test_beneficiary.columns:
        df_test_beneficiary['RenalDiseaseIndicator'] = df_test_beneficiary['RenalDiseaseIndicator'].replace({'Y':1,'0':0})
    for c in chronic_cols:
        df_test_beneficiary[c] = df_test_beneficiary[c].replace({2:0})

    # Age & DOD
    if 'DOB' in df_test_beneficiary.columns:
        df_test_beneficiary['Age'] = (pd.to_datetime('2009-01-01') - df_test_beneficiary['DOB']).dt.days // 365.25
    if 'DOD' in df_test_beneficiary.columns:
        df_test_beneficiary['DOD'] = df_test_beneficiary['DOD'].fillna(pd.to_datetime('2009-12-31'))

    # Merge claims
    test_all_claims = pd.concat([df_test_inpatient, df_test_outpatient], ignore_index=True, sort=False)

    # Merge with beneficiary
    if 'BeneID' in test_all_claims.columns and 'BeneID' in df_test_beneficiary.columns:
        test_claims_beneficiary = pd.merge(test_all_claims, df_test_beneficiary, on='BeneID', how='left')
    else:
        test_claims_beneficiary = test_all_claims.copy()

    # Durations
    if 'ClaimEndDt' in test_claims_beneficiary.columns and 'ClaimStartDt' in test_claims_beneficiary.columns:
        test_claims_beneficiary['ClaimDuration'] = (test_claims_beneficiary['ClaimEndDt'] - test_claims_beneficiary['ClaimStartDt']).dt.days
    if 'DischargeDt' in test_claims_beneficiary.columns and 'AdmissionDt' in test_claims_beneficiary.columns:
        test_claims_beneficiary['InpatientStayDuration'] = (test_claims_beneficiary['DischargeDt'] - test_claims_beneficiary['AdmissionDt']).dt.days

    # Provider-level aggregate
    def _agg(df):
        cols = {}
        if 'ClaimID' in df.columns: cols['TotalClaims'] = ('ClaimID','nunique')
        if 'AdmissionDt' in df.columns:
            cols['TotalInpatientClaims'] = ('AdmissionDt', lambda x: x.notna().sum())
            cols['TotalOutpatientClaims'] = ('AdmissionDt', lambda x: x.isna().sum())
        for c in ['InscClaimAmtReimbursed','DeductibleAmtPaid','Age','Gender','Race','RenalDiseaseIndicator','ClaimDuration','InpatientStayDuration']:
            if c in df.columns:
                if c in ['InscClaimAmtReimbursed','DeductibleAmtPaid']:
                    cols[f"Sum{c}"] = (c,'sum')
                    cols[f"Avg{c}"] = (c,'mean')
                else:
                    cols[f"Avg{c}"] = (c,'mean')
        for c in [c for c in df.columns if 'ChronicCond' in c]:
            cols[f"Avg{c}"] = (c,'mean')
        for c in ['BeneID','AttendingPhysician','OperatingPhysician','OtherPhysician']:
            if c in df.columns:
                cols[f"Unique{c}s"] = (c,'nunique')
                cols[f"PropMissing{c}"] = (c, lambda x: x.isnull().mean())
        gb = df.groupby('Provider').agg(**cols).reset_index() if 'Provider' in df.columns else df.copy()
        return gb

    provider_features = _agg(test_claims_beneficiary)

    # Merge with df_test
    if 'Provider' in df_test.columns and 'Provider' in provider_features.columns:
        test_data_final = pd.merge(df_test, provider_features, on='Provider', how='left')
    else:
        test_data_final = provider_features.copy()

    # Missing handling
    if 'AvgDeductibleAmtPaid' in test_data_final.columns:
        med = test_data_final['AvgDeductibleAmtPaid'].median()
        test_data_final['AvgDeductibleAmtPaid'] = test_data_final['AvgDeductibleAmtPaid'].fillna(med)
    if 'AvgInpatientStayDuration' in test_data_final.columns:
        test_data_final['AvgInpatientStayDuration'] = test_data_final['AvgInpatientStayDuration'].fillna(0)

    info.update({"strategy":"user_provider_aggregation","shape":list(test_data_final.shape)})
    return test_data_final, info

def preprocess_for_model(df: pd.DataFrame, encoders=None, metadata=None):
    """Use metadata['model_info']['features_used'] and encoders to produce numeric X.
       Returns X, available_features, optimal_threshold
    """
    features = None
    threshold = 0.5
    if isinstance(metadata, dict):
        model_info = metadata.get('model_info', {})
        features = model_info.get('features_used', None)
        threshold = float(model_info.get('optimal_threshold', 0.5))
    if features is None:
        features = list(df.columns)

    available = [f for f in features if f in df.columns]
    X = df[available].copy()

    # Convert datelike object columns
    def to_num_date(series):
        s = pd.to_datetime(series, errors='coerce')
        return (s.astype('int64') // 10**9).astype('float64')

    for col in X.columns:
        if X[col].dtype == "object":
            if any(k in col.lower() for k in ["dt","date","dob","dod","admission","discharge","start","end"]):
                X[col] = to_num_date(X[col])
            else:
                if isinstance(encoders, dict) and col in encoders:
                    le = encoders[col]
                    vals = X[col].astype(str)
                    known = set(getattr(le,"classes_", []))
                    unseen = set(vals.unique()) - known
                    if unseen and len(getattr(le,"classes_", []))>0:
                        vals = vals.replace(list(unseen), le.classes_[0])
                    try:
                        X[col] = le.transform(vals.astype(str))
                    except Exception:
                        X[col] = pd.factorize(vals)[0]
                else:
                    X[col] = pd.factorize(X[col].astype(str))[0]

    for col in X.columns:
        if not np.issubdtype(X[col].dtype, np.number):
            X[col] = pd.to_numeric(X[col], errors='coerce')

    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return X.astype('float64'), available, threshold

def build_results_dataframe(df_original: pd.DataFrame, available_features, proba, std_pred, opt_pred, threshold):
    res = pd.DataFrame()
    for name in ['ProviderID','Provider_ID','providerid','provider_id','Provider','PROVIDER_ID']:
        if name in df_original.columns:
            res['ProviderID'] = df_original[name].copy()
            break
    for f in available_features:
        res[f] = df_original[f].copy()

    res['Fraud_Probability'] = proba
    # keep model's optimized prediction but make sure it's integer
    res['Predicted_Optimized'] = np.asarray(opt_pred).astype(int)

    # Risk level buckets (include_lowest to capture 0 properly)
    res['Risk_Level'] = pd.cut(
        res['Fraud_Probability'],
        bins=[0, 0.3, 0.7, 0.9, 1.0],
        labels=['Low', 'Medium', 'High', 'Critical'],
        include_lowest=True
    )

    # Force positive prediction for High & Critical
    res.loc[res['Risk_Level'].isin(['High', 'Critical']), 'Predicted_Optimized'] = 1

    # optional: set a simple PotentialFraud column immediately
    res['PotentialFraud'] = np.where(res['Predicted_Optimized'] == 1, 'Yes', 'No')

    res['Confidence'] = pd.cut(
        res['Fraud_Probability'],
        bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
        labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'],
        include_lowest=True
    )
    return res

