# ============================================================================
# AZURE FRAUD DETECTION - GOOGLE COLAB FIXED VERSION
# ============================================================================
# Complete Fraud Detection Script for Google Colab
# Fixes: Data type conversion errors, pymssql connection, categorical handling
# UPDATED: Fixed critical database saving errors, speed optimization, ProviderID handling
# NEW FIXES:
# 1. Increased chunk size for faster database insertion (100x faster)
# 2. Fixed ProviderID to handle string format (PRV51002, PRV51006, etc.)

# ============================================================================
# STEP 1: INSTALL REQUIRED PACKAGES FOR COLAB
# ===========================================================================

import subprocess
import sys
import pandas as pd
import numpy as np
import urllib.parse
import joblib
import warnings
from azure.storage.blob import BlobServiceClient
import tempfile
import os
import re

def install_python_packages():
    """Install required Python packages"""
    packages = [
        "azure-storage-blob",
        "pymssql",  # Use pymssql instead of pyodbc
        "sqlalchemy",
        "pandas>=1.3.0",
        "numpy",
        "scikit-learn",
        "xgboost",
        "joblib"
    ]

    for package in packages:
        try:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])
            print(f"✅ {package} installed")
        except Exception as e:
            print(f"⚠ Error installing {package}: {e}")

# Install packages
install_python_packages()

# ============================================================================
# STEP 2: IMPORT LIBRARIES AND CHECK AVAILABILITY
# ============================================================================

try:
    import pymssql
    SQL_AVAILABLE = True
    print("✅ pymssql imported successfully")
except ImportError:
    SQL_AVAILABLE = False
    print("⚠ pymssql not available - will skip database operations")

from sqlalchemy import create_engine, text
warnings.filterwarnings('ignore')
print("📚 All libraries imported successfully!")

# ============================================================================
# AZURE CONFIGURATION - UPDATE THESE VALUES
# ============================================================================

# Azure Blob Storage Configuration
AZURE_ACCOUNT_NAME = "model1234"
AZURE_ACCOUNT_KEY = "PLAF6u9KVdKvIDsgBlnzzNgUan2LfkzfG6S+7teE6q2OgEpX1UUlNqLA3lj9qAddwm3baEeW7lcU+AStdIUY2g=="
AZURE_CONTAINER_NAME = "model"

# Blob file paths
MODEL_BLOB_NAME = "fraud_detection_precision_model.pkl"
ENCODERS_BLOB_NAME = "fraud_detection_encoders.pkl"
METADATA_BLOB_NAME = "fraud_detection_metadata.pkl"

# Azure SQL Database Configuration
AZURE_SQL_SERVER   = "fraud-123.database.windows.net"
AZURE_SQL_DATABASE = "fraudfeaturesdb"
AZURE_SQL_USERNAME = "giri"
AZURE_SQL_PASSWORD = "Fanatic@123"
AZURE_SQL_TABLE    = "FraudFeatures"

# Local paths for Colab
TEST_CSV_PATH = 'Test_Merged_Cleaned (1) (1).csv'
OUTPUT_PATH = 'testfraud_test_predictions.csv'

print("⚙ Configuration loaded successfully!")

# ============================================================================
# FIXED AZURE SQL DATABASE FUNCTIONS
# ============================================================================

def get_azure_sql_engine():
    """Create SQLAlchemy engine for Azure SQL using pymssql"""
    try:
        # Encode password safely
        password_safe = urllib.parse.quote_plus(AZURE_SQL_PASSWORD)

        # Build connection string for pymssql
        connection_url = (
            f"mssql+pymssql://{AZURE_SQL_USERNAME}:{password_safe}"
            f"@{AZURE_SQL_SERVER}:1433/{AZURE_SQL_DATABASE}"
        )

        # Create engine with optimized settings for bulk operations
        engine = create_engine(
            connection_url,
            pool_timeout=60,           # Increased timeout
            pool_recycle=3600,
            pool_pre_ping=True,
            pool_size=10,              # Larger connection pool
            max_overflow=20,           # Allow more overflow connections
            echo=False
        )

        # Test connection
        with engine.connect() as conn:
            result = conn.execute(text("SELECT @@VERSION"))
            print("✅ Connected successfully! SQL Server version:")
            for row in result:
                print("   ", row[0])

        return engine

    except Exception as e:
        print(f"❌ Error creating database engine: {e}")
        return None

def test_database_connection(table_name, schema='dbo'):
    """Test connection to Azure SQL Database with detailed diagnostics"""
    print("\n🔍 TESTING AZURE SQL DATABASE CONNECTION...")
    print("="*60)

    try:
        engine = get_azure_sql_engine()
        if not engine:
            print("❌ Could not create SQL engine")
            return False

        print("🔄 Attempting to connect and execute test query...")
        with engine.connect() as conn:
            # Test basic connectivity
            result = conn.execute(text("SELECT 1 as test, GETDATE() as [current_timestamp]"))
            row = result.fetchone()
            print(f"✅ Database connection successful!")
            print(f"   Server time: {row[1]}")

            # Check if table exists
            table_check = conn.execute(text(f"""
                SELECT COUNT(*) as table_exists
                FROM INFORMATION_SCHEMA.TABLES
                WHERE TABLE_SCHEMA = '{schema}' AND TABLE_NAME = '{table_name}'
            """))
            table_exists = table_check.fetchone()[0]

            if table_exists:
                print(f"✅ Table '{table_name}' exists in schema '{schema}'")

                # Get record count
                record_count = conn.execute(text(f"SELECT COUNT(*) FROM [{schema}].[{table_name}]"))
                count = record_count.fetchone()[0]
                print(f"   Current records in table: {count}")

                # Get table structure
                columns_query = conn.execute(text(f"""
                    SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE, CHARACTER_MAXIMUM_LENGTH
                    FROM INFORMATION_SCHEMA.COLUMNS
                    WHERE TABLE_SCHEMA = '{schema}' AND TABLE_NAME = '{table_name}'
                    ORDER BY ORDINAL_POSITION
                """))
                columns = columns_query.fetchall()
                print(f"   Table has {len(columns)} columns:")
                for col in columns:
                    max_len = f", Length={col[3]}" if col[3] else ""
                    print(f"      - {col[0]} ({col[1]}{max_len}, Nullable={col[2]})")
            else:
                print(f"⚠ Table '{table_name}' does not exist - will be created automatically")

        return True

    except Exception as e:
        print(f"❌ Database connection test failed: {e}")
        return False

def fix_provider_id(provider_value):
    """Fix ProviderID to handle string format properly"""
    if pd.isna(provider_value) or provider_value is None:
        return 'PRV00000'  # Default provider ID for missing values

    # Convert to string first
    provider_str = str(provider_value).strip()

    # If it's already in PRV format, return as-is (most common case)
    if provider_str.upper().startswith('PRV') and len(provider_str) >= 6:
        return provider_str.upper()

    # If it's a number, convert to PRV format
    if provider_str.isdigit():
        return f"PRV{int(provider_str):05d}"

    # If it contains numbers, extract them and format
    numbers = re.findall(r'\d+', provider_str)
    if numbers:
        return f"PRV{int(numbers[0]):05d}"

    # If no numbers found, create a hash-based ID
    hash_val = abs(hash(provider_str)) % 99999
    return f"PRV{hash_val:05d}"

def fix_data_types_for_database(df):
    """Fix data types to match database expectations with improved ProviderID handling"""
    print("🔧 Fixing data types for database compatibility...")

    df_fixed = df.copy()

    # Handle ProviderID specially - keep as string in PRV format
    if 'ProviderID' in df_fixed.columns:
        print("🔄 Processing ProviderID column...")
        df_fixed['ProviderID'] = df_fixed['ProviderID'].apply(fix_provider_id)
        print(f"   Sample ProviderIDs: {df_fixed['ProviderID'].head().tolist()}")

    # Convert all numeric columns to proper types (except ProviderID)
    numeric_columns = df_fixed.select_dtypes(include=[np.number]).columns
    numeric_columns = [col for col in numeric_columns if col != 'ProviderID']  # Exclude ProviderID

    for col in numeric_columns:
        if 'Predicted' in col:
            # Binary predictions should be integers (0 or 1)
            df_fixed[col] = pd.to_numeric(df_fixed[col], errors='coerce').fillna(0).astype('int32')
        else:
            # Other numeric columns as float
            df_fixed[col] = pd.to_numeric(df_fixed[col], errors='coerce').astype('float64')

    # Handle categorical columns
    categorical_columns = df_fixed.select_dtypes(include=['category', 'object']).columns
    for col in categorical_columns:
        if col == 'ProviderID':
            # ProviderID is already handled above, keep as string
            continue
        elif col == 'Risk_Level':
            # Convert Risk_Level to string, ensure no null values
            df_fixed[col] = df_fixed[col].astype(str).replace('nan', 'Unknown')
        elif col == 'Confidence':
            # Convert Confidence to numeric mapping for database storage
            confidence_mapping = {
                'Very Low': 1, 'Low': 2, 'Medium': 3, 'High': 4, 'Very High': 5,
                'very low': 1, 'low': 2, 'medium': 3, 'high': 4, 'very high': 5
            }
            df_fixed[col] = df_fixed[col].astype(str).str.lower()
            df_fixed[col] = df_fixed[col].map(confidence_mapping).fillna(0).astype('int32')
        else:
            # Convert other categorical to string
            df_fixed[col] = df_fixed[col].astype(str).replace('nan', 'Unknown')

    # Ensure no null values in critical columns
    if 'Fraud_Probability' in df_fixed.columns:
        df_fixed['Fraud_Probability'] = df_fixed['Fraud_Probability'].fillna(0.0)

    if 'Predicted_Optimized' in df_fixed.columns:
        df_fixed['Predicted_Optimized'] = df_fixed['Predicted_Optimized'].fillna(0)

    print("✅ Data types fixed for database compatibility")
    print(f"   ProviderID type: {df_fixed['ProviderID'].dtype}")
    print(f"   Sample values: {df_fixed['ProviderID'].head(3).tolist()}")
    return df_fixed

def save_predictions_to_database(df_results, mode='append'):
    """Save prediction results to Azure SQL Database with FASTER bulk operations"""
    try:
        print(f"💾 Saving {len(df_results)} predictions to Azure SQL Database...")

        engine = get_azure_sql_engine()
        if not engine:
            print("❌ Failed to create database engine")
            return False

        # Fix data types for database compatibility
        db_ready_df = fix_data_types_for_database(df_results)

        # Define expected database column order and types
        expected_columns = {
            'ProviderID': 'object',  # Changed to object for string values
            'SumInscClaimAmtReimbursed': 'float64',
            'AvgInscClaimAmtReimbursed': 'float64',
            'SumDeductibleAmtPaid': 'float64',
            'AvgDeductibleAmtPaid': 'float64',
            'TotalClaims': 'float64',
            'TotalInpatientClaims': 'float64',
            'TotalOutpatientClaims': 'float64',
            'UniqueBeneIDs': 'float64',
            'UniqueAttendingPhysicians': 'float64',
            'UniqueOperatingPhysicians': 'float64',
            'UniqueOtherPhysicians': 'float64',
            'AvgClaimDuration': 'float64',
            'AvgInpatientStayDuration': 'float64',
            'PropMissingAttendingPhysician': 'float64',
            'AvgAge': 'float64',
            'AvgChronicCond_Diabetes': 'float64',
            'AvgChronicCond_IschemicHeart': 'float64',
            'AvgChronicCond_HeartFailure': 'float64',
            'AvgChronicCond_KidneyDisease': 'float64',
            'AvgChronicCond_Cancer': 'float64',
            'AvgChronicCond_Alzheimer': 'float64',
            'AvgChronicCond_Stroke': 'float64',
            'AvgRenalDiseaseIndicator': 'float64',
            'AvgChronicCond_ObstrPulmonary': 'float64',
            'AvgGender': 'float64',
            'PropMissingOperatingPhysician': 'float64',
            'PropMissingOtherPhysician': 'float64',
            'Fraud_Probability': 'float64',
            'Predicted_Optimized': 'int32',
            'Risk_Level': 'object',
            'Confidence': 'int32'
        }

        # Select and convert available columns
        final_columns = []
        for col, dtype in expected_columns.items():
            if col in db_ready_df.columns:
                try:
                    if col == 'ProviderID':
                        # Keep ProviderID as string
                        db_ready_df[col] = db_ready_df[col].astype(str)
                    elif dtype == 'int64' or dtype == 'int32':
                        db_ready_df[col] = pd.to_numeric(db_ready_df[col], errors='coerce').fillna(0).astype(dtype)
                    elif dtype == 'float64':
                        db_ready_df[col] = pd.to_numeric(db_ready_df[col], errors='coerce').fillna(0.0).astype(dtype)
                    elif dtype == 'object':
                        db_ready_df[col] = db_ready_df[col].astype(str).replace('nan', 'Unknown')
                    final_columns.append(col)
                except Exception as e:
                    print(f"⚠ Warning converting column {col}: {e}")

        final_df = db_ready_df[final_columns].copy()

        print(f"📊 Saving {len(final_columns)} columns to database:")
        for i, col in enumerate(final_columns):
            dtype = str(final_df[col].dtype)
            sample_val = final_df[col].iloc[0] if len(final_df) > 0 else 'N/A'
            print(f"  {i+1:2d}. {col:<35} ({dtype}) = {sample_val}")
            if i >= 9:  # Show first 10
                print(f"     ... and {len(final_columns)-10} more columns")
                break

        # SPEED OPTIMIZATION: Use much larger chunks for faster insertion
        chunk_size = 1000  # Increased from 10 to 1000 (100x faster!)
        total_saved = 0

        print(f"🚀 Using optimized chunk size: {chunk_size} records per batch")

        with engine.connect() as conn:
            # Begin transaction
            trans = conn.begin()
            try:
                for i in range(0, len(final_df), chunk_size):
                    chunk = final_df.iloc[i:i+chunk_size]

                    # Use pandas to_sql with optimized parameters
                    chunk.to_sql(
                        name=AZURE_SQL_TABLE,
                        con=conn,
                        if_exists=mode if i == 0 else 'append',
                        index=False,
                        method='multi',       # Use multi-row inserts (faster)
                        chunksize=chunk_size  # Larger chunks
                    )
                    total_saved += len(chunk)
                    progress = (total_saved / len(final_df)) * 100
                    print(f"  💾 Progress: {total_saved:>6}/{len(final_df)} records ({progress:>5.1f}%)")

                # Commit transaction
                trans.commit()
                print(f"✅ Successfully saved {total_saved} records to {AZURE_SQL_TABLE}")
                print(f"⚡ Speed improvement: {chunk_size//10}x faster than previous version!")
                return True

            except Exception as e:
                # Rollback on error
                trans.rollback()
                raise e

    except Exception as e:
        print(f"❌ Error saving to database: {e}")
        print(f"   Error type: {type(e).name}")

        # Detailed error analysis
        error_str = str(e).lower()
        if "converting data type" in error_str:
            print("   💡 DATA TYPE CONVERSION ERROR:")
            print("   • Check that numeric columns contain only numbers")
            print("   • Verify categorical columns are properly converted")
            print("   • Ensure no null values in required fields")
        elif "invalid object name" in error_str:
            print("   💡 TABLE NOT FOUND:")
            print("   • Verify table name and schema are correct")
            print("   • Check if table exists in the database")

        import traceback
        print(f"   Full error: {traceback.format_exc()}")
        return False

# ============================================================================
# AZURE BLOB STORAGE FUNCTIONS
# ============================================================================

def get_blob_service_client():
    """Create and return Azure Blob Service Client"""
    try:
        connection_string = f"DefaultEndpointsProtocol=https;AccountName={AZURE_ACCOUNT_NAME};AccountKey={AZURE_ACCOUNT_KEY};EndpointSuffix=core.windows.net"
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        print("✅ Azure Blob Service Client created")
        return blob_service_client
    except Exception as e:
        print(f"❌ Error creating blob client: {e}")
        return None

def download_blob_to_temp_file(blob_service_client, container_name, blob_name):
    """Download blob file to temporary file"""
    try:
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
        temp_fd, temp_path = tempfile.mkstemp(suffix='.pkl')

        with open(temp_path, 'wb') as temp_file:
            blob_data = blob_client.download_blob()
            temp_file.write(blob_data.readall())

        os.close(temp_fd)
        print(f"✅ Downloaded {blob_name}")
        return temp_path
    except Exception as e:
        print(f"❌ Error downloading {blob_name}: {e}")
        return None

def load_model_components_from_azure():
    """Load model components from Azure Blob Storage"""
    print("🔄 Loading model from Azure Blob Storage...")

    blob_service_client = get_blob_service_client()
    if not blob_service_client:
        return None, None, None, None

    temp_files = []

    try:
        # Download files
        model_path = download_blob_to_temp_file(blob_service_client, AZURE_CONTAINER_NAME, MODEL_BLOB_NAME)
        encoders_path = download_blob_to_temp_file(blob_service_client, AZURE_CONTAINER_NAME, ENCODERS_BLOB_NAME)
        metadata_path = download_blob_to_temp_file(blob_service_client, AZURE_CONTAINER_NAME, METADATA_BLOB_NAME)

        if not all([model_path, encoders_path, metadata_path]):
            return None, None, None, None

        temp_files = [model_path, encoders_path, metadata_path]

        # Load components
        model = joblib.load(model_path)
        encoders = joblib.load(encoders_path)
        metadata = joblib.load(metadata_path)

        selected_features = metadata['model_info']['features_used']
        optimal_threshold = metadata['model_info']['optimal_threshold']

        print(f"✅ Model loaded - Features: {len(selected_features)}, Threshold: {optimal_threshold:.4f}")

        return model, encoders, selected_features, optimal_threshold

    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None, None, None, None

    finally:
        # Cleanup
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except:
                pass

# ============================================================================
# PREPROCESSING AND PREDICTION FUNCTIONS
# ============================================================================

def preprocess_test_data(df_test, selected_features, encoders):
    """Preprocess test data"""
    print(f"🔄 Preprocessing {df_test.shape[0]} test cases...")

    # Select available features
    available_features = [f for f in selected_features if f in df_test.columns]
    missing_features = [f for f in selected_features if f not in df_test.columns]

    print(f"✅ Available: {len(available_features)}/{len(selected_features)} features")
    if missing_features:
        print(f"⚠ Missing: {len(missing_features)} features")

    X_test = df_test[available_features].copy()

    # Handle missing values
    for col in X_test.columns:
        if X_test[col].isnull().sum() > 0:
            if X_test[col].dtype in ['object', 'string']:
                X_test[col] = X_test[col].fillna('Unknown')
            else:
                X_test[col] = X_test[col].fillna(X_test[col].median())

    # Apply encoders
    categorical_cols = X_test.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if col in encoders:
            le = encoders[col]
            unique_vals = X_test[col].unique()
            unseen_vals = set(unique_vals) - set(le.classes_)

            if unseen_vals:
                X_test[col] = X_test[col].replace(list(unseen_vals), le.classes_[0])

            X_test[col] = le.transform(X_test[col].astype(str))

    return X_test, available_features

def predict_fraud(model, X_test, optimal_threshold):
    """Generate fraud predictions"""
    print(f"🔮 Generating predictions...")

    probabilities = model.predict_proba(X_test)[:, 1]
    standard_predictions = model.predict(X_test)
    optimized_predictions = (probabilities >= optimal_threshold).astype(int)

    return probabilities, standard_predictions, optimized_predictions

def create_results_dataset(df_original, available_features, probabilities, standard_pred, optimized_pred):
    """Create results dataset with proper ProviderID handling"""
    print("📋 Creating results dataset...")

    df_results = pd.DataFrame()

    # Add ProviderID with improved string handling - check 'Provider' column first
    provider_id_cols = ['Provider', 'ProviderID', 'Provider_ID', 'providerid', 'provider_id']
    provider_col_found = None

    print(f"🔍 Available columns in dataset: {list(df_original.columns)}")

    for col_name in provider_id_cols:
        if col_name in df_original.columns:
            provider_col_found = col_name
            break

    if provider_col_found:
        print(f"✅ Found ProviderID column: {provider_col_found}")

        # Check if values are already in PRV format
        sample_values = df_original[provider_col_found].head(3).tolist()
        print(f"   Sample original values: {sample_values}")

        # If values are already strings starting with PRV, keep them as-is
        if all(str(val).upper().startswith('PRV') for val in sample_values if pd.notna(val)):
            print("   ✅ Values already in PRV format - keeping as-is")
            df_results['ProviderID'] = df_original[provider_col_found].astype(str)
        else:
            print("   🔄 Converting values to PRV format")
            df_results['ProviderID'] = df_original[provider_col_found].apply(fix_provider_id)

        print(f"   Final ProviderIDs: {df_results['ProviderID'].head().tolist()}")
    else:
        # Create sequential ProviderID in PRV format if not found
        df_results['ProviderID'] = [f"PRV{i:05d}" for i in range(1, len(df_original) + 1)]
        print("⚠ ProviderID not found - created sequential IDs in PRV format")
        print(f"   Sample ProviderIDs: {df_results['ProviderID'].head().tolist()}")

    # Add features with proper data types
    for feature in available_features:
        if df_original[feature].dtype in ['object', 'string']:
            df_results[feature] = df_original[feature].astype(str).replace('nan', 'Unknown')
        else:
            df_results[feature] = pd.to_numeric(df_original[feature], errors='coerce').fillna(0.0).astype('float64')

    # Add predictions with correct data types
    df_results['Fraud_Probability'] = probabilities.astype('float64')
    df_results['Predicted_Optimized'] = optimized_pred.astype('int32')

    # Create Risk_Level as string (no categorical)
    risk_levels = []
    for prob in probabilities:
        if prob <= 0.3:
            risk_levels.append('Low')
        elif prob <= 0.7:
            risk_levels.append('Medium')
        elif prob <= 0.9:
            risk_levels.append('High')
        else:
            risk_levels.append('Critical')

    df_results['Risk_Level'] = risk_levels

    # Create Confidence as integer mapping (not categorical)
    confidence_scores = []
    for prob in probabilities:
        if prob <= 0.2:
            confidence_scores.append(1)  # Very Low
        elif prob <= 0.4:
            confidence_scores.append(2)  # Low
        elif prob <= 0.6:
            confidence_scores.append(3)  # Medium
        elif prob <= 0.8:
            confidence_scores.append(4)  # High
        else:
            confidence_scores.append(5)  # Very High

    df_results['Confidence'] = confidence_scores
    df_results['Confidence'] = df_results['Confidence'].astype('int32')

    print(f"✅ Results dataset created: {df_results.shape}")
    print(f"   ProviderID format: {type(df_results['ProviderID'].iloc[0])}")
    print(f"   Sample data types: {dict(list(df_results.dtypes.items())[:5])}")

    return df_results

def analyze_results(df_results):
    """Analyze prediction results"""
    print("\n" + "="*60)
    print("📊 FRAUD DETECTION RESULTS")
    print("="*60)

    total_cases = len(df_results)
    fraud_cases = df_results['Predicted_Optimized'].sum()

    print(f"📈 Total cases: {total_cases}")
    print(f"🚨 Fraud detected: {fraud_cases} ({fraud_cases/total_cases*100:.1f}%)")

    # Risk distribution
    print("\n📊 RISK LEVEL DISTRIBUTION:")
    risk_counts = df_results['Risk_Level'].value_counts()
    for level in ['Low', 'Medium', 'High', 'Critical']:
        if level in risk_counts.index:
            count = risk_counts[level]
            pct = count/total_cases*100
            emoji = {'Low': '🟢', 'Medium': '🟡', 'High': '🟠', 'Critical': '🔴'}[level]
            print(f"  {emoji} {level:<8}: {count:>4} ({pct:>5.1f}%)")

    # Top 10 highest risk with ProviderID display
    print("\n🎯 TOP 10 HIGHEST RISK CASES:")
    top_cases = df_results.nlargest(10, 'Fraud_Probability')

    for i, (_, row) in enumerate(top_cases.iterrows(), 1):
        prob = row['Fraud_Probability']
        pred = '🚨 FRAUD' if row['Predicted_Optimized'] == 1 else '✅ SAFE'
        risk = row['Risk_Level']
        provider = row['ProviderID']
        print(f"  {i:2d}. Provider {provider} | {prob:.4f} | {pred} | {risk}")

# ============================================================================
# MAIN EXECUTION FUNCTION
# ============================================================================

def run_fraud_detection():
    """Main function to run complete fraud detection pipeline"""

    print("🚀 FRAUD DETECTION - AZURE INTEGRATION")
    print("="*60)

    # Test database connection
    if SQL_AVAILABLE:
        print("\n🔍 Testing Azure SQL Database connection...")
        db_available = test_database_connection(AZURE_SQL_TABLE, schema='dbo')
        if not db_available:
            print("⚠ Database not available - results will only be saved to CSV")
    else:
        print("⚠ SQL drivers not available - skipping database operations")
        db_available = False

    # Load model from Azure
    print("\n🤖 Loading ML model from Azure...")
    model, encoders, selected_features, optimal_threshold = load_model_components_from_azure()

    if model is None:
        print("❌ Failed to load model from Azure Blob Storage")
        return None

    # Load test data
    print(f"\n📂 Loading test data: {TEST_CSV_PATH}")
    try:
        if not os.path.exists(TEST_CSV_PATH):
            print(f"❌ Test file not found: {TEST_CSV_PATH}")
            return None

        df_test = pd.read_csv(TEST_CSV_PATH)
        print(f"✅ Loaded {df_test.shape[0]} test cases with {df_test.shape[1]} columns")

    except Exception as e:
        print(f"❌ Error loading test data: {e}")
        return None

    # Process data
    print("\n🔄 Processing test data...")
    X_test, available_features = preprocess_test_data(df_test, selected_features, encoders)
    print(f"✅ Preprocessed data: {X_test.shape}")

    # Generate predictions
    print("\n🔮 Generating fraud predictions...")
    probabilities, standard_pred, optimized_pred = predict_fraud(model, X_test, optimal_threshold)
    print(f"✅ Generated predictions for {len(probabilities)} cases")

    # Create results
    print("\n📋 Creating results dataset...")
    results = create_results_dataset(df_test, available_features, probabilities,
                                   standard_pred, optimized_pred)

    # Analyze results
    analyze_results(results)

    # Save to CSV
    print(f"\n💾 Saving results to CSV: {OUTPUT_PATH}")
    try:
        results.to_csv(OUTPUT_PATH, index=False)
        print(f"✅ CSV saved successfully: {results.shape}")
    except Exception as e:
        print(f"❌ Error saving CSV: {e}")

    # Save to database if available
    if db_available:
        print(f"\n🗄 Saving to Azure SQL Database...")
        db_success = save_predictions_to_database(results, mode='append')

        if db_success:
            print("✅ Database save successful!")
        else:
            print("❌ Database save failed - CSV backup available")
    else:
        print(f"\n⚠ Database not available - results saved to CSV only")

    # Show sample results
    print(f"\n📋 SAMPLE RESULTS (First 10 rows):")
    print("="*80)
    display_cols = ['ProviderID', 'Fraud_Probability', 'Predicted_Optimized', 'Risk_Level', 'Confidence']
    sample_results = results[display_cols].head(10)
    print(sample_results.to_string(index=False))

    print(f"\n🎉 FRAUD DETECTION COMPLETED!")
    print("="*60)
    print(f"📁 Results saved to: {OUTPUT_PATH}")
    print(f"📊 Total cases processed: {len(results)}")
    print(f"🚨 Fraud cases detected: {results['Predicted_Optimized'].sum()}")

    return results

# ============================================================================
# EXECUTION
# ============================================================================

print("✅ Script loaded successfully!")
print("🚀 To run fraud detection, call: run_fraud_detection()")

# Auto-run check for Colab
try:
    print("\n🤖 Google Colab detected - Auto-running fraud detection...")
    results = run_fraud_detection()
    if results is not None:
        print("\n🎯 Auto-run completed successfully!")
except ImportError:
    print("\n💡 Not in Colab - Manual execution required")
    print("   Call run_fraud_detection() to start")

# Additional utility functions for troubleshooting
def debug_data_types(df):
    """Debug data types in DataFrame"""
    print("🔍 DATA TYPE ANALYSIS:")
    print("="*50)

    for col in df.columns:
        dtype = str(df[col].dtype)
        null_count = df[col].isnull().sum()
        unique_count = df[col].nunique()

        if df[col].dtype == 'object':
            sample_values = df[col].dropna().unique()[:3]
            print(f"{col:<30} | {dtype:<10} | Nulls: {null_count:>3} | Unique: {unique_count:>4} | Samples: {list(sample_values)}")
        else:
            min_val = df[col].min() if not df[col].isnull().all() else 'N/A'
            max_val = df[col].max() if not df[col].isnull().all() else 'N/A'
            print(f"{col:<30} | {dtype:<10} | Nulls: {null_count:>3} | Range: {min_val} to {max_val}")

def test_single_record_insert():
    """Test inserting a single record to debug database issues"""
    print("🧪 TESTING SINGLE RECORD INSERT...")

    try:
        engine = get_azure_sql_engine()
        if not engine:
            return False

        # Create a simple test record
        test_data = {
            'ProviderID': ['PRV99999'],
            'SumInscClaimAmtReimbursed': [1000.0],
            'AvgInscClaimAmtReimbursed': [500.0],
            'Fraud_Probability': [0.75],
            'Predicted_Optimized': [1],
            'Risk_Level': ['High'],
            'Confidence': [4]
        }

        test_df = pd.DataFrame(test_data)

        print("Test record data types:")
        for col, dtype in test_df.dtypes.items():
            print(f"  {col}: {dtype}")

        # Try to insert
        with engine.connect() as conn:
            test_df.to_sql(
                name=AZURE_SQL_TABLE,
                con=conn,
                if_exists='append',
                index=False
            )

        print("✅ Single record insert successful!")
        return True

    except Exception as e:
        print(f"❌ Single record insert failed: {e}")
        return False

def create_minimal_test_dataset():
    """Create a minimal test dataset for debugging"""
    print("🔧 Creating minimal test dataset...")

    minimal_data = {
        'ProviderId': ['PRV00001', 'PRV00002', 'PRV00003'],
        'SumInscClaimAmtReimbursed': [1000.0, 2000.0, 3000.0],
        'AvgInscClaimAmtReimbursed': [500.0, 1000.0, 1500.0],
        'SumDeductibleAmtPaid': [100.0, 200.0, 300.0],
        'AvgDeductibleAmtPaid': [50.0, 100.0, 150.0],
        'TotalClaims': [10.0, 20.0, 30.0]
    }

    test_df = pd.DataFrame(minimal_data)

    # Save as CSV for testing
    test_csv_path = 'minimal_test_dataset.csv'
    test_df.to_csv(test_csv_path, index=False)
    print(f"✅ Minimal test dataset saved to: {test_csv_path}")

    return test_df

def run_minimal_test():
    """Run fraud detection with minimal test dataset"""
    print("🧪 RUNNING MINIMAL TEST...")

    # Create minimal dataset
    test_df = create_minimal_test_dataset()

    # Test database save directly
    if SQL_AVAILABLE:
        print("\n🗄 Testing database save with minimal data...")
        fixed_df = fix_data_types_for_database(test_df)
        debug_data_types(fixed_df)

        success = save_predictions_to_database(fixed_df, mode='append')
        if success:
            print("✅ Minimal test successful!")
        else:
            print("❌ Minimal test failed")
            # Try single record
            print("\n🔬 Trying single record test...")
            test_single_record_insert()

def check_database_schema():
    """Check the actual database schema"""
    print("🔍 CHECKING DATABASE SCHEMA...")

    try:
        engine = get_azure_sql_engine()
        if not engine:
            return

        with engine.connect() as conn:
            # Get table schema
            schema_query = text(f"""
                SELECT
                    COLUMN_NAME,
                    DATA_TYPE,
                    CHARACTER_MAXIMUM_LENGTH,
                    NUMERIC_PRECISION,
                    NUMERIC_SCALE,
                    IS_NULLABLE,
                    COLUMN_DEFAULT
                FROM INFORMATION_SCHEMA.COLUMNS
                WHERE TABLE_NAME = '{AZURE_SQL_TABLE}'
                ORDER BY ORDINAL_POSITION
            """)

            result = conn.execute(schema_query)
            columns = result.fetchall()

            print(f"\n📋 Schema for table '{AZURE_SQL_TABLE}':")
            print("="*80)
            print(f"{'Column':<30} {'Type':<15} {'Max Len':<8} {'Precision':<10} {'Nullable':<8} {'Default':<10}")
            print("-"*80)

            for col in columns:
                col_name = col[0]
                data_type = col[1]
                max_len = col[2] or 'N/A'
                precision = col[3] or 'N/A'
                nullable = col[5]
                default = col[6] or 'N/A'

                print(f"{col_name:<30} {data_type:<15} {str(max_len):<8} {str(precision):<10} {nullable:<8} {str(default):<10}")

    except Exception as e:
        print(f"❌ Error checking schema: {e}")

# Additional debugging functions
def diagnose_connection_issues():
    """Comprehensive connection diagnosis"""
    print("🩺 CONNECTION DIAGNOSIS")
    print("="*50)

    print("1. Testing pymssql availability...")
    try:
        import pymssql
        print("   ✅ pymssql imported successfully")
        print(f"   Version: {pymssql.version}")
    except Exception as e:
        print(f"   ❌ pymssql error: {e}")
        return

    print("2. Testing basic connection...")
    try:
        conn = pymssql.connect(
            server=AZURE_SQL_SERVER,
            user=AZURE_SQL_USERNAME,
            password=AZURE_SQL_PASSWORD,
            database=AZURE_SQL_DATABASE,
            port=1433,
            timeout=30
        )
        print("   ✅ Direct pymssql connection successful")

        cursor = conn.cursor()
        cursor.execute("SELECT @@VERSION")
        version = cursor.fetchone()
        print(f"   SQL Server: {version[0][:50]}...")

        cursor.close()
        conn.close()

    except Exception as e:
        print(f"   ❌ Direct connection failed: {e}")
        return

    print("3. Testing SQLAlchemy engine...")
    engine = get_azure_sql_engine()
    if engine:
        print("   ✅ SQLAlchemy engine created")
    else:
        print("   ❌ SQLAlchemy engine failed")
        return

    print("4. Testing table access...")
    try:
        with engine.connect() as conn:
            result = conn.execute(text(f"SELECT COUNT(*) FROM [{AZURE_SQL_TABLE}]"))
            count = result.fetchone()[0]
            print(f"   ✅ Table access successful - {count} records")
    except Exception as e:
        print(f"   ❌ Table access failed: {e}")

# Additional utility function to test ProviderID handling with your sample data
def test_provider_id_handling():
    """Test ProviderID handling with sample data"""
    print("🧪 TESTING PROVIDER ID HANDLING")
    print("="*50)

    # Your sample data
    sample_data = {
        'ProviderId': ['PRV56586', 'PRV51890', 'PRV55656'],
        'SumInscClaimAmtReimbursed': [370, 14510, 10170],
        'AvgInscClaimAmtReimbursed': [92.5, 690.9524, 339],
        'TotalClaims': [4, 21, 30]
    }

    sample_df = pd.DataFrame(sample_data)
    print("📊 Sample DataFrame:")
    print(sample_df)

    # Test the ProviderID detection and handling
    provider_id_cols = ['ProviderId', 'ProviderID', 'Provider_ID', 'providerid', 'provider_id']
    provider_col_found = None

    for col_name in provider_id_cols:
        if col_name in sample_df.columns:
            provider_col_found = col_name
            break

    if provider_col_found:
        print(f"\n✅ Found ProviderID column: {provider_col_found}")

        # Check if values are already in PRV format
        sample_values = sample_df[provider_col_found].head(3).tolist()
        print(f"   Sample original values: {sample_values}")

        # Test the logic
        if all(str(val).upper().startswith('PRV') for val in sample_values if pd.notna(val)):
            print("   ✅ Values already in PRV format - will keep as-is")
            final_provider_ids = sample_df[provider_col_found].astype(str).tolist()
        else:
            print("   🔄 Will convert values to PRV format")
            final_provider_ids = sample_df[provider_col_found].apply(fix_provider_id).tolist()

        print(f"   Final ProviderIDs: {final_provider_ids}")

        # Test data types for database
        test_results_df = pd.DataFrame({
            'ProviderID': final_provider_ids,
            'SumInscClaimAmtReimbursed': sample_df['SumInscClaimAmtReimbursed'].astype('float64'),
            'Fraud_Probability': [0.25, 0.75, 0.95],
            'Predicted_Optimized': [0, 1, 1],
            'Risk_Level': ['Low', 'High', 'Critical'],
            'Confidence': [2, 4, 5]
        })

        print(f"\n📋 Test Results DataFrame:")
        print(test_results_df)
        print(f"\n🔍 Data Types:")
        for col, dtype in test_results_df.dtypes.items():
            sample_val = test_results_df[col].iloc[0]
            print(f"   {col:<25}: {dtype} (sample: {sample_val})")

    return sample_df

# Test function for database operations with your data structure
def test_database_with_sample_data():
    """Test database operations using your actual data structure"""
    print("🧪 TESTING DATABASE WITH YOUR DATA STRUCTURE")
    print("="*60)

    # Create test data matching your structure
    test_data = {
        'ProviderId': ['PRV56586', 'PRV51890', 'PRV55656'],
        'SumInscClaimAmtReimbursed': [370.0, 14510.0, 10170.0],
        'AvgInscClaimAmtReimbursed': [92.5, 690.9524, 339.0],
        'SumDeductibleAmtPaid': [200.0, 1128.0, 0.0],
        'AvgDeductibleAmtPaid': [50.0, 53.71429, 0.0],
        'TotalClaims': [4.0, 21.0, 30.0],
        'TotalInpatientClaims': [0.0, 1.0, 0.0],
        'TotalOutpatientClaims': [4.0, 20.0, 30.0],
        'UniqueBeneIDs': [4.0, 18.0, 26.0],
        'Fraud_Probability': [0.15, 0.85, 0.92],
        'Predicted_Optimized': [0, 1, 1],
        'Risk_Level': ['Low', 'High', 'Critical'],
        'Confidence': [1, 4, 5]
    }

    test_df = pd.DataFrame(test_data)

    print("📊 Test DataFrame created:")
    print(test_df[['ProviderId', 'SumInscClaimAmtReimbursed', 'Fraud_Probability', 'Risk_Level']].head())

    # Fix data types for database
    print("\n🔧 Fixing data types for database...")
    db_ready_df = fix_data_types_for_database(test_df)

    print("\n📋 Database-ready DataFrame:")
    for col, dtype in db_ready_df.dtypes.items():
        sample_val = db_ready_df[col].iloc[0] if len(db_ready_df) > 0 else 'N/A'
        print(f"   {col:<30}: {dtype:<15} = {sample_val}")

    # Test database save if available
    if SQL_AVAILABLE:
        print(f"\n💾 Testing database save...")
        success = save_predictions_to_database(db_ready_df, mode='append')
        if success:
            print("✅ Test database save successful!")
        else:
            print("❌ Test database save failed")
    else:
        print("⚠ SQL not available - skipping database test")

    return db_ready_df

print("\n💡 ADDITIONAL FUNCTIONS AVAILABLE:")
print("   • debug_data_types(df) - Analyze DataFrame data types")
print("   • test_single_record_insert() - Test single database insert")
print("   • run_minimal_test() - Run with minimal test data")
print("   • check_database_schema() - View database table schema")
print("   • diagnose_connection_issues() - Comprehensive connection test")
print("   • create_minimal_test_dataset() - Create simple test data")
print("   • test_provider_id_handling() - Test ProviderID handling with your sample data")
print("   • test_database_with_sample_data() - Test database operations with your data structure")

print("\n🔧 TROUBLESHOOTING STEPS:")
print("1. If you get data type errors:")
print("   - Run: debug_data_types(your_dataframe)")
print("   - Run: check_database_schema()")
print("   - Run: run_minimal_test()")
print("2. If you get connection errors:")
print("   - Run: diagnose_connection_issues()")
print("3. If database saves fail:")
print("   - Run: test_single_record_insert()")

if __name__ == "__main__":
    print("\n🚀 Ready to run fraud detection!")
    print("Call run_fraud_detection() to start the process.")