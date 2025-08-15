import pandas as pd
import os
from sqlalchemy import create_engine, text
import awswrangler as wr
import numpy as np
from sklearn.cluster import DBSCAN
import boto3
from datetime import datetime, timezone, timedelta
import sys
import io
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback
import googlemaps
import json
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

    # Make sure to set the correct URI string
def vin_loan_address():
    prod_uri = os.environ.get('PROD_DATABASE_URI')
    if not prod_uri:
        raise ValueError("PROD_DATABASE_URI environment variable not set")
    
    logger.info("Starting vin_loan_address function...")


    def connect_to_database(db_uri: str):
        try:
            # Add connect_args to change the SSL verification mode
            engine = create_engine(
                db_uri,
                connect_args={'sslmode': 'verify-ca'} # This still verifies the certificate is valid, but not the hostname
            )
            with engine.connect() as connection:
                logger.info("✅ Connection successful!")
                
                result = connection.execute(text("""
                WITH cte1 as 
        (SELECT 
            ld.lms_id AS lms_id, 
            A.address_line_1 as Permanent_address_1,
            A.address_line_2 as Permanent_address_2,
            A.city as Peramanent_address_city,
            A.state as Peramanent_address_state,
            A.pincode as permanant_pincode
        FROM leads_service_leads_service_loan_details ld 
        LEFT JOIN leads_service_leads_service_credit_profile CP
            ON LD.uuid = CP.loan_details_id AND CP.is_active = 1
            AND CP.profile_type = 'APPLICANT'
        LEFT JOIN leads_service_leads_service_loan_user_profile_address_mapping LUPAM 
            ON LUPAM.LOAN_DETAILS_UUID = LD.UUID AND LUPAM.address_type = 'PERMANENT_RESIDENCE' and LUPAM.user_profile_uuid = cp.user_profile_uuid
        LEFT JOIN LEADS_SERVICE_LEADS_SERVICE_ADDRESS A
            ON A.UUID = LUPAM.address_uuid
    ), cte2 as
    (SELECT 
            ld.lms_id AS lms_id, 
            A.address_line_1 as gurantor_address_1,
            A.address_line_2 as gurantor_address_2,
            A.city as gurantor_address_city,
            A.state as gurantor_address_state,
            A.pincode as gurantor_pincode
        FROM leads_service_leads_service_loan_details ld 
        LEFT JOIN leads_service_leads_service_credit_profile CP
            ON LD.uuid = CP.loan_details_id AND CP.is_active = 1
            AND CP.profile_type = 'GURANTOR'
        LEFT JOIN leads_service_leads_service_loan_user_profile_address_mapping LUPAM 
            ON LUPAM.LOAN_DETAILS_UUID = LD.UUID AND LUPAM.address_type = 'CURRENT_RESIDENCE' and LUPAM.user_profile_uuid = cp.user_profile_uuid
        LEFT JOIN LEADS_SERVICE_LEADS_SERVICE_ADDRESS A
            ON A.UUID = LUPAM.address_uuid
    ),
    cte3 as
    (SELECT 
            ld.lms_id AS lms_id, 
            A.address_line_1 as co_app_address_1,
            A.address_line_2 as co_app_address_2,
            A.city as co_app_address_city,
            A.state as co_app_address_state,
            A.pincode as co_app_pincode
        FROM leads_service_leads_service_loan_details ld 
        LEFT JOIN leads_service_leads_service_credit_profile CP
            ON LD.uuid = CP.loan_details_id AND CP.is_active = 1
            AND CP.profile_type = 'CO_APPLICANT'
        LEFT JOIN leads_service_leads_service_loan_user_profile_address_mapping LUPAM 
            ON LUPAM.LOAN_DETAILS_UUID = LD.UUID AND LUPAM.address_type = 'CURRENT_RESIDENCE' and LUPAM.user_profile_uuid = cp.user_profile_uuid
        LEFT JOIN LEADS_SERVICE_LEADS_SERVICE_ADDRESS A
            ON A.UUID = LUPAM.address_uuid)
    ,
    cte AS (
        SELECT 
            ld.lms_id AS lms_id, 
            A.address_line_1 as bussiness_address_1,
            A.address_line_2 as bussiness_address_2,
            A.city as Bussiness_address_city,
            A.state as Business_address_state,
            A.pincode as business_pincode,
            COALESCE(DATEDIFF(MONTH, A.residing_since, A.marked_at), CP.business_stability) AS business_stability
        FROM leads_service_leads_service_loan_details ld 
        LEFT JOIN leads_service_leads_service_credit_profile CP
            ON LD.uuid = CP.loan_details_id AND CP.is_active = 1
            AND CP.profile_type = 'APPLICANT'
        LEFT JOIN leads_service_leads_service_loan_user_profile_address_mapping LUPAM 
            ON LUPAM.LOAN_DETAILS_UUID = LD.UUID AND LUPAM.address_type = 'OFFICE' and LUPAM.user_profile_uuid = cp.user_profile_uuid
        LEFT JOIN LEADS_SERVICE_LEADS_SERVICE_ADDRESS A
            ON A.UUID = LUPAM.address_uuid
    )

    SELECT 
        ld.lms_id AS lms_id,
        o.ref_id AS order_id,
        coalesce(EU.NAME, JSON_EXTRACT_PATH_TEXT(JSON_SERIALIZE(COALESCE(CP.CREDIT_PROFILE_ADDITIONAL_DETAILS, JSON_PARSE('{}'))), 'displayName')) AS customer_name,
        EU.gender as gender,
        ld.assigned_to,
        CASE
            WHEN CP.IS_NTC = 1 THEN 'NTC'
            ELSE CAST(JSON_EXTRACT_PATH_TEXT(JSON_SERIALIZE(COALESCE(CR.credit_report_extra, JSON_PARSE('{}'))), 'score') AS TEXT) 
        END AS bureau,
        JSON_EXTRACT_PATH_TEXT(JSON_SERIALIZE(COALESCE(CP.CREDIT_PROFILE_ADDITIONAL_DETAILS, JSON_PARSE('{}'))), 'abbValue') AS abb_value,
        JSON_EXTRACT_PATH_TEXT(JSON_SERIALIZE(COALESCE(CP.CREDIT_PROFILE_ADDITIONAL_DETAILS, JSON_PARSE('{}'))), 'abcValue') AS abc_value,
        LD.loan_applicant_type AS constitution,
        JSON_EXTRACT_PATH_TEXT(JSON_SERIALIZE(COALESCE(CP.CREDIT_PROFILE_ADDITIONAL_DETAILS, JSON_PARSE('{}'))), 'scheme') AS scheme,
        JSON_EXTRACT_PATH_TEXT(JSON_SERIALIZE(COALESCE(CP.CREDIT_PROFILE_ADDITIONAL_DETAILS, JSON_PARSE('{}'))), 'segment') AS customer_segment,
        JSON_EXTRACT_PATH_TEXT(JSON_SERIALIZE(COALESCE(EU.PROFILE_INFO, JSON_PARSE('{}'))), 'occupation') AS occupation,
        EU.use_case AS Use_case_1,
        CP.monthly_income,
        CASE
            WHEN LD.co_applicant_added = 1 THEN 'Available'
            WHEN LD.co_applicant_added THEN 'NA'
            ELSE 'Unknown' -- Optional, in case there are other values for Co_applicant
        END AS co_applicant_availability,
        EU.dob,
        COALESCE(DATEDIFF(MONTH, A.residing_since, A.marked_at), CP.residence_stability) AS residence_stability,
        cte.business_stability, -- Reference to the CTE column
        vb.source AS source,
        ld.loan_reference_id AS loan_no,
        ld.loan_status AS loan_status,
        LP.NAME AS nbfc,
        A.house_ownership,
        A.address_line_1 as current_address1,
        A.address_line_2 as current_address2,
        A.CITY AS current_city,
        A.STATE AS current_state,
        A.pincode as curr_pincode,
        Permanent_address_1,
        Permanent_address_2,
        Peramanent_address_city,
        Peramanent_address_state,
        permanant_pincode,
        bussiness_address_1,
        bussiness_address_2,
        Bussiness_address_city,
        Business_address_state,
        co_app_address_1,
        co_app_address_2, 
        co_app_pincode,
        gurantor_address_1,
        gurantor_address_2,
        gurantor_pincode,
        OE.NAME AS oem_name,
        V.NAME AS vehicle_name,
        V.model as vehicle_model,
        V.vehicle_category_by_wheels,
        V.vehicle_type,
        V.VARIANT AS vehicle_variant,
        V.is_active as vehicle_active_status,
        vb.vehicle_number,
        vd.vin, -- <<<<<<<<<<<<<<<<<<<<<<<< CHANGE HERE
        DS.FULL_NAME AS dealership_name,
        LD.ON_ROAD_PRICE AS on_road_price,
        DATE(CR.created_at AT TIME ZONE 'UTC' AT TIME ZONE 'Asia/Kolkata') AS soft_check_date,
        DATE(LD.decision_date AT TIME ZONE 'UTC' AT TIME ZONE 'Asia/Kolkata') AS loan_approval_date,
        DATE(LD.LOAN_START_DATE AT TIME ZONE 'UTC' AT TIME ZONE 'Asia/Kolkata') AS disbursal_date,
        DATE(LD.DP_DATE AT TIME ZONE 'UTC' AT TIME ZONE 'Asia/Kolkata') AS downpayment_date,
        DATE(DD.Pre_EMI_DATE AT TIME ZONE 'UTC' AT TIME ZONE 'Asia/Kolkata') AS pre_emi_date,
        DATE(DD.EMI_Start_DATE AT TIME ZONE 'UTC' AT TIME ZONE 'Asia/Kolkata') AS emi_start_date,
        dd.down_payment AS downpayment,
        dd.pre_emi_amount AS pre_emi_amount,
        dd.emi_amount AS emi_amount,
        dd.utr_reference AS utr_reference,
        LD.LOAN_AMOUNT AS loan_amount,
        LD.DURATION_IN_MONTHS AS tenure,
        LD.INTEREST_RATE AS rate_of_interest,
        CT.NAME AS dealership_city_name,
        ST.name AS dealers_state_name
    FROM leads_service_leads_service_orders o
    INNER JOIN leads_service_leads_service_vehicle_bookings vb
        ON o.ref_id = vb.ref_id
    RIGHT JOIN leads_service_leads_service_loan_details ld
        ON ld.vehicle_booking_id = vb.uuid
    -- <<<<<<<<<<<<<<<<<<<<<<<< NEW JOIN ADDED HERE
    LEFT JOIN leads_service_leads_service_vehicle_details vd 
        ON vd.uuid = vb.vehicle_details_uuid
    -- <<<<<<<<<<<<<<<<<<<<<<<<
    LEFT JOIN leads_service_leads_service_credit_profile CP
        ON LD.uuid = CP.loan_details_id AND CP.is_active = 1 AND CP.profile_type = 'APPLICANT'
    LEFT JOIN leads_service_leads_service_credit_report CR
        ON CP.uuid = CR.credit_profile_id 
    LEFT JOIN leads_service_leads_service_external_user_profiles EU
        ON EU.uuid = CP.user_profile_uuid
    LEFT JOIN leads_service_leads_service_loan_provider LP
        ON LP.UUID = LD.LOAN_PROVIDER
    LEFT JOIN LEADS_SERVICE_LEADS_SERVICE_VEHICLES V
        ON V.UUID = VB.VEHICLE_UUID
    LEFT JOIN LEADS_SERVICE_LEADS_SERVICE_OEMS OE
        ON OE.UUID = V.OEM_UUID
    LEFT JOIN LEADS_SERVICE_LEADS_SERVICE_DEALERSHIPS DS
        ON DS.uuid = VB.DEALERSHIP_UUID
    LEFT JOIN leads_service_leads_service_disbursal_details dd
        ON ld.uuid = dd.loan_details_uuid
    LEFT JOIN LEADS_SERVICE_LEADS_SERVICE_CITIES CT
        ON CT.UUID = DS.CITY_UUID
    LEFT JOIN leads_service_leads_service_states ST
        ON CT.state_uuid = ST.uuid
    LEFT JOIN LEADS_SERVICE_LEADS_SERVICE_LOAN_USER_PROFILE_ADDRESS_MAPPING LUPAM
        ON LUPAM.LOAN_DETAILS_UUID = LD.UUID AND LUPAM.address_type = 'CURRENT_RESIDENCE' AND LUPAM.user_profile_uuid = cp.user_profile_uuid
    LEFT JOIN LEADS_SERVICE_LEADS_SERVICE_ADDRESS A
        ON A.UUID = LUPAM.address_uuid
    LEFT JOIN cte
        ON cte.lms_id = ld.lms_id
    left join cte1
        on cte1.lms_id = ld.lms_id
    left join cte2
        on cte2.lms_id = ld.lms_id
    left join cte3
        on cte3.lms_id = ld.lms_id
    where ld.loan_status in ('DISBURSED','APPROVED')
                """))

                # Convert result to list of Row objects
                rows = result.fetchall()
                # Get column names from the result metadata
                columns = result.keys()
                # Convert to DataFrame
                df = pd.DataFrame(rows, columns=columns)

                logger.info("✅ Query executed and data converted to DataFrame.")
                return df

        except Exception as e:
            logger.info(f"❌ An error occurred during the database connection or query: {e}")
            return pd.DataFrame()  # Return empty DataFrame on error
    df = connect_to_database(prod_uri)
    #logger.info(df)
    #df.columns
    def connect_to_database1(db_uri: str):
        try:
            # Add connect_args to change the SSL verification mode
            engine = create_engine(
                db_uri,
                connect_args={'sslmode': 'verify-ca'} # This still verifies the certificate is valid, but not the hostname
            )
            with engine.connect() as connection:
                logger.info("✅ Connection successful!")
                
                result = connection.execute(text("""
                select loan_no,chasisno
                from public.gsheet_mis__post_disbursement_documents___loan_forecloser_post_disbursement_docs
                where chasisno is not null and loan_no is not null;
                """))

                # Convert result to list of Row objects
                rows = result.fetchall()
                # Get column names from the result metadata
                columns = result.keys()
                # Convert to DataFrame
                df = pd.DataFrame(rows, columns=columns)

                logger.info("✅ Query executed and data converted to DataFrame.")
                return df

        except Exception as e:
            logger.info(f"❌ An error occurred during the database connection or query: {e}")
            return pd.DataFrame()  # Return empty DataFrame on error

    df2 = connect_to_database1(prod_uri)

    # Configuration - get from environment variables
    s3_uri = "s3://data-science-iot-reports/vin_loan_address/Address of vins.xlsx"
    
    # Get API key from environment variable
    API_KEY = os.environ.get('GOOGLE_MAPS_API_KEY')
    if not API_KEY:
        raise ValueError("GOOGLE_MAPS_API_KEY environment variable not set")
    
    gmaps = googlemaps.Client(key=API_KEY)

    # Define the final columns needed in the output file
    final_columns = ['vin', 'loan_no', 'current_address(lat,long)', 'permanent_address(lat,long)']

    logger.info("Starting the process to find, geocode, and update vehicle addresses...")

    # ==============================================================================
    # 2. LOAD EXISTING DATA & IDENTIFY ALL LOANS TO PROCESS
    # ==============================================================================

    # Get storage options from environment variables
    storage_options = {
        "key": os.environ.get('AWS_ACCESS_KEY_WRITE'),
        "secret": os.environ.get('AWS_SECRET_KEY_WRITE'),
    }

    # Validate AWS credentials
    if not storage_options["key"] or not storage_options["secret"]:
        raise ValueError("AWS credentials not set in environment variables")


    try:
        logger.info(f"Attempting to read existing address file from S3 ...")
        address_df = pd.read_excel(s3_uri, storage_options=storage_options)
        # Standardize column types for reliable merging
        address_df['loan_no'] = address_df['loan_no'].astype(str)
        address_df['vin'] = address_df['vin'].astype(str)
        logger.info("✅ Successfully read 'Address of vins.xlsx' from S3.")
        
        # Identify existing rows that are missing coordinates and need an update
        missing_coords_mask = address_df['current_address(lat,long)'].isnull() | address_df['permanent_address(lat,long)'].isnull()
        loans_to_update = address_df[missing_coords_mask][['loan_no', 'vin']]
        
        # Keep only the rows that are already complete
        clean_address_df = address_df[~missing_coords_mask]
        
    except Exception:
        logger.info("⚠️ Could not read 'Address of vins.xlsx'. Assuming no existing data. A new file will be created.")
        address_df = pd.DataFrame(columns=final_columns)
        loans_to_update = pd.DataFrame(columns=['loan_no', 'vin'])
        clean_address_df = pd.DataFrame(columns=final_columns)


    # Prepare df2 by renaming 'chasisno' to 'vin'
    df2_prepared = df2.rename(columns={'chasisno': 'vin'})
    df2_prepared['loan_no'] = df2_prepared['loan_no'].astype(str)
    df2_prepared['vin'] = df2_prepared['vin'].astype(str)

    # Identify completely new loans by finding what's in df2 but not in our existing address_df
    merged_df = df2_prepared.merge(address_df, on=['loan_no', 'vin'], how='left', indicator=True)
    new_loans_to_add = merged_df[merged_df['_merge'] == 'left_only'][['loan_no', 'vin']]

    # Combine the lists of loans to update and new loans to add
    loans_to_process = pd.concat([loans_to_update, new_loans_to_add], ignore_index=True).drop_duplicates()



    # ==============================================================================
    # 3. PROCESS AND GEOCODE ALL REQUIRED LOANS (if any)
    # ==============================================================================

    if not loans_to_process.empty:
        logger.info(f"\nFound {len(loans_to_process)} total loans to process (new and existing).")

        # Get the full address details for the loans from the main 'df'
        loans_with_addresses = loans_to_process.merge(df, on=['loan_no', 'vin'], how='left')

        # --- Combine Address Parts into Full Strings ---
        address_cols_to_fill = [
            'current_address1', 'current_city', 'current_state', 'curr_pincode',
            'permanent_address_1', 'peramanent_address_city', 'peramanent_address_state', 'permanant_pincode'
        ]
        for col in address_cols_to_fill:
            if col in loans_with_addresses.columns:
                loans_with_addresses[col] = loans_with_addresses[col].fillna('').astype(str)
            else:
                loans_with_addresses[col] = ''
                
        loans_with_addresses['current_full_address'] = loans_with_addresses['current_address1'] + ', ' + loans_with_addresses['current_city'] + ', ' + loans_with_addresses['current_state'] + ' ' + loans_with_addresses['curr_pincode']
        loans_with_addresses['permanent_full_address'] = loans_with_addresses['permanent_address_1'] + ', ' + loans_with_addresses['peramanent_address_city'] + ', ' + loans_with_addresses['peramanent_address_state'] + ' ' + loans_with_addresses['permanant_pincode']

        # --- Efficiently Geocode All Unique Addresses ---
        unique_addresses = pd.unique(loans_with_addresses[['current_full_address', 'permanent_full_address']].values.ravel('K'))
        geocoded_cache = {}
        logger.info("Geocoding addresses...")
        for address in unique_addresses:
            address = address.strip(', ').strip()
            if not address or address in geocoded_cache:
                continue
            try:
                geocode_result = gmaps.geocode(address)
                if geocode_result:
                    lat = geocode_result[0]['geometry']['location']['lat']
                    lng = geocode_result[0]['geometry']['location']['lng']
                    geocoded_cache[address] = (lat, lng)
                else:
                    geocoded_cache[address] = (np.nan, np.nan)
            except Exception as e:
                logger.info(f"--> Error geocoding '{address}': {e}")
                geocoded_cache[address] = (np.nan, np.nan)
        logger.info("✅ Geocoding complete.")

        # --- Map Geocoded Coordinates Back to the DataFrame ---
        loans_with_addresses['current_address(lat,long)'] = loans_with_addresses['current_full_address'].map(geocoded_cache)
        loans_with_addresses['permanent_address(lat,long)'] = loans_with_addresses['permanent_full_address'].map(geocoded_cache)

        # Prepare the newly processed rows for concatenation
        processed_rows = loans_with_addresses[final_columns]

        # --- Combine the clean old data with the newly processed data ---
        final_df = pd.concat([clean_address_df, processed_rows], ignore_index=True)
        logger.info("\n✅ All required rows have been processed and combined.")

    else:
        logger.info("\n✅ No new or incomplete loans found. The data is already up-to-date.")
        final_df = clean_address_df.copy()

    # ==============================================================================
    # 4. SAVE FINAL RESULT TO S3
    # ==============================================================================

    try:
        # Ensure the final DataFrame has columns in the correct order before saving
        final_df = final_df[final_columns]

        logger.info(f"\nAttempting to save the updated file back to S3 at: {s3_uri}")
        final_df.to_excel(s3_uri, index=False, storage_options=storage_options)
        logger.info(f"✅ Process finished. The complete and updated data has been saved to S3.")

        logger.info("\n--- Preview of the Final DataFrame ---")
        logger.info(final_df)
        logger.info("------------------------------------")

    except Exception as e:
        logger.info(f"❌ An error occurred while saving the file to S3: {e}")
    
    return

# ==============================================================================
# --- 1. USER INPUTS & CONFIGURATION ---
# ==============================================================================
# --- S3 File Paths ---
S3_ADDRESS_FILE_PATH = "s3://data-science-iot-reports/vin_loan_address/Address of vins.xlsx"
S3_BUCKET_NAME = "data-science-iot-reports"
# Path to store intermediate state files for incremental aggregation
S3_STATE_FILE_PATH = f"s3://{S3_BUCKET_NAME}/vehicle_locations/state_files"


# --- Analysis Configuration ---
HISTORICAL_DAYS_LOOKBACK = 60 # Used only for the first run or if state is lost
MIN_NIGHTS_PRESENT = 2

# --- AWS Configuration ---
ATHENA_DATABASE = "oem-iot-data"
ATHENA_S3_STAGING_DIR = f"s3://aws-athena-query-results-822787745626-ap-south-1"

# --- VENDOR CONFIGURATION MAP ---
## MODIFIED: Added back gearposition logic for Mahindra.
VENDOR_CONFIG = {
    'mahindra': {
        'table': '"oem-iot-data"."mahindra_parquet_vehicle_data"',
        'vin_col': 'vin', 'eventat_col': 'eventat', 'lat_col': 'latitude', 'lon_col': 'longitude',
        'soc_col': 'soc', 'dte_col': 'distancetoempty',
        'charging_logic': "state = 'CHARGING' OR {soc_col} > prev_soc OR {dte_col} > prev_dte",
        'required_cols': ['state', 'soc', 'distancetoempty', 'latitude', 'longitude', 'odometer', 'gearposition'],
        'min_duration': 10, 'min_points': 5,
        'odometer_col': 'odometer',
        'extra_where_clause': "AND gearposition = 'Neutral'" # Re-added for Mahindra-specific filtering
    },
    'euler': {
        'table': '"oem-iot-data"."euler_parquet_vehicle_data"',
        'vin_col': 'vin', 'eventat_col': 'eventat', 'location_col': 'location',
        'soc_col': 'batterysoc',
        'charging_logic': "batterycurrent > 0.1 OR {soc_col} > prev_soc",
        'required_cols': ['location', 'batterysoc', 'batterycurrent', 'odometer'],
        'min_duration': 10, 'min_points': 5,
        'odometer_col': 'odometer'
    },
    'piaggio': {
        'table': '"oem-iot-data"."piaggio_parquet_vehicle_data"',
        'vin_col': 'vin', 'eventat_col': 'eventat', 'lat_col': 'latitude', 'lon_col': 'longitude',
        'soc_col': 'soc', 'dte_col': 'distancetillempty',
        'charging_logic': "batterycharging = 1 OR {soc_col} > prev_soc OR {dte_col} > prev_dte",
        'required_cols': ['batterycharging', 'soc', 'latitude', 'longitude', 'distancetillempty', 'odometer'],
        'min_duration': 1, 'min_points': 1,
        'odometer_col': 'odometer'
    },
    'montra': {
        'table': '"oem-iot-data"."montra-gps-parquet-gps_data"',
        'vin_col': 'vin', 'eventat_col': 'eventat', 'lat_col': 'latitude', 'lon_col': 'longitude',
        'soc_col': 'soc',
        'charging_logic': "{soc_col} > prev_soc",
        'required_cols': ['soc', 'latitude', 'longitude', 'odometer'],
        'min_duration': 1, 'min_points': 2,
        'odometer_col': 'odometer'
    },
    'intellicar': {
        'table': '"oem-iot-data"."intellicar_location_parquet_location_data"',
        'vin_col': 'vin', 'eventat_col': 'eventat', 'lat_col': 'lat', 'lon_col': 'lng',
        'charging_logic': None
    },
}

# --- Dynamic Final Output Path ---
REPORT_PREFIX = f"s3://{S3_BUCKET_NAME}/vehicle_locations/daily_combined_reports/"
FINAL_OUTPUT_CSV_PATH = f"{REPORT_PREFIX}all_vendors_combined_locations.csv"
FINAL_OUTPUT_JSON_PATH = f"{REPORT_PREFIX}all_vendors_combined_locations.json"  # Changed to JSON


# ==============================================================================
# --- 2. SETUP & HELPER FUNCTIONS ---
# ==============================================================================
def convert_latlong_to_maps_link(lat_long_str):
    """
    Converts a 'lat,long' string to a clickable Google Maps link.
    Returns empty string if input is invalid.
    """
    if pd.isna(lat_long_str) or lat_long_str == '' or str(lat_long_str).strip() == '':
        return ''
    
    try:
        # Handle tuple format (from geocoding)
        if isinstance(lat_long_str, tuple) and len(lat_long_str) == 2:
            lat, lng = lat_long_str
        else:
            # Handle string format "lat,lng"
            lat_long_str = str(lat_long_str).strip()
            if ',' not in lat_long_str:
                return ''
            
            parts = lat_long_str.split(',')
            if len(parts) != 2:
                return ''
            
            lat = float(parts[0].strip())
            lng = float(parts[1].strip())
        
        # Create Google Maps link
        maps_link = f"https://www.google.com/maps?q={lat},{lng}"
        return maps_link
        
    except (ValueError, IndexError, AttributeError):
        return ''

def convert_dataframe_to_json_format(df):
    """
    Converts the DataFrame to a list of dictionaries with Google Maps links.
    """
    result_list = []
    
    for _, row in df.iterrows():
        record = {}
        
        # Convert each column to the appropriate format
        for col in df.columns:
            if 'location' in col.lower() and col != 'loan_no' and col != 'vin' and 'charging_location_visits_' not in col.lower():
                # Convert location columns to Google Maps links
                record[col] = convert_latlong_to_maps_link(row[col])
            elif col == 'current_address':
                row[col] = str(row[col]).strip("()")
                row[col] = row[col].replace(" ", "")
                record[col] = convert_latlong_to_maps_link(row[col])
            elif 'charging_location_visits_' in col.lower():
                # Keep visit counts as integers
                record[col] = int(row[col]) if pd.notna(row[col]) else '0'
            else:
                # Keep other fields as strings
                record[col] = str(row[col]) if pd.notna(row[col]) else ''
        
        result_list.append(record)
    
    return result_list

def initialize_boto3_sessions():
    """Initializes and returns AWS sessions."""
    try:
        company_read_session = boto3.Session(
            aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_READ'),
            aws_secret_access_key=os.environ.get('AWS_SECRET_KEY_READ'),
            region_name='ap-south-1'
        )

        personal_write_session = boto3.Session(
            aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_WRITE'),
            aws_secret_access_key=os.environ.get('AWS_SECRET_KEY_WRITE'),
            region_name='ap-south-1'
        )
        
        logger.info("✅ AWS sessions initialized successfully.")
        return company_read_session, personal_write_session
    except Exception as e:
        logger.error(f"❌ Error initializing AWS sessions: {e}")
        raise

def execute_athena_query(sql_query, boto3_session, vendor, query_type=""):
    """Executes an Athena query using awswrangler."""
    logger.info(f"🚀 Executing Athena query for {query_type} (Vendor: {vendor})...")
    try:
        return wr.athena.read_sql_query(
            sql=sql_query, database=ATHENA_DATABASE, s3_output=ATHENA_S3_STAGING_DIR,
            ctas_approach=False, boto3_session=boto3_session
        )
    except Exception as e:
        logger.info(f"❌ Query for {vendor} ({query_type}) failed: {str(e)}")
        logger.info("--- FAILED SQL QUERY ---")
        logger.info(sql_query)
        logger.info("------------------------")
        return pd.DataFrame()

def generate_partition_filter(lookback_days):
    """Generates an Athena SQL WHERE clause for date partitions."""
    today = datetime.now(timezone.utc)
    date_partitions = {}
    for i in range(lookback_days + 1):
        target_date = today - timedelta(days=i)
        year_month_key = (target_date.strftime('%Y'), target_date.strftime('%m'))
        if year_month_key not in date_partitions:
            date_partitions[year_month_key] = []
        date_partitions[year_month_key].append(target_date.strftime('%d'))

    clauses = []
    for (year, month), days in date_partitions.items():
        day_list_str = ", ".join([f"'{d}'" for d in days])
        clauses.append(f"(year = '{year}' AND month = '{month}' AND day IN ({day_list_str}))")
    return " OR ".join(clauses)

def pivot_locations_to_wide_format(df, type_prefix, include_visits=False):
    """Pivots a long-format location DataFrame to a wide format for top 3 locations."""
    if df.empty:
        return pd.DataFrame()
    df['location_str'] = df['latitude'].round(5).astype(str) + ', ' + df['longitude'].round(5).astype(str)
    df['location_num'] = (df.groupby('vin').cumcount() + 1)
    df_wide_loc = df.pivot_table(index='vin', columns='location_num', values='location_str', aggfunc='first').reset_index()
    df_wide_loc.columns = ['vin'] + [f'{type_prefix}_{i}' for i in df_wide_loc.columns[1:]]
    df_wide_loc.columns.name = None
    final_df = df_wide_loc
    final_cols = ['vin']
    if include_visits and 'visits' in df.columns:
        df_wide_visits = df.pivot_table(index='vin', columns='location_num', values='visits', aggfunc='first').reset_index()
        df_wide_visits.columns = ['vin'] + [f'{type_prefix}_visits_{i}' for i in df_wide_visits.columns[1:]]
        df_wide_visits.columns.name = None
        final_df = pd.merge(df_wide_loc, df_wide_visits, on='vin', how='outer')
    for i in range(1, 4):
        loc_col = f'{type_prefix}_{i}'
        final_cols.append(loc_col)
        if loc_col not in final_df.columns:
            final_df[loc_col] = ''
        if include_visits:
            visit_col = f'{type_prefix}_visits_{i}'
            final_cols.append(visit_col)
            if visit_col not in final_df.columns:
                final_df[visit_col] = 0
    fill_values = {col: 0 for col in final_df.columns if 'visits' in col}
    final_df.fillna(value=fill_values, inplace=True)
    final_df.fillna('', inplace=True)
    for col in final_df.columns:
        if 'visits' in col:
            final_df[col] = final_df[col].astype(int)
    return final_df[final_cols]

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great-circle distance between two points
    on the earth (specified in decimal degrees).
    Vectorized version for pandas series.
    Returns distance in meters.
    """
    # Radius of earth in kilometers. Use 6371 for kilometers, 6371000 for meters
    R = 6371000

    # Convert decimal degrees to radians
    lat1_rad, lon1_rad, lat2_rad, lon2_rad = map(np.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    distance = R * c
    return distance


# ==============================================================================
# --- 3. ADVANCED INCREMENTAL ANALYSIS FUNCTIONS ---
# ==============================================================================
def get_all_night_locations_incrementally(vendor, config, read_session, previous_state):
    logger.info(f"\n--- Starting INCREMENTAL Night Location Analysis for {vendor.upper()} ---")
    lookback_days = 1 if previous_state is not None else HISTORICAL_DAYS_LOOKBACK
    logger.info(f"INFO: Previous night state {'found' if previous_state is not None else 'NOT found'}. Querying last {lookback_days} days of data.")
    today = datetime.now(timezone.utc)
    partition_filter = generate_partition_filter(lookback_days)
    odometer_col = config.get('odometer_col')
    ## NEW: Get the extra filter which will be used for Mahindra's gear position check.
    extra_filter = config.get('extra_where_clause', '')

    if 'location_col' in config:
        loc_col = config['location_col']
        location_validation_str = f'"{loc_col}" IS NOT NULL AND cardinality(split("{loc_col}", \',\')) = 2 AND try_cast(split_part("{loc_col}", \',\', 1) as double) IS NOT NULL AND try_cast(split_part("{loc_col}", \',\', 2) as double) IS NOT NULL'
        lat_col_str, lon_col_str = f"CAST(split_part(\"{loc_col}\", ',', 1) AS DOUBLE)", f"CAST(split_part(\"{loc_col}\", ',', 2) AS DOUBLE)"
    else:
        lat_col, lon_col = config.get('lat_col', 'lat'), config.get('lon_col', 'lng')
        location_validation_str = f'try_cast("{lat_col}" as double) IS NOT NULL AND try_cast("{lon_col}" as double) IS NOT NULL'
        lat_col_str, lon_col_str = f"CAST(\"{lat_col}\" AS DOUBLE)", f"CAST(\"{lon_col}\" AS DOUBLE)"

    if odometer_col:
        logger.info(f"INFO: Applying stationary vehicle logic using '{odometer_col}' column.")
        if extra_filter:
            logger.info(f"INFO: Applying additional filter for {vendor}: {extra_filter}")
        ## MODIFIED: Added {extra_filter} to the WHERE clause to handle gear position.
        sql_query = f"""
        WITH OdometerLag AS (
            SELECT 
                "{config['vin_col']}" AS vin, "{config['eventat_col']}" AS eventat, 
                {lat_col_str} AS latitude, {lon_col_str} AS longitude,
                CAST("{odometer_col}" as DOUBLE) as odometer_val,
                LAG(CAST("{odometer_col}" as DOUBLE), 1, CAST("{odometer_col}" as DOUBLE)) OVER (PARTITION BY "{config['vin_col']}" ORDER BY "{config['eventat_col']}") as prev_odometer
            FROM {config['table']}
            WHERE ({partition_filter})
              AND hour(from_unixtime(CAST("{config['eventat_col']}" / 1000 AS BIGINT))) <= 6
              AND {location_validation_str} AND {lat_col_str} BETWEEN 8.0 AND 37.0 AND {lon_col_str} BETWEEN 68.0 AND 98.0
              AND "{odometer_col}" IS NOT NULL
              {extra_filter}
        ),
        StationaryPoints AS (
            SELECT vin, eventat, latitude, longitude FROM OdometerLag
            WHERE ABS(odometer_val - prev_odometer) <= 0.1
        ),
        nightly_points_ranked AS (
            SELECT *, ROW_NUMBER() OVER (PARTITION BY vin, CAST(from_unixtime(CAST(eventat / 1000 AS BIGINT)) AS DATE) ORDER BY eventat DESC) AS rn
            FROM StationaryPoints
        )
        SELECT vin, eventat, latitude, longitude FROM nightly_points_ranked WHERE rn = 1 
        """
    else:
        sql_query = f"""
        WITH nightly_points_ranked AS (
            SELECT "{config['vin_col']}" AS vin, "{config['eventat_col']}" AS eventat, {lat_col_str} AS latitude, {lon_col_str} AS longitude,
                   ROW_NUMBER() OVER (PARTITION BY "{config['vin_col']}", CAST(from_unixtime(CAST("{config['eventat_col']}" / 1000 AS BIGINT)) AS DATE) ORDER BY "{config['eventat_col']}" DESC) AS rn
            FROM {config['table']}
            WHERE ({partition_filter})
              AND hour(from_unixtime(CAST("{config['eventat_col']}" / 1000 AS BIGINT))) <= 6
              AND {location_validation_str} AND {lat_col_str} BETWEEN 8.0 AND 37.0 AND {lon_col_str} BETWEEN 68.0 AND 98.0
        ) SELECT vin, eventat, latitude, longitude FROM nightly_points_ranked WHERE rn = 1 
        """
    df_new_data = execute_athena_query(sql_query, read_session, vendor, "Incremental Night Data")
    if previous_state is not None:
        df_night_data = pd.concat([previous_state, df_new_data], ignore_index=True).drop_duplicates(subset=['vin', 'eventat'])
    else:
        df_night_data = df_new_data
    if df_night_data.empty: return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    df_night_data['event_date'] = pd.to_datetime(df_night_data['eventat'], unit='ms', utc=True).dt.date
    cutoff_date = (today - timedelta(days=HISTORICAL_DAYS_LOOKBACK)).date()
    updated_state_df = df_night_data[df_night_data['event_date'] >= cutoff_date].copy()
    yesterday_date = (today - timedelta(days=1)).date()
    df_last_night_points = updated_state_df[updated_state_df['event_date'] == yesterday_date].copy()
    last_night_df = pd.DataFrame()
    if not df_last_night_points.empty:
        df_last_night_points['last_night_location'] = df_last_night_points['latitude'].round(5).astype(str) + ', ' + df_last_night_points['longitude'].round(5).astype(str)
        last_night_df = df_last_night_points[['vin', 'last_night_location']]
    if updated_state_df.empty: return last_night_df, pd.DataFrame(), updated_state_df[['vin', 'eventat', 'latitude', 'longitude']]
    lat_precision = round(-np.log10(100 / 111100.0))
    lon_precision = round(-np.log10(100 / (111100.0 * np.cos(np.radians(updated_state_df['latitude'].mean())))))
    updated_state_df['grid_cell'] = updated_state_df['latitude'].round(lat_precision).astype(str) + ',' + updated_state_df['longitude'].round(lon_precision).astype(str)
    counts = updated_state_df.groupby(['vin', 'grid_cell'])['event_date'].nunique().reset_index(name='night_count')
    valid_cells = counts[counts['night_count'] >= MIN_NIGHTS_PRESENT]
    top_3_historical_df = pd.DataFrame()
    if not valid_cells.empty:
        valid_points = updated_state_df.merge(valid_cells[['vin', 'grid_cell']], on=['vin', 'grid_cell'])
        centroids = valid_points.groupby(['vin', 'grid_cell']).agg(latitude=('latitude', 'mean'), longitude=('longitude', 'mean'), night_count=('event_date', 'nunique')).reset_index()
        top_3_df = centroids.sort_values(['vin', 'night_count'], ascending=[True, False]).groupby('vin').head(3)
        top_3_historical_df = pivot_locations_to_wide_format(top_3_df, 'night_location')
    return last_night_df, top_3_historical_df, updated_state_df[['vin', 'eventat', 'latitude', 'longitude']]

def get_top_charging_locations_incrementally(vendor, config, read_session, previous_state):
    logger.info(f"\n--- Starting INCREMENTAL Charging Location Analysis for {vendor.upper()} ---")
    if not config.get('charging_logic'): return pd.DataFrame(), pd.DataFrame()
    # For Euler, override to use only yesterday's partition
    if vendor.lower() == 'euler':
        today = datetime.now(timezone.utc)
        yesterday = today - timedelta(days=1)
        year = yesterday.strftime('%Y')
        month = yesterday.strftime('%m')
        day = yesterday.strftime('%d')
        partition_filter = f"(year = '{year}' AND month = '{month}' AND day = '{day}')"
    else:
        lookback_days = 1 if previous_state is not None else HISTORICAL_DAYS_LOOKBACK
        logger.info(f"INFO: Previous charging state {'found' if previous_state is not None else 'NOT found'}. Querying last {lookback_days} days of data.")
        today = datetime.now(timezone.utc)
        partition_filter = generate_partition_filter(lookback_days)
    
    min_duration, min_points = config.get('min_duration', 1), config.get('min_points', 1)
    format_mapping = {'soc_col': config.get('soc_col'), 'dte_col': config.get('dte_col')}
    all_required_cols = set(config.get('required_cols', []) + [config['vin_col'], config['eventat_col']])
    odometer_col = config.get('odometer_col')
    extra_filter = config.get('extra_where_clause', '') # Get extra filter

    if 'location_col' in config:
        loc_col = config['location_col']
        all_required_cols.add(loc_col)
        location_validation_str = f'"{loc_col}" IS NOT NULL AND cardinality(split("{loc_col}", \',\')) = 2 AND try_cast(split_part("{loc_col}", \',\', 1) as double) BETWEEN 8.0 AND 37.0 AND try_cast(split_part("{loc_col}", \',\', 2) as double) BETWEEN 68.0 AND 98.0'
        location_parsing_str = f"CAST(split_part(\"{loc_col}\", ',', 1) AS DOUBLE) AS latitude, CAST(split_part(\"{loc_col}\", ',', 2) AS DOUBLE) AS longitude"
    else:
        lat_col, lon_col = config['lat_col'], config['lon_col']
        all_required_cols.add(lat_col); all_required_cols.add(lon_col)
        location_validation_str = f'try_cast("{lat_col}" as double) BETWEEN 8.0 AND 37.0 AND try_cast("{lon_col}" as double) BETWEEN 68.0 AND 98.0'
        location_parsing_str = f"CAST(\"{lat_col}\" AS DOUBLE) AS latitude, CAST(\"{lon_col}\" AS DOUBLE) AS longitude"

    charging_logic_formatted = config['charging_logic'].format(**format_mapping)
    if odometer_col:
        logger.info(f"INFO: Applying stationary vehicle logic using '{odometer_col}' column to charging.")
        final_charging_logic = f"({charging_logic_formatted}) AND ABS(CAST(\"{odometer_col}\" AS DOUBLE) - prev_odometer) <= 0.1"
        odometer_lag_str = f""", LAG(CAST("{odometer_col}" AS DOUBLE), 1, CAST("{odometer_col}" AS DOUBLE)) OVER (PARTITION BY "{config['vin_col']}" ORDER BY "{config['eventat_col']}") AS prev_odometer"""
        odometer_select_str = f", \"{odometer_col}\""
        location_validation_str += f" AND \"{odometer_col}\" IS NOT NULL"
    else:
        final_charging_logic = charging_logic_formatted
        odometer_lag_str = ""
        odometer_select_str = ""
    
    # The extra_filter is already included in the FilteredData CTE query below.
    if extra_filter:
        logger.info(f"INFO: Applying additional filter for {vendor}: {extra_filter}")

    select_cols_str = ", ".join(f'"{col}"' for col in all_required_cols)
    prev_dte_creation_str = f', LAG(CAST("{config["dte_col"]}" AS DOUBLE), 1, 0) OVER (PARTITION BY "{config["vin_col"]}" ORDER BY "{config["eventat_col"]}") AS prev_dte' if config.get('dte_col') else ""
    dte_selection_str = f', CAST("{config["dte_col"]}" AS DOUBLE) AS {config["dte_col"]}' if config.get('dte_col') else ""
    other_logic_cols = [c for c in config.get('required_cols', []) if c not in [config.get('soc_col'), config.get('dte_col'), config.get('location_col'), config.get('lat_col'), config.get('lon_col'), config.get('odometer_col')]]
    other_required_cols_str = ", ".join(f'"{c}"' for c in other_logic_cols) if other_logic_cols else ""

    sql_query = f"""
    WITH FilteredData AS (SELECT {select_cols_str} FROM {config['table']} WHERE ({partition_filter}) AND {location_validation_str} {extra_filter})
    , LaggedData AS (
        SELECT "{config['vin_col']}" AS vin, "{config['eventat_col']}" AS eventat, {location_parsing_str}, 
               CAST("{config.get('soc_col', 'NULL')}" AS DOUBLE) AS {config.get('soc_col')}, 
               LAG(CAST("{config.get('soc_col', 'NULL')}" AS DOUBLE), 1, 0) OVER (PARTITION BY "{config['vin_col']}" ORDER BY "{config['eventat_col']}") AS prev_soc 
               {prev_dte_creation_str} {dte_selection_str} {',' if other_required_cols_str else ''} {other_required_cols_str}
               {odometer_select_str} {odometer_lag_str}
        FROM FilteredData)
    , ChargingEvents AS (SELECT *, CASE WHEN {final_charging_logic} THEN 1 ELSE 0 END AS is_charging_signal FROM LaggedData)
    , SessionMarkers AS (SELECT *, CASE WHEN is_charging_signal = 1 AND LAG(is_charging_signal, 1, 0) OVER (PARTITION BY vin ORDER BY eventat) = 0 THEN 1 ELSE 0 END AS is_new_session_marker FROM ChargingEvents)
    , SessionIdentifier AS (SELECT *, SUM(is_new_session_marker) OVER (PARTITION BY vin ORDER BY eventat) AS session_id FROM SessionMarkers WHERE is_charging_signal = 1)
    , SessionAnalysis AS (
        SELECT vin, session_id, MIN(eventat) as session_start_time, AVG(latitude) AS latitude, AVG(longitude) AS longitude, COUNT(*) AS point_count,
               CAST(DATE_DIFF('minute', from_unixtime(MIN(eventat) / 1000), from_unixtime(MAX(eventat) / 1000)) AS DOUBLE) AS duration_minutes
        FROM SessionIdentifier GROUP BY vin, session_id)
    SELECT vin, session_start_time, latitude, longitude FROM SessionAnalysis 
    WHERE duration_minutes >= {min_duration} AND point_count >= {min_points}
    """
    df_new_data = execute_athena_query(sql_query, read_session, vendor, "Incremental Charging Data")
    if previous_state is not None:
        df_charging = pd.concat([previous_state, df_new_data], ignore_index=True).drop_duplicates(subset=['vin', 'session_start_time'])
    else:
        df_charging = df_new_data
    if df_charging.empty: return pd.DataFrame(), pd.DataFrame()
    cutoff_ts = (today - timedelta(days=HISTORICAL_DAYS_LOOKBACK)).timestamp() * 1000
    updated_state_df = df_charging[df_charging['session_start_time'] >= cutoff_ts].copy()
    if updated_state_df.empty: return pd.DataFrame(), updated_state_df
    coords = updated_state_df[['latitude', 'longitude']].to_numpy()
    db = DBSCAN(eps=(0.1 / 6371.0088), min_samples=1, metric='haversine', algorithm='ball_tree')
    updated_state_df['cluster'] = db.fit_predict(np.radians(coords))
    consolidated_df = updated_state_df.groupby(['vin', 'cluster']).agg(latitude=('latitude', 'mean'), longitude=('longitude', 'mean'), visits=('session_start_time', 'count')).reset_index()
    top_3_df = consolidated_df.sort_values(['vin', 'visits'], ascending=[True, False]).groupby('vin').head(3)
    new_state_to_save = updated_state_df[['vin', 'session_start_time', 'latitude', 'longitude']]
    return pivot_locations_to_wide_format(top_3_df, 'charging_location', include_visits=True), new_state_to_save


# ==============================================================================
# --- 4. PARALLEL PROCESSING WORKER FUNCTION ---
# ==============================================================================
def process_vendor(vendor, config, read_session, write_session):
    logger.info(f"🚀 Starting processing for VENDOR: {vendor.upper()}")
    night_state_path = f"{S3_STATE_FILE_PATH}/{vendor}_night_state.parquet"
    charging_state_path = f"{S3_STATE_FILE_PATH}/{vendor}_charging_state.parquet"
    try:
        previous_night_state = wr.s3.read_parquet(path=night_state_path, boto3_session=write_session)
    except wr.exceptions.NoFilesFound:
        previous_night_state = None
    try:
        previous_charging_state = wr.s3.read_parquet(path=charging_state_path, boto3_session=write_session)
    except wr.exceptions.NoFilesFound:
        previous_charging_state = None
    df_last_night, df_night_wide, new_night_state = get_all_night_locations_incrementally(vendor, config, read_session, previous_night_state)
    df_charging_wide, new_charging_state = get_top_charging_locations_incrementally(vendor, config, read_session, previous_charging_state)
    if not new_night_state.empty:
        wr.s3.to_parquet(df=new_night_state, path=night_state_path, boto3_session=write_session)
    if not new_charging_state.empty:
        wr.s3.to_parquet(df=new_charging_state, path=charging_state_path, boto3_session=write_session)
    if df_last_night.empty and df_night_wide.empty and df_charging_wide.empty:
        return None
    all_vins = pd.concat([df['vin'] for df in [df_last_night, df_night_wide, df_charging_wide] if not df.empty]).unique()
    if len(all_vins) == 0: return None
    vendor_df = pd.DataFrame(all_vins, columns=['vin'])
    if not df_last_night.empty: vendor_df = pd.merge(vendor_df, df_last_night, on='vin', how='left')
    if not df_night_wide.empty: vendor_df = pd.merge(vendor_df, df_night_wide, on='vin', how='left')
    if not df_charging_wide.empty: vendor_df = pd.merge(vendor_df, df_charging_wide, on='vin', how='left')
    logger.info(f"✅ Finished processing for VENDOR: {vendor.upper()}")
    return vendor_df


# ==============================================================================
# --- 5. MAIN EXECUTION BLOCK ---
# ==============================================================================
#if __name__ == "__main__":
def get_changes_iot():
    vin_loan_address()
    try:
        read_session, write_session = initialize_boto3_sessions()
        
        logger.info("\n" + "="*80 + "\nLOADING MASTER VEHICLE LIST FROM ADDRESS FILE\n" + "="*80)
        try:
            logger.info(f"🔄 Loading address data from: {S3_ADDRESS_FILE_PATH}")
            df_address = wr.s3.read_excel(path=S3_ADDRESS_FILE_PATH, boto3_session=write_session)
            required_address_cols = ['vin', 'loan_no', 'current_address(lat,long)']
            if not all(col in df_address.columns for col in required_address_cols):
                raise ValueError(f"S3 Excel file missing required columns: {required_address_cols}")
            
            df_address.drop_duplicates(subset=['vin'], keep='first', inplace=True)
            base_df = df_address[required_address_cols].copy()
            base_df.rename(columns={'current_address(lat,long)': 'current_address'}, inplace=True)
            logger.info(f"✅ Master list created with {len(base_df)} unique vehicles.")

        except Exception as e:
            logger.info(f"❌ FATAL: Could not load the master address file from {S3_ADDRESS_FILE_PATH}.")
            logger.info(f"   Error: {e}")
            sys.exit()

        all_vendors_data = []
        logger.info("\n" + "="*80 + "\nSTARTING PARALLEL PROCESSING FOR ALL VENDORS\n" + "="*80)
        with ThreadPoolExecutor(max_workers=len(VENDOR_CONFIG)) as executor:
            future_to_vendor = {executor.submit(process_vendor, vendor, config, read_session, write_session): vendor for vendor, config in VENDOR_CONFIG.items()}
            for future in as_completed(future_to_vendor):
                vendor = future_to_vendor[future]
                try:
                    result_df = future.result()
                    if result_df is not None: all_vendors_data.append(result_df)
                except Exception as exc:
                    logger.info(f"❌ Vendor {vendor} generated an exception: {exc}")
                    traceback.print_exc()

        if not all_vendors_data:
            logger.info("\n⚠️ No IoT data was processed for any vendor. The final report will only contain address data.")
            iot_data_df = pd.DataFrame(columns=['vin'])
        else:
            iot_data_df = pd.concat(all_vendors_data, ignore_index=True)
            iot_data_df.drop_duplicates(subset=['vin'], keep='first', inplace=True)

        logger.info("\n" + "="*80 + "\nCOMBINING AND PREPARING FULL DAILY REPORT\n" + "="*80)
        full_daily_df = pd.merge(base_df, iot_data_df, on='vin', how='left')

        final_column_order = [
            'loan_no', 'vin', 'last_night_location', 
            'night_location_1', 'night_location_2', 'night_location_3', 
            'current_address', 
            'charging_location_1', 'charging_location_visits_1',
            'charging_location_2', 'charging_location_visits_2',
            'charging_location_3', 'charging_location_visits_3'
        ]
        for col in final_column_order:
            if col not in full_daily_df.columns: full_daily_df[col] = ''
        
        full_daily_df = full_daily_df[final_column_order]
        for col in full_daily_df.columns:
            if 'visits' in col:
                full_daily_df[col] = full_daily_df[col].fillna(0).astype(int)
            else:
                full_daily_df[col] = full_daily_df[col].fillna('')

        full_daily_df.sort_values(by=['last_night_location', 'loan_no', 'vin'],ascending=False, inplace=True)
        logger.info(f"\n✅ Assembled today's full report for {len(full_daily_df)} total vehicles from the master list.")

        logger.info("\n" + "="*80 + "\nCOMPARING WITH PREVIOUS DAY TO COUNT UPDATES\n" + "="*80)
        try:
            previous_report_path = f"{REPORT_PREFIX}all_vendors_combined_locations.csv"
            logger.info(f"🔄 Reading previous report from: {previous_report_path}")
            df_previous = wr.s3.read_csv(path=previous_report_path, boto3_session=write_session)
            
            today_df_prepared = full_daily_df.copy()
            prev_df_prepared = df_previous.copy()
            all_cols = set(today_df_prepared.columns) | set(prev_df_prepared.columns)
            
            for col in all_cols:
                if col not in today_df_prepared: today_df_prepared[col] = ''
                if col not in prev_df_prepared: prev_df_prepared[col] = ''
                if 'visits' in col:
                    today_df_prepared[col] = pd.to_numeric(today_df_prepared[col], errors='coerce').fillna(0).astype(int)
                    prev_df_prepared[col] = pd.to_numeric(prev_df_prepared[col], errors='coerce').fillna(0).astype(int)
                else:
                    today_df_prepared[col] = today_df_prepared[col].astype(str).fillna('')
                    prev_df_prepared[col] = prev_df_prepared[col].astype(str).fillna('')

            # +++ This is the NEW code block to use +++
            merged_df = pd.merge(today_df_prepared, prev_df_prepared, on='vin', how='outer', suffixes=('_today', '_prev'))

            # Fill NaNs created by the outer merge to handle new/removed vehicles correctly
            for col in merged_df.columns:
                if 'visits' in col:
                    merged_df[col] = pd.to_numeric(merged_df[col], errors='coerce').fillna(0).astype(int)
                elif 'location' in col:
                    merged_df[col] = merged_df[col].astype(str).fillna('')

            today_vins = set(today_df_prepared['vin'])
            prev_vins = set(prev_df_prepared['vin'])
            new_vehicle_count = len(today_vins - prev_vins)

            DISTANCE_THRESHOLD_METERS = 100

            def get_location_update_mask(df, col_name):
                """Helper to calculate distance-based updates for a location column."""
                # 1. Identify rows with a change in presence (e.g., '' -> 'lat,lon' or vice-versa)
                presence_changed = (df[f'{col_name}_today'] != '') != (df[f'{col_name}_prev'] != '')

                # 2. For rows where BOTH locations exist, calculate distance
                both_exist = (df[f'{col_name}_today'] != '') & (df[f'{col_name}_prev'] != '')
                
                # Initialize distance mask
                distance_exceeded = pd.Series(False, index=df.index)

                if both_exist.any():
                    # Parse lat/lon for today's data
                    today_coords = df.loc[both_exist, f'{col_name}_today'].str.split(',', expand=True)
                    lat_today = pd.to_numeric(today_coords[0], errors='coerce')
                    lon_today = pd.to_numeric(today_coords[1], errors='coerce')
                    
                    # Parse lat/lon for previous data
                    prev_coords = df.loc[both_exist, f'{col_name}_prev'].str.split(',', expand=True)
                    lat_prev = pd.to_numeric(prev_coords[0], errors='coerce')
                    lon_prev = pd.to_numeric(prev_coords[1], errors='coerce')

                    # Calculate haversine distance only for valid coordinate pairs
                    valid_coords = lat_today.notna() & lon_today.notna() & lat_prev.notna() & lon_prev.notna()
                    if valid_coords.any():
                        dist = haversine_distance(
                            lat_today[valid_coords], lon_today[valid_coords],
                            lat_prev[valid_coords], lon_prev[valid_coords]
                        )
                        # Update the distance mask where distance > threshold
                        distance_exceeded.loc[both_exist & valid_coords] = dist > DISTANCE_THRESHOLD_METERS

                # Final mask: An update is a change in presence OR distance exceeded threshold
                return distance_exceeded

            # Calculate updates using the new distance-based logic
            updated_last_night_count = get_location_update_mask(merged_df, 'last_night_location').sum()

            night_cols = ['night_location_1', 'night_location_2', 'night_location_3']
            night_mask = pd.Series(False, index=merged_df.index)
            for col in night_cols:
                night_mask |= get_location_update_mask(merged_df, col)
            updated_night_count = night_mask.sum()

            charging_cols = ['charging_location_1', 'charging_location_2', 'charging_location_3']
            charging_mask = pd.Series(False, index=merged_df.index)
            for col in charging_cols:
                charging_mask |= get_location_update_mask(merged_df, col)
            updated_charging_count = charging_mask.sum()

            # This logic for newly charged remains correct
            newly_charged_mask = (merged_df['charging_location_1_today'] != '') & (merged_df['charging_location_1_prev'] == '')
            newly_charged_count = newly_charged_mask.sum()
            
            logger.info("✅ Comparison complete. Update counts:")
            logger.info(f"   - New Vehicles in report (vs. yesterday): {new_vehicle_count}")
            logger.info(f"   - Updated Last Night Locations: {updated_last_night_count}")
            logger.info(f"   - Vehicles with updated Historical Night Locations: {updated_night_count}")
            logger.info(f"   - New vehicles with first-time Charging Locations: {newly_charged_count}")
            logger.info(f"   - Vehicles with any change in top 3 Charging Locations: {updated_charging_count}")
        except wr.exceptions.NoFilesFound:
            logger.info("⚠️ Previous report not found. Cannot count updates. This is expected on the first run.")
        except Exception as e:
            logger.info(f"❌ Error during comparison: {e}. Unable to count updates.")

        output_df = full_daily_df
        if output_df.empty:
            logger.info("\n✅ No vehicle information to report today.")
        else:
            logger.info(f"\nINFO: Final output contains {len(output_df)} total rows.")

            # Convert DataFrame to JSON format with Google Maps links
            logger.info("🔄 Converting location data to Google Maps links...")
            json_output = convert_dataframe_to_json_format(output_df)

            # Define the filename
            filename = "night_locations.json"

            # Use 'with' statement to open the file
            with open(filename, 'w') as file:
                # Use json.dump() to write the dictionary to the file
                json.dump(json_output, file, indent=4)

            logger.info(f"JSON data has been saved to '{filename}'")

            logger.info(f"💾 Writing final JSON report to: {FINAL_OUTPUT_JSON_PATH}")
            
            wr.s3.upload(local_file=filename, path=FINAL_OUTPUT_JSON_PATH, boto3_session=write_session)

            logger.info(f"💾 Writing final report to: {FINAL_OUTPUT_CSV_PATH}")
            wr.s3.to_csv(df=output_df, path=FINAL_OUTPUT_CSV_PATH, index=False, boto3_session=write_session)
        
        logger.info(f"\n✅✨ ADVANCED OPTIMIZED analysis complete!")
        # return {"status": "success", "timestamp": datetime.now().isoformat()}
        return json_output
        
    except Exception as script_error:
        logger.info(f"\n❌ A major script execution error occurred: {script_error}")
        traceback.print_exc()
    

def main():
    """Main entry point for GitHub Actions"""
    try:
        logger.info("🚀 Starting daily vehicle analysis...")
        result = get_changes_iot()
        logger.info("✅ Analysis completed successfully!")
        return result
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()
