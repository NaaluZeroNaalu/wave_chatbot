import io
import streamlit as st
import requests
import json
import urllib.parse
import urllib3
import certifi
import pandas as pd  
from bs4 import BeautifulSoup
from datetime import datetime
import re
import logging
import os
from dotenv import load_dotenv
from io import BytesIO
import base64
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from typing import Tuple, Dict, Any
from Veridiaa_new import *
from EWS_Final import *
from Eden_Final import *
from club_new import *
from Eligo_new import *


# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# WatsonX configuration
WATSONX_API_URL = os.getenv("WATSONX_API_URL")
MODEL_ID = os.getenv("MODEL_ID")
PROJECT_ID = os.getenv("PROJECT_ID")
API_KEY = os.getenv("API_KEY")

# Check environment variables
if not all([API_KEY, WATSONX_API_URL, MODEL_ID, PROJECT_ID]):
    st.error("❌ Missing environment variables. Please set API_KEY, WATSONX_API_URL, MODEL_ID, and PROJECT_ID in your .env file.")
    st.markdown("**Setup Instructions**:\n1. Create a `.env` file with the following:\n```\nAPI_KEY=your_api_key\nWATSONX_API_URL=your_url\nMODEL_ID=your_model_id\nPROJECT_ID=your_project_id\n```\n2. Restart the application.")
    logger.error("Missing one or more required environment variables")
    st.stop()

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# API Endpoints
LOGIN_URL = "https://dms.asite.com/apilogin/"
SEARCH_URL = "https://adoddleak.asite.com/commonapi/formsearchapi/search"
IAM_TOKEN_URL = "https://iam.cloud.ibm.com/identity/token"

# Function to generate access token
def get_access_token(API_KEY):
    headers = {"Content-Type": "application/x-www-form-urlencoded", "Accept": "application/json"}
    data = {"grant_type": "urn:ibm:params:oauth:grant-type:apikey", "apikey": API_KEY}
    try:
        response = requests.post(IAM_TOKEN_URL, headers=headers, data=data, verify=certifi.where(), timeout=50)
        if response.status_code == 200:
            token_info = response.json()
            logger.info("Access token generated successfully")
            return token_info['access_token']
        else:
            logger.error(f"Failed to get access token: {response.status_code} - {response.text}")
            st.error(f"❌ Failed to get access token: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        logger.error(f"Exception getting access token: {str(e)}")
        st.error(f"❌ Error getting access token: {str(e)}")
        return None

# Login Function
def login_to_asite(email, password):
    headers = {"Accept": "application/json", "Content-Type": "application/x-www-form-urlencoded"}
    payload = {"emailId": email, "password": password}
    response = requests.post(LOGIN_URL, headers=headers, data=payload, verify=certifi.where(), timeout=50)
    if response.status_code == 200:
        try:
            session_id = response.json().get("UserProfile", {}).get("Sessionid")
            logger.info(f"Login successful, Session ID: {session_id}")
            return session_id
        except json.JSONDecodeError:
            logger.error("JSONDecodeError during login")
            st.error("❌ Failed to parse login response")
            return None
    logger.error(f"Login failed: {response.status_code}")
    st.error(f"❌ Login failed: {response.status_code}")
    return None

# Fetch Data Function
def fetch_project_data(session_id, project_name, form_name, record_limit=1000):
    headers = {"Accept": "application/json", "Content-Type": "application/x-www-form-urlencoded", "Cookie": f"ASessionID={session_id}"}
    all_data = []
    start_record = 1
    total_records = None

    with st.spinner("Fetching data from Asite..."):
        while True:
            search_criteria = {"criteria": [{"field": "ProjectName", "operator": 1, "values": [project_name]}, {"field": "FormName", "operator": 1, "values": [form_name]}], "recordStart": start_record, "recordLimit": record_limit}
            search_criteria_str = json.dumps(search_criteria)
            encoded_payload = f"searchCriteria={urllib.parse.quote(search_criteria_str)}"
            response = requests.post(SEARCH_URL, headers=headers, data=encoded_payload, verify=certifi.where(), timeout=50)

            try:
                response_json = response.json()
                if total_records is None:
                    total_records = response_json.get("responseHeader", {}).get("results-total", 0)
                all_data.extend(response_json.get("FormList", {}).get("Form", []))
                st.info(f"🔄 Fetched {len(all_data)} / {total_records} records")
                if start_record + record_limit - 1 >= total_records:
                    break
                start_record += record_limit
            except Exception as e:
                # logger.error(f"Error fetching data: {str(e)}")
                # st.error(f"❌ Error fetching data: {str(e)}")
                return "Asite Error"

    return {"responseHeader": {"results": len(all_data), "total_results": total_records}}, all_data, encoded_payload

# Process JSON Data
def process_json_data(json_data):
    data = []
    for item in json_data:
        form_details = item.get('FormDetails', {})
        created_date = form_details.get('FormCreationDate', None)
        expected_close_date = form_details.get('UpdateDate', None)
        form_status = form_details.get('FormStatus', None)
        
        discipline = None
        description = None
        custom_fields = form_details.get('CustomFields', {}).get('CustomField', [])
        for field in custom_fields:
            if field.get('FieldName') == 'CFID_DD_DISC':
                discipline = field.get('FieldValue', None)
            elif field.get('FieldName') == 'CFID_RTA_DES':
                description = BeautifulSoup(field.get('FieldValue', None) or '', "html.parser").get_text()

        days_diff = None
        if created_date and expected_close_date:
            try:
                created_date_obj = datetime.strptime(created_date.split('#')[0], "%d-%b-%Y")
                expected_close_date_obj = datetime.strptime(expected_close_date.split('#')[0], "%d-%b-%Y")
                days_diff = (expected_close_date_obj - created_date_obj).days
            except Exception as e:
                logger.error(f"Error calculating days difference: {str(e)}")
                days_diff = None

        data.append([days_diff, created_date, expected_close_date, description, form_status, discipline])

    df = pd.DataFrame(data, columns=['Days', 'Created Date (WET)', 'Expected Close Date (WET)', 'Description', 'Status', 'Discipline'])
    df['Created Date (WET)'] = pd.to_datetime(df['Created Date (WET)'].str.split('#').str[0], format="%d-%b-%Y", errors='coerce')
    df['Expected Close Date (WET)'] = pd.to_datetime(df['Expected Close Date (WET)'].str.split('#').str[0], format="%d-%b-%Y", errors='coerce')
    logger.debug(f"DataFrame columns after processing: {df.columns.tolist()}")
    if df.empty:
        logger.warning("DataFrame is empty after processing")
        st.warning("⚠️ No data processed. Check the API response.")
    return df

# Clean and Parse JSON
def clean_and_parse_json(text):
    import re
    import json
    
    json_match = re.search(r'(\{.*\})', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass
    
    start_idx = text.find('{')
    end_idx = text.rfind('}')
    
    if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
        try:
            json_str = text[start_idx:end_idx+1]
            return json.loads(json_str)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse JSON even after extraction: {json_str}")
            
    logger.error(f"Could not extract valid JSON from: {text}")
    return None

def project_dropdown():

    project_options = [
        "WAVE CITY CLUB @ PSP 14A",
        "EWS_LIG Veridia PH04",
        "GH-8 Phase-2 (ELIGO) Wave City",
        "GH-8 Phase-3 (EDEN) Wave City",
        "Wave Oakwood, Wave City"
    ]
    project_name = st.sidebar.selectbox(
        "Project Name",
        options=project_options,
        index=project_options.index("Wave Oakwood, Wave City") if "Wave Oakwood, Wave City" in project_options else 0,
        key="project_name_selectbox",
        help="Select a project to fetch data and generate individual reports."
    )
    form_name = st.sidebar.text_input(
        "Form Name",
        "Non Conformity Report",
        key="form_name_input",
        help="Enter the form name for the report."
    )
    return project_name, form_name, project_options

def extract_ncr_summary_text(data):
    try:
        sites = data["Combined_NCR"]["NCR open beyond 21 days"].get("Sites", {})
        total_issues = 0
        site_summaries = []

        for site, details in sites.items():
            site_total = details.get("Total", 0)
            total_issues += site_total

            # Get unique modules per site
            modules_nested = details.get("Modules", [])
            modules_flat = sorted({m for sublist in modules_nested for m in sublist})

            site_summary = (
                f"{site}: {site_total} open issue(s), Modules: {', '.join(modules_flat)}"
            )
            site_summaries.append(site_summary)

        summary_text = (
            f"Total NCRs open beyond 21 days: {total_issues}\n\n"
            + "\n".join(site_summaries)
        )
        return summary_text

    except Exception as e:
        return ""

st.title("NCR Safety Housekeeping Reports")

# Initialize session state (unchanged)
if "ncr_df" not in st.session_state:
    st.session_state["ncr_df"] = None
if "safety_df" not in st.session_state:
    st.session_state["safety_df"] = None
if "housekeeping_df" not in st.session_state:
    st.session_state["housekeeping_df"] = None
if "session_id" not in st.session_state:
    st.session_state["session_id"] = None

def ProcessFiles(project):
    try:
        if project not in  ["WAVE CITY CLUB @ PSP 14A", "EWS_LIG Veridia PH04", "GH-8 Phase-2 (ELIGO) Wave City", "GH-8 Phase-3 (EDEN) Wave City", "Wave Oakwood, Wave City"]:
            change = {"veridia":"Wave Oakwood, Wave City", "eden":"GH-8 Phase-3 (EDEN) Wave City", "ews_lig":"EWS_LIG Veridia PH04","wave city club":"WAVE CITY CLUB @ PSP 14A","eligo":"GH-8 Phase-2 (ELIGO) Wave City"}
            project = change.get(project.lower())
            # st.write(project)
            st.write("Login to Asite 🔑")
            session_id = login_to_asite("impwatson@gadieltechnologies.com", "Srihari@790$")
            if session_id:
                st.session_state["session_id"] = session_id
    #Non Conformity Report
        if "session_id" in st.session_state:
            header, data, payload = fetch_project_data(st.session_state["session_id"], project, "Non Conformity Report")
            if data:
                df = process_json_data(data)
                st.session_state["ncr_df"] = df.copy()
                st.session_state["safety_df"] = df.copy()
                st.session_state["housekeeping_df"] = df.copy()
                st.dataframe(df)
                st.success("✅ Data fetched and processed successfully for all report types!")
        else:
            st.write("Login to Asite 🔑")
            session_id = login_to_asite("impwatson@gadieltechnologies.com", "Srihari@790$")
            if session_id:
                st.session_state["session_id"] = session_id
        #Non Conformity Report
            if "session_id" in st.session_state:
                header, data, payload = fetch_project_data(st.session_state["session_id"], project, "Non Conformity Report")
                if data:
                    df = process_json_data(data)
                    st.session_state["ncr_df"] = df.copy()
                    st.session_state["safety_df"] = df.copy()
                    st.session_state["housekeeping_df"] = df.copy()
                    st.dataframe(df)
                    st.success("✅ Data fetched and processed successfully for all report types!")
    except Exception as e:
        st.info("oops Login failed due to server error in Asite. Please try again later.")
        return "Asite Login Failed"

# Helper function to generate report title
def generate_report_title(prefix):
    now = datetime.now()  # Current date: April 25, 2025
    day = now.strftime("%d")
    month_name = now.strftime("%B")
    year = now.strftime("%Y")
    return f"{prefix}: {day}_{month_name}_{year}"




def GenerateFile(project, closed_end, open_end, closed_start):
        #===================================================VERIDIA=========================================================
        if project == "Wave Oakwood, Wave City":
            st.write("Thanks for the input 🤝")
            if st.session_state["ncr_df"] is not None and st.session_state["safety_df"] is not None and st.session_state["housekeeping_df"] is not None:
                st.write("Fetching data for Wave Oakwood, Wave City")
                ncr_df = st.session_state["ncr_df"]
                safety_df = st.session_state["safety_df"]
                housekeeping_df = st.session_state["housekeeping_df"]
                now = datetime.now()
                day = now.strftime("%d")
                year = now.strftime("%Y")
                month_name = closed_end.strftime("%B") if closed_end else now.strftime("%B")

                # Validate and format open_end
                if open_end is None:
                    st.error("❌ Please select a valid Open Until Date.")
                    logger.error("Open Until Date is not provided.")
                else:
                    # Convert open_end to string format for generate_ncr_report
                    open_end_str = open_end.strftime('%Y/%m/%d')

                    report_title_ncr = f"NCR: {day}_{month_name}_{year}"
                    closed_result_ncr, closed_raw_ncr = generate_ncr_report_for_veridia(ncr_df, "Closed", closed_start, closed_end)
                    open_result_ncr, open_raw_ncr = generate_ncr_report_for_veridia(ncr_df, "Open", Until_Date=open_end_str)

                    combined_result_ncr = {}
                    if "error" not in closed_result_ncr:
                        combined_result_ncr["NCR resolved beyond 21 days"] = closed_result_ncr["Closed"]
                    else:
                        combined_result_ncr["NCR resolved beyond 21 days"] = {"error": closed_result_ncr["error"]}
                    if "error" not in open_result_ncr:
                        combined_result_ncr["NCR open beyond 21 days"] = open_result_ncr["Open"]
                    else:
                        combined_result_ncr["NCR open beyond 21 days"] = {"error": open_result_ncr["error"]}

                    report_title_safety = f"Safety NCR: {day}_{month_name}_{year}"
                    closed_result_safety, closed_raw_safety = generate_ncr_Safety_report_for_veridia(
                        safety_df,
                        report_type="Closed",
                        start_date=closed_start.strftime('%Y/%m/%d') if closed_start else None,
                        end_date=closed_end.strftime('%Y/%m/%d') if closed_end else None,
                        open_until_date=None
                    )
                    open_result_safety, open_raw_safety = generate_ncr_Safety_report_for_veridia(
                        safety_df,
                        report_type="Open",
                        start_date=None,
                        end_date=None,
                        open_until_date=open_end_str
                    )

                    report_title_housekeeping = f"Housekeeping NCR: {day}_{month_name}_{year}"
                    closed_result_housekeeping, closed_raw_housekeeping = generate_ncr_Housekeeping_report_for_veridia(
                        housekeeping_df,
                        report_type="Closed",
                        start_date=closed_start.strftime('%Y/%m/%d') if closed_start else None,
                        end_date=closed_end.strftime('%Y/%m/%d') if closed_end else None,
                        open_until_date=None
                    )
                    open_result_housekeeping, open_raw_housekeeping = generate_ncr_Housekeeping_report_for_veridia(
                        housekeeping_df,
                        report_type="Open",
                        start_date=None,
                        end_date=None,
                        open_until_date=open_end_str
                    )

                    # st.subheader("Combined NCR Report (JSON)")
                    # st.json(combined_result_ncr)

                    st.session_state.all_reports_veridia = {
                        "Combined_NCR": combined_result_ncr,
                        "Safety_NCR_Closed": closed_result_safety,
                        "Safety_NCR_Open": open_result_safety,
                        "Housekeeping_NCR_Closed": closed_result_housekeeping,
                        "Housekeeping_NCR_Open": open_result_housekeeping
                    }

                    # st.write(st.session_state.all_reports_veridia)

                    st.session_state.veridiasummaries = extract_ncr_summary_text(st.session_state.all_reports_veridia)
                    st.session_state.excel_file = generate_combined_excel_report_for_veridia( st.session_state.all_reports_veridia, f"All_Reports_{day}_{month_name}_{year}")
                    st.download_button(
                        label="📥 Download All Reports Excel",
                        data=st.session_state.excel_file,
                        file_name=f"All_Reports_{day}_{month_name}_{year}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key="download_all_reports"
                    ) 
                    with st.chat_message("assistant"):
                        st.markdown("Whew! That was a lot of data. Now feel free to ask your questions in Veridia 🤗")
                        st.session_state.messages.append({"role": "assistant", "content": "Whew! That was a lot of data. Now feel free to ask your questions in Veridia 🤗"})
                       
            # else:
            #     st.error("Please fetch data first!")
    #===================================================VERIDIA=========================================================

    #===================================================EWS LIG=========================================================
        if project == "EWS_LIG Veridia PH04":
            
            if st.session_state["ncr_df"] is not None and st.session_state["safety_df"] is not None and st.session_state["housekeeping_df"] is not None:
                st.write("Fetching data for EWS_LIG Veridia PH04")
                ncr_df = st.session_state["ncr_df"]
                safety_df = st.session_state["safety_df"]
                housekeeping_df = st.session_state["housekeeping_df"]
                now = datetime.now()
                day = now.strftime("%d")
                year = now.strftime("%Y")
                month_name = closed_end.strftime("%B") if closed_end else now.strftime("%B")

                # Validate and format open_end
                if open_end is None:
                    st.error("❌ Please select a valid Open Until Date.")
                    logger.error("Open Until Date is not provided.")
                else:
                    # Convert open_end to string format for generate_ncr_report
                    open_end_str = open_end.strftime('%Y/%m/%d')

                    report_title_ncr = f"NCR: {day}_{month_name}_{year}"
                    closed_result_ncr, closed_raw_ncr = generate_ncr_report_for_ews(ncr_df, "Closed", closed_start, closed_end)
                    open_result_ncr, open_raw_ncr = generate_ncr_report_for_ews(ncr_df, "Open", Until_Date=open_end_str)

                    combined_result_ncr = {}
                    if "error" not in closed_result_ncr:
                        combined_result_ncr["NCR resolved beyond 21 days"] = closed_result_ncr["Closed"]
                    else:
                        combined_result_ncr["NCR resolved beyond 21 days"] = {"error": closed_result_ncr["error"]}
                    if "error" not in open_result_ncr:
                        combined_result_ncr["NCR open beyond 21 days"] = open_result_ncr["Open"]
                    else:
                        combined_result_ncr["NCR open beyond 21 days"] = {"error": open_result_ncr["error"]}

                    report_title_safety = f"Safety NCR: {day}_{month_name}_{year}"
                    closed_result_safety, closed_raw_safety = generate_ncr_Safety_report_for_ews(
                        safety_df,
                        report_type="Closed",
                        start_date=closed_start.strftime('%Y/%m/%d') if closed_start else None,
                        end_date=closed_end.strftime('%Y/%m/%d') if closed_end else None,
                        open_until_date=None
                    )
                    open_result_safety, open_raw_safety = generate_ncr_Safety_report_for_ews(
                        safety_df,
                        report_type="Open",
                        start_date=None,
                        end_date=None,
                        open_until_date=open_end_str
                    )

                    report_title_housekeeping = f"Housekeeping NCR: {day}_{month_name}_{year}"
                    closed_result_housekeeping, closed_raw_housekeeping = generate_ncr_Housekeeping_report_for_ews(
                        housekeeping_df,
                        report_type="Closed",
                        start_date=closed_start.strftime('%Y/%m/%d') if closed_start else None,
                        end_date=closed_end.strftime('%Y/%m/%d') if closed_end else None,
                        open_until_date=None
                    )
                    open_result_housekeeping, open_raw_housekeeping = generate_ncr_Housekeeping_report_for_ews(
                        housekeeping_df,
                        report_type="Open",
                        start_date=None,
                        end_date=None,
                        open_until_date=open_end_str
                    )

                    # st.subheader("Combined NCR Report (JSON)")
                    # st.json(combined_result_ncr)
                    # st.subheader("Safety NCR Closed Report (JSON)")
                    # st.json(closed_result_safety)
                    # st.subheader("Safety NCR Open Report (JSON)")
                    # st.json(open_result_safety)
                    # st.subheader("Housekeeping NCR Closed Report (JSON)")
                    # st.json(closed_result_housekeeping)
                    # st.subheader("Housekeeping NCR Open Report (JSON)")
                    # st.json(open_result_housekeeping)
                    st.session_state.all_reports_ewslig = {
                        "Combined_NCR": combined_result_ncr,
                        "Safety_NCR_Closed": closed_result_safety,
                        "Safety_NCR_Open": open_result_safety,
                        "Housekeeping_NCR_Closed": closed_result_housekeeping,
                        "Housekeeping_NCR_Open": open_result_housekeeping
                    }

                    st.session_state.ewsligsummaries = extract_ncr_summary_text(st.session_state.all_reports_ewslig)
              
                    st.session_state.excel_file = generate_combined_excel_report_for_veridia(st.session_state.all_reports_ewslig, f"All_Reports_{day}_{month_name}_{year}")
                    st.download_button(
                        label="📥 Download All Reports Excel",
                        data=st.session_state.excel_file,
                        file_name=f"All_Reports_{day}_{month_name}_{year}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key="download_all_reports"
                    )
            # else:
            #     st.error("Please fetch data first!")
    #===================================================EWS LIG=========================================================

    #===================================================EDEN=========================================================

        if project == "GH-8 Phase-3 (EDEN) Wave City":
            
            try:
                if st.session_state["ncr_df"] is not None and st.session_state["safety_df"] is not None and st.session_state["housekeeping_df"] is not None:
                    st.write("Fetching data for GH-8 Phase-3 (EDEN) Wave City")
                    ncr_df = st.session_state["ncr_df"]
                    safety_df = st.session_state["safety_df"]
                    housekeeping_df = st.session_state["housekeeping_df"]
                    now = datetime.now()
                    day = now.strftime("%d")
                    year = now.strftime("%Y")
                    month_name = closed_end.strftime("%B") if closed_end else now.strftime("%B")

                    # Validate and format open_end
                    if open_end is None:
                        st.error("❌ Please select a valid Open Until Date.")
                        logger.error("Open Until Date is not provided.")
                    else:
                        # Convert open_end to string format for generate_ncr_report
                        open_end_str = open_end.strftime('%Y/%m/%d')

                        report_title_ncr = f"NCR: {day}_{month_name}_{year}"
                        closed_result_ncr, closed_raw_ncr = generate_ncr_report_for_eden(ncr_df, "Closed", closed_start, closed_end)
                        open_result_ncr, open_raw_ncr = generate_ncr_report_for_eden(ncr_df, "Open", until_date=open_end_str)

                        combined_result_ncr = {}
                        if "error" not in closed_result_ncr:
                            combined_result_ncr["NCR resolved beyond 21 days"] = closed_result_ncr["Closed"]
                        else:
                            combined_result_ncr["NCR resolved beyond 21 days"] = {"error": closed_result_ncr["error"]}
                        if "error" not in open_result_ncr:
                            combined_result_ncr["NCR open beyond 21 days"] = open_result_ncr["Open"]
                        else:
                            combined_result_ncr["NCR open beyond 21 days"] = {"error": open_result_ncr["error"]}

                        report_title_safety = f"Safety NCR: {day}_{month_name}_{year}"
                        closed_result_safety, closed_raw_safety = generate_ncr_Safety_report_for_eden(
                            safety_df,
                            report_type="Closed",
                            start_date=closed_start.strftime('%Y/%m/%d') if closed_start else None,
                            end_date=closed_end.strftime('%Y/%m/%d') if closed_end else None,
                            until_date=None
                        )
                        open_result_safety, open_raw_safety = generate_ncr_Safety_report_for_eden(
                            safety_df,
                            report_type="Open",
                            start_date=None,
                            end_date=None,
                            until_date=open_end_str
                        )

                        report_title_housekeeping = f"Housekeeping NCR: {day}_{month_name}_{year}"
                        closed_result_housekeeping, closed_raw_housekeeping = generate_ncr_Housekeeping_report_for_eden(
                            housekeeping_df,
                            report_type="Closed",
                            start_date=closed_start.strftime('%Y/%m/%d') if closed_start else None,
                            end_date=closed_end.strftime('%Y/%m/%d') if closed_end else None,
                            until_date=None
                        )
                        open_result_housekeeping, open_raw_housekeeping = generate_ncr_Housekeeping_report_for_eden(
                            housekeeping_df,
                            report_type="Open",
                            start_date=None,
                            end_date=None,
                            until_date=open_end_str
                        )

                        st.session_state.all_reports_eden = {
                            "Combined_NCR": combined_result_ncr,
                            "Safety_NCR_Closed": closed_result_safety,
                            "Safety_NCR_Open": open_result_safety,
                            "Housekeeping_NCR_Closed": closed_result_housekeeping,
                            "Housekeeping_NCR_Open": open_result_housekeeping
                        }

                        st.session_state.edensummaries = extract_ncr_summary_text(st.session_state.all_reports_eden)
                
                        st.session_state.excel_file = generate_combined_excel_report_for_veridia(st.session_state.all_reports_eden, f"All_Reports_{day}_{month_name}_{year}")
                        st.download_button(
                            label="📥 Download All Reports Excel",
                            data=st.session_state.excel_file,
                            file_name=f"All_Reports_{day}_{month_name}_{year}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            key="download_all_reports"
                        )
            except Exception as e:
                st.error(f"Error generating reports: {e}")
        # else:
        #     st.error("Please fetch data first!")

        if project == "WAVE CITY CLUB @ PSP 14A":
            
            if st.session_state["ncr_df"] is not None and st.session_state["safety_df"] is not None and st.session_state["housekeeping_df"] is not None:

                st.write("Fetching data for WAVE CITY CLUB @ PSP 14A")
                ncr_df = st.session_state["ncr_df"]
                safety_df = st.session_state["safety_df"]
                housekeeping_df = st.session_state["housekeeping_df"]
                now = datetime.now()
                day = now.strftime("%d")
                year = now.strftime("%Y")
                month_name = closed_end.strftime("%B") if closed_end else now.strftime("%B")

                # Validate and format open_end
                if open_end is None:
                    st.error("❌ Please select a valid Open Until Date.")
                    logger.error("Open Until Date is not provided.")
                else:
                    # Convert open_end to string format for generate_ncr_report
                    open_end_str = open_end.strftime('%Y/%m/%d')

                    report_title_ncr = f"NCR: {day}_{month_name}_{year}"
                    closed_result_ncr, closed_raw_ncr = generate_ncr_report_for_club(ncr_df, "Closed", closed_start, closed_end)
                    open_result_ncr, open_raw_ncr = generate_ncr_report_for_club(ncr_df, "Open", Until_Date=open_end_str)

                    combined_result_ncr = {}
                    if "error" not in closed_result_ncr:
                        combined_result_ncr["NCR resolved beyond 21 days"] = closed_result_ncr["Closed"]
                    else:
                        combined_result_ncr["NCR resolved beyond 21 days"] = {"error": closed_result_ncr["error"]}
                    if "error" not in open_result_ncr:
                        combined_result_ncr["NCR open beyond 21 days"] = open_result_ncr["Open"]
                    else:
                        combined_result_ncr["NCR open beyond 21 days"] = {"error": open_result_ncr["error"]}

                    report_title_safety = f"Safety NCR: {day}_{month_name}_{year}"
                    closed_result_safety, closed_raw_safety = generate_ncr_Safety_report_for_club(
                        safety_df,
                        report_type="Closed",
                        start_date=closed_start.strftime('%Y/%m/%d') if closed_start else None,
                        end_date=closed_end.strftime('%Y/%m/%d') if closed_end else None,
                        open_until_date=None
                    )
                    open_result_safety, open_raw_safety = generate_ncr_Safety_report_for_club(
                        safety_df,
                        report_type="Open",
                        start_date=None,
                        end_date=None,
                        open_until_date=open_end_str
                    )

                    report_title_housekeeping = f"Housekeeping NCR: {day}_{month_name}_{year}"
                    closed_result_housekeeping, closed_raw_housekeeping = generate_ncr_Housekeeping_report_for_club(
                        housekeeping_df,
                        report_type="Closed",
                        start_date=closed_start.strftime('%Y/%m/%d') if closed_start else None,
                        end_date=closed_end.strftime('%Y/%m/%d') if closed_end else None,
                        open_until_date=None
                    )
                    open_result_housekeeping, open_raw_housekeeping = generate_ncr_Housekeeping_report_for_club(
                        housekeeping_df,
                        report_type="Open",
                        start_date=None,
                        end_date=None,
                        open_until_date=open_end_str
                    )

                    st.session_state.all_reports_for_club = {
                        "Combined_NCR": combined_result_ncr,
                        "Safety_NCR_Closed": closed_result_safety,
                        "Safety_NCR_Open": open_result_safety,
                        "Housekeeping_NCR_Closed": closed_result_housekeeping,
                        "Housekeeping_NCR_Open": open_result_housekeeping
                    }

                    # st.subheader("Combined NCR Report (JSON)")
                    # st.json(combined_result_ncr)
                    # st.subheader("Safety NCR Closed Report (JSON)")
                    # st.json(closed_result_safety)
                    # st.subheader("Safety NCR Open Report (JSON)")
                    # st.json(open_result_safety)
                    # st.subheader("Housekeeping NCR Closed Report (JSON)")
                    # st.json(closed_result_housekeeping)
                    # st.subheader("Housekeeping NCR Open Report (JSON)")
                    # st.json(open_result_housekeeping)

                    st.session_state.wavecitysummaries = extract_ncr_summary_text(st.session_state.all_reports_for_club)
                    st.session_state.excel_file  = generate_combined_excel_report_for_club(st.session_state.all_reports_for_club, f"All_Reports_{day}_{month_name}_{year}")
                    st.download_button(
                        label="📥 Download All Reports Excel",
                        data=st.session_state.excel_file,
                        file_name=f"All_Reports_{day}_{month_name}_{year}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key="download_all_reports"
                    )
        # else:
        #     st.error("Please fetch data first!")



            




