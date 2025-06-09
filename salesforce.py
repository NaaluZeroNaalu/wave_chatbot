import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
from simple_salesforce import Salesforce
import datetime
import time
import logging
from typing import Dict, Any, Optional, Tuple
import json

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
if not logger.hasHandlers():
    logger.addHandler(handler)

# Initialize session state
def init_session_state():
    """Initialize all session state variables"""
    defaults = {
        "leads_df": pd.DataFrame(),
        "messages": [],
        "filters": {},
        "industry_trends": {
            "Real Estate Demand": "Increasing in urban areas",
            "Mortgage Rates": "Currently at 6.5% (down 0.2% from last month)",
            "Hot Markets": "Bangalore, Hyderabad, Pune",
            "Buyer Preferences": "Shift towards affordable luxury segment"
        },
        "forecast_period": 30,
        "financial_years": {},
        "last_refresh": 0.0,
        "data_module": None,
        "sf_connection": None
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# Financial Year Functions
def get_current_financial_year() -> str:
    """Get current financial year (April to March)"""
    today = datetime.date.today()
    year = today.year
    if today.month < 4:
        return f"FY{year-1}-{str(year)[-2:]}"
    else:
        return f"FY{year}-{str(year+1)[-2:]}"

def get_financial_year(date) -> str:
    """Get financial year for a given date"""
    if pd.isnull(date):
        return "Unknown"
    
    if isinstance(date, pd.Timestamp):
        date = date.date()
    elif isinstance(date, str):
        try:
            date = pd.to_datetime(date).date()
        except:
            return "Unknown"
    
    year = date.year
    if date.month < 4:
        return f"FY{year-1}-{str(year)[-2:]}"
    else:
        return f"FY{year}-{str(year+1)[-2:]}"

def categorize_by_financial_year(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Categorize dataframe by financial year"""
    if df.empty or 'CreatedDate' not in df.columns:
        return {}
    
    # Create financial year column
    df = df.copy()
    df['FinancialYear'] = df['CreatedDate'].apply(get_financial_year)
    
    # Get valid years (last 3 years)
    current_year = datetime.date.today().year
    min_year = current_year - 2
    
    fy_data = {}
    for fy in df['FinancialYear'].unique():
        if fy != "Unknown":
            year_part = int(fy.split('-')[0][2:]) + 2000
            if year_part >= min_year:
                fy_data[fy] = df[df['FinancialYear'] == fy].copy()
    
    return fy_data

# Mock Data Generator (for testing without Salesforce)
def generate_mock_data() -> pd.DataFrame:
    """Generate mock lead data for testing"""
    np.random.seed(42)
    
    # Generate dates for the last 2 years
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=730)
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Generate random data
    n_records = 5000
    data = {
        'Id': [f'00Q{str(i).zfill(15)}' for i in range(n_records)],
        'CreatedDate': np.random.choice(date_range, n_records),
        'Name': [f'Lead {i}' for i in range(n_records)],
        'Email': [f'lead{i}@example.com' for i in range(n_records)],
        'Phone': [f'+91{np.random.randint(7000000000, 9999999999)}' for _ in range(n_records)],
        'Status': np.random.choice(['New', 'Working', 'Qualified', 'Unqualified', 'Nurturing'], n_records, p=[0.3, 0.25, 0.15, 0.2, 0.1]),
        'LeadSource': np.random.choice(['Website', 'Facebook', 'Google Ads', 'Referral', 'Direct'], n_records),
        'Project__c': np.random.choice(['Project Alpha', 'Project Beta', 'Project Gamma', 'Project Delta'], n_records),
        'Campaign_Name__c': np.random.choice(['Summer Campaign', 'Festive Offer', 'New Year Special', 'Monsoon Sale', 'Winter Campaign'], n_records),
        'City__c': np.random.choice(['Bangalore', 'Hyderabad', 'Pune', 'Chennai', 'Mumbai'], n_records),
        'Budget_Range__c': np.random.choice(['Below 50L', '50L-1Cr', '1Cr-2Cr', 'Above 2Cr'], n_records),
        'Property_Type__c': np.random.choice(['Apartment', 'Villa', 'Plot', 'Commercial'], n_records),
        'OwnerId': np.random.choice([f'005{str(i).zfill(15)}' for i in range(10)], n_records),
        'Junk_Reason__c': np.random.choice([None, 'Invalid Phone', 'Duplicate', 'Test Lead'], n_records, p=[0.85, 0.05, 0.05, 0.05]),
        'Is_appointment_Booked__c': np.random.choice([True, False], n_records, p=[0.1, 0.9]),
        # Suppose we have a 'FlatType__c' column directly in mock for demonstration
        # Randomly assign "2 BHK" or "3 BHK" in mock
        'FlatType__c': np.random.choice(['2 BHK', '3 BHK'], n_records, p=[0.6, 0.4])
    }
    
    df = pd.DataFrame(data)
    df['CreatedDate'] = pd.to_datetime(df['CreatedDate'])
    # Make CreatedDate timezone-aware (UTC) to match Salesforce timestamps
    df['CreatedDate'] = df['CreatedDate'].dt.tz_localize('UTC')
    
    return df

# IBM Granite Conversational AI Integration
def query_ibm_model(prompt: str) -> str:
    """Call IBM Watsonx Granite model to generate a conversational answer."""
    api_key = os.getenv("WATSONX_API_KEY") or os.getenv("IBM_API_KEY")
    project_id = os.getenv("WATSONX_PROJECT_ID") or os.getenv("IBM_PROJECT_ID")
    cluster_url = os.getenv("WATSONX_URL") or os.getenv("IBM_CLUSTER_URL")  # e.g., "us-south.ml.cloud.ibm.com"
    model_id = os.getenv("WATSONX_MODEL_ID") or "ibm-granite/granite-3.1-8b-instruct"

    # Ensure credentials are set
    if not (api_key and project_id and cluster_url and model_id):
        return "Error: IBM GenAI credentials not configured."
    
    # 1. Get IAM token from IBM Cloud
    token_resp = requests.post(
        "https://iam.cloud.ibm.com/identity/token",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        data={"grant_type": "urn:ibm:params:oauth:grant-type:apikey", "apikey": api_key}
    )
    if not token_resp.ok:
        return f"Error getting IBM IAM token: {token_resp.text}"
    token = token_resp.json().get("access_token")

    # 2. Call the Granite text-generation endpoint
    endpoint = f"https://{cluster_url}/ml/v1/text/generation?version=2025-02-11"
    payload = {
        "input": prompt,
        "model_id": model_id,
        "project_id": project_id,
        "parameters": {
            "max_new_tokens": 500,
            "temperature": 0.7,
            "top_p": 0.95
        }
    }
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    gen_resp = requests.post(endpoint, json=payload, headers=headers)
    if not gen_resp.ok:
        return f"Error from IBM GenAI: {gen_resp.text}"
    return gen_resp.json().get("generated_text", "")

# Salesforce Connection
def get_salesforce_data() -> pd.DataFrame:
    """Get data from Salesforce or return mock data, then apply default date filter."""
    try:
        # Check if we should use cached data
        current_time = time.time()
        last_refresh = st.session_state.get('last_refresh', 0.0)
        
        if (current_time - last_refresh) < 3600 and not st.session_state.leads_df.empty:
            df_cached = st.session_state.leads_df.copy()
            # Apply default filter on cached data as well
            if 'CreatedDate' in df_cached.columns:
                cutoff = pd.Timestamp('2023-04-01', tz='UTC')
                df_cached = df_cached[df_cached['CreatedDate'] >= cutoff]
            return df_cached
        
        # Check if Salesforce credentials are available
        required_env_vars = ['SALESFORCE_CLIENT_ID', 'SALESFORCE_CLIENT_SECRET', 
                             'SALESFORCE_USERNAME', 'SALESFORCE_PASSWORD']
        
        if not all(os.getenv(var) for var in required_env_vars):
            st.warning("Salesforce credentials not found. Using mock data for demonstration.")
            df = generate_mock_data()
            # Apply default filter to mock data
            if 'CreatedDate' in df.columns:
                cutoff = pd.Timestamp('2023-04-01', tz='UTC')
                df = df[df['CreatedDate'] >= cutoff]
            st.session_state.leads_df = df
            st.session_state.last_refresh = current_time
            return df
        
        # Try to connect to Salesforce
        auth_url = "https://login.salesforce.com/services/oauth2/token"
        payload = {
            "grant_type": "password",
            "client_id": os.getenv("SALESFORCE_CLIENT_ID"),
            "client_secret": os.getenv("SALESFORCE_CLIENT_SECRET"),
            "username": os.getenv("SALESFORCE_USERNAME"),
            "password": os.getenv("SALESFORCE_PASSWORD")
        }
        
        response = requests.post(auth_url, data=payload, timeout=30)
        
        if response.status_code != 200:
            st.warning(f"Salesforce authentication failed. Using mock data. Error: {response.text}")
            df = generate_mock_data()
            # Apply default filter to mock data
            if 'CreatedDate' in df.columns:
                cutoff = pd.Timestamp('2023-04-01', tz='UTC')
                df = df[df['CreatedDate'] >= cutoff]
            st.session_state.leads_df = df
            st.session_state.last_refresh = current_time
            return df
        
        auth_data = response.json()
        access_token = auth_data['access_token']
        instance_url = auth_data['instance_url']
        
        # Create Salesforce connection
        sf = Salesforce(instance_url=instance_url, session_id=access_token)
        
        # Calculate date range
        today = datetime.date.today()
        start_date = today - datetime.timedelta(days=730)  # 2 years
        start_date_str = start_date.strftime('%Y-%m-%d')
        
        # SOQL Query - using only standard Lead fields
        query = f"""
        SELECT 
            Id, Adgroup__c, Phone__c, Area_Interested_In__c, Banner_Size__c, 
            C_O__c, Campaign_Name__c, Campaign_Short_Code__c, City__c, 
            Communication_C_O__c, Communication_City__c, Communication_Country__c, 
            Communication_District__c, Communication_House_Number__c, 
            Communication_Postal_Code__c, Communication_State__c, 
            Communication_Street_1__c, Communication_Street_2__c, 
            Communication_Street_3__c, Company, Contact_Medium__c, 
            Mobile__c, Contact_on_Whatsapp__c, Country_SAP__c, 
            CreatedById, CreatedDate, Customer_Feedback__c, 
            Description, Disqualification_Date__c, Disqualification_Reason__c, 
            Disqualified_Date_Time__c, District__c, Email__c, 
            Facebbok_Campaign_Name__c, Facebook_Ad_Name__c, 
            Facebook_Ad_Set_Name__c, Facebook_Platform__c, 
            Follow_up__c, Follow_Up_Date_Time__c, Follow_UP_Remarks__c, 
            House_Number__c, International_Contact_Number__c, 
            IP_Address__c, Junk_Reason__c, LastModifiedById, 
            OwnerId, Lead_Converted__c, LeadSource, 
            Lead_Sources_Comment__c, Lead_Source_Sub_Category__c, 
            Status, Name, Open_Lead_reasons__c, 
            Source_Campaign__c, Postal_Code__c, Preferred_Location__c, 
            Project_Category__c, Project__c, Property_Size__c, 
            Property_Type__c, Purpose_of_Purchase__c, 
            Budget_Range__c, Rating, Reason_for_Purchase__c, 
            Same_As_Permanent_Address__c, State__c, Street_1__c, 
            Street_2__c, Street_3__c, Time_Frame_In_Which_Looking_To_Buy__c, 
            Title, UTM_Code__c 
        FROM Lead
        WHERE CreatedDate >= {start_date_str}T00:00:00Z
        ORDER BY CreatedDate DESC
        """
        
        # Execute query
        results = sf.query_all(query)['records']
        
        if results:
            df = pd.DataFrame(results)
            df = df.drop('attributes', axis=1, errors='ignore')
            
            # Process dates from Salesforce (UTC-aware)
            date_cols = ['CreatedDate', 'LastModifiedDate']
            for col in date_cols:
                if col in df.columns:
                    # Parse with UTC
                    df[col] = pd.to_datetime(df[col], errors='coerce', utc=True)
            
            # Add mock custom fields for demonstration (if missing)
            n_records = len(df)
            if 'Project__c' not in df.columns:
                df['Project__c'] = np.random.choice(['Project Alpha', 'Project Beta', 'Project Gamma'], n_records)
            if 'Campaign_Name__c' not in df.columns:
                df['Campaign_Name__c'] = np.random.choice(['Summer Campaign', 'Festive Offer', 'New Year Special'], n_records)
            if 'Budget_Range__c' not in df.columns:
                df['Budget_Range__c'] = np.random.choice(['Below 50L', '50L-1Cr', '1Cr-2Cr'], n_records)
            if 'Property_Type__c' not in df.columns:
                df['Property_Type__c'] = np.random.choice(['Apartment', 'Villa', 'Plot'], n_records)
            if 'Junk_Reason__c' not in df.columns:
                df['Junk_Reason__c'] = np.random.choice([None, 'Invalid Phone', 'Duplicate'], n_records, p=[0.9, 0.05, 0.05])
            if 'Is_appointment_Booked__c' not in df.columns:
                df['Is_appointment_Booked__c'] = np.random.choice([True, False], n_records, p=[0.1, 0.9])
            
            # Ensure we have FlatType__c in Salesforce output. If not, create a placeholder:
            if 'FlatType__c' not in df.columns:
                # You may already have a field in Salesforce for bedroom count;
                # if not, create a dummy column for demonstration:
                df['FlatType__c'] = np.random.choice(['2 BHK', '3 BHK'], n_records, p=[0.6, 0.4])
            
        else:
            st.warning("No records found in Salesforce. Using mock data.")
            df = generate_mock_data()
        
        # **Apply default date filter to Salesforce or mock data**
        if 'CreatedDate' in df.columns:
            cutoff = pd.Timestamp('2023-04-01', tz='UTC')
            # If CreatedDate is naive (mock), localize to UTC first
            if df['CreatedDate'].dt.tz is None:
                df['CreatedDate'] = df['CreatedDate'].dt.tz_localize('UTC')
            df = df[df['CreatedDate'] >= cutoff]
        
        # Fill missing values
        text_cols = ['Status', 'LeadSource', 'Project__c', 'Campaign_Name__c']
        for col in text_cols:
            if col in df.columns:
                df[col] = df[col].fillna('Unknown')
        
        # Store in session state
        st.session_state.leads_df = df
        st.session_state.last_refresh = current_time
        
        return df
        
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        logger.error(f"Salesforce connection error: {str(e)}")
        
        # Fallback to mock data
        df = generate_mock_data()
        # Apply default date filter to fallback data
        if 'CreatedDate' in df.columns:
            cutoff = pd.Timestamp('2023-04-01', tz='UTC')
            df = df[df['CreatedDate'] >= cutoff]
        st.session_state.leads_df = df
        st.session_state.last_refresh = time.time()
        return df

# Analytics Functions
def calculate_overall_funnel(df: pd.DataFrame, fy: Optional[str] = None) -> Dict[str, Any]:
    """Calculate funnel metrics"""
    if df.empty:
        return {}
    
    # Calculate basic metrics
    total_leads = len(df)
    valid_leads = len(df[df['Junk_Reason__c'].isnull()]) if 'Junk_Reason__c' in df.columns else total_leads
    qualified_leads = len(df[df['Status'].str.lower().str.contains('qualified', na=False)])
    meetings_booked = len(df[df.get('Is_appointment_Booked__c', pd.Series([False] * len(df))) == True])
    disqualified = len(df[df['Status'].str.lower().str.contains('unqualified', na=False)])
    open_leads = len(df[df['Status'].isin(['New', 'Working', 'Nurturing'])])
    
    metrics = {
        'Total Leads': total_leads,
        'Valid Leads': valid_leads,
        'Qualified Leads': qualified_leads,
        'Meetings Booked': meetings_booked,
        'Disqualified': disqualified,
        'Open Leads': open_leads
    }
    
    # Calculate ratios safely
    metrics['Junk %'] = round(((total_leads - valid_leads) / total_leads) * 100, 2) if total_leads > 0 else 0
    metrics['VL:Qualified'] = round(valid_leads / qualified_leads, 2) if qualified_leads > 0 else 0
    metrics['Qualified:Meetings'] = round(qualified_leads / meetings_booked, 2) if meetings_booked > 0 else 0
    metrics['Conversion Rate'] = round((qualified_leads / total_leads) * 100, 2) if total_leads > 0 else 0
    
    if fy:
        metrics['Financial Year'] = fy
    
    return metrics

def calculate_employee_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate metrics by employee/owner"""
    if df.empty or 'OwnerId' not in df.columns:
        return pd.DataFrame()
    
    results = []
    for owner_id, owner_df in df.groupby('OwnerId'):
        metrics = calculate_overall_funnel(owner_df)
        results.append({
            'Employee ID': owner_id,
            'Total Leads': metrics['Total Leads'],
            'Valid Leads': metrics['Valid Leads'],
            'Qualified': metrics['Qualified Leads'],
            'Meetings': metrics['Meetings Booked'],
            'Conversion Rate': metrics['Conversion Rate']
        })
    
    return pd.DataFrame(results)

def calculate_campaign_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate metrics by campaign"""
    if df.empty or 'Campaign_Name__c' not in df.columns:
        return pd.DataFrame()
    
    results = []
    for campaign, campaign_df in df.groupby('Campaign_Name__c'):
        metrics = calculate_overall_funnel(campaign_df)
        
        # Mock cost data
        cost = np.random.randint(10000, 100000)
        cpl = round(cost / metrics['Total Leads'], 2) if metrics['Total Leads'] > 0 else 0
        
        results.append({
            'Campaign': campaign,
            'Total Leads': metrics['Total Leads'],
            'Valid Leads': metrics['Valid Leads'],
            'Qualified': metrics['Qualified Leads'],
            'Meetings': metrics['Meetings Booked'],
            'Cost': cost,
            'CPL': cpl,
            'Conversion Rate': metrics['Conversion Rate']
        })
    
    return pd.DataFrame(results)

# Visualization Functions
def plot_funnel(df: pd.DataFrame):
    """Create funnel visualization"""
    if df.empty:
        st.warning("No data available for funnel visualization")
        return
    
    metrics = calculate_overall_funnel(df)
    
    stages = ['Total Leads', 'Valid Leads', 'Qualified Leads', 'Meetings Booked']
    values = [
        metrics['Total Leads'],
        metrics['Valid Leads'],
        metrics['Qualified Leads'],
        metrics['Meetings Booked']
    ]
    
    fig = px.funnel(
        x=values,
        y=stages,
        title='Lead Conversion Funnel',
        color_discrete_sequence=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    )
    
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

def plot_trends(df: pd.DataFrame):
    """Plot lead trends over time"""
    if df.empty or 'CreatedDate' not in df.columns:
        st.warning("No date data available for trends")
        return
    
    # Daily lead counts
    # Ensure CreatedDate is converted to date (drop timezone)
    daily_counts = df.copy()
    daily_counts['Date'] = daily_counts['CreatedDate'].dt.date
    daily_counts = daily_counts.groupby('Date').size().reset_index()
    daily_counts.columns = ['Date', 'Leads']
    
    fig = px.line(
        daily_counts, 
        x='Date', 
        y='Leads',
        title='Daily Lead Trends',
        markers=True
    )
    
    st.plotly_chart(fig, use_container_width=True)

def plot_source_distribution(df: pd.DataFrame):
    """Plot lead source distribution"""
    if df.empty or 'LeadSource' not in df.columns:
        return
    
    source_counts = df['LeadSource'].value_counts()
    
    fig = px.pie(
        values=source_counts.values,
        names=source_counts.index,
        title='Lead Sources Distribution'
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Filtering System
def apply_filters(df: pd.DataFrame, filters: Dict) -> pd.DataFrame:
    """Apply filters to dataframe"""
    if not filters or df.empty:
        return df
    
    filtered_df = df.copy()
    
    # Status filter
    if 'status' in filters and filters['status']:
        filtered_df = filtered_df[filtered_df['Status'].isin(filters['status'])]
    
    # Date range filter - Handle timezone issues
    if 'CreatedDate' in filtered_df.columns:
        
        # Start date filter
        if 'start_date' in filters and filters['start_date']:
            start_datetime = pd.to_datetime(filters['start_date'])
            # Make timezone-aware to match df
            start_datetime = pd.Timestamp(start_datetime.date(), tz='UTC')
            
            try:
                filtered_df = filtered_df[filtered_df['CreatedDate'] >= start_datetime]
            except Exception as e:
                logger.warning(f"Could not apply start date filter: {e}")
        
        # End date filter
        if 'end_date' in filters and filters['end_date']:
            end_datetime = pd.to_datetime(filters['end_date']) + pd.Timedelta(days=1)
            end_datetime = pd.Timestamp(end_datetime.date(), tz='UTC')
            
            try:
                filtered_df = filtered_df[filtered_df['CreatedDate'] < end_datetime]
            except Exception as e:
                logger.warning(f"Could not apply end date filter: {e}")
    
    # Other filters - Updated to match the CSV column names
    filter_mappings = {
        'project': 'Project__c',
        'lead_source': 'LeadSource', 
        'campaign': 'Campaign_Name__c',
        'city': 'City__c',
        'property_type': 'Property_Type__c',
        'budget_range': 'Budget_Range__c'
    }
    
    for filter_key, column_name in filter_mappings.items():
        if filter_key in filters and filters[filter_key] and column_name in filtered_df.columns:
            filtered_df = filtered_df[filtered_df[column_name] == filters[filter_key]]
    
    return filtered_df

# Also update the date processing in get_salesforce_data function
def process_salesforce_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Process and standardize date columns from Salesforce/CSV data"""
    date_columns = [
        'CreatedDate', 
        'LastModifiedDate', 
        'Disqualification_Date__c',
        'Disqualified_Date_Time__c',
        'Follow_Up_Date_Time__c',
        'Preferred_Date_of_Visit__c'
    ]
    
    for col in date_columns:
        if col in df.columns:
            try:
                # Convert to datetime and handle timezone
                df[col] = pd.to_datetime(df[col], errors='coerce', utc=True)
            except Exception as e:
                logger.warning(f"Could not process date column {col}: {e}")
                # Keep as is if conversion fails
                pass
    
    return df

# AI Assistant Functions
def get_ai_response(query: str, df: pd.DataFrame) -> str:
    """Generate AI response based on query and data, using IBM Granite model."""
    
    # Handle specific commands
    if "funnel" in query.lower():
        plot_funnel(df)
        return "Here's the lead conversion funnel visualization 📊"
    
    elif "trend" in query.lower():
        plot_trends(df)
        return "Here are the lead trends over time 📈"
    
    elif "source" in query.lower():
        plot_source_distribution(df)
        return "Here's the lead source distribution 🎯"
    
    elif "employee" in query.lower() or "performance" in query.lower():
        emp_df = calculate_employee_metrics(df)
        if not emp_df.empty:
            st.dataframe(emp_df, use_container_width=True)
            return "Here are the employee performance metrics 👨‍💼"
        return "No employee data available"
    
    elif "campaign" in query.lower():
        campaign_df = calculate_campaign_metrics(df)
        if not campaign_df.empty:
            st.dataframe(campaign_df, use_container_width=True)
            return "Here's the campaign performance analysis 📣"
        return "No campaign data available"
    
    elif "financial year" in query.lower() or "fy" in query.lower():
        if st.session_state.financial_years:
            fy_metrics = []
            for fy, fy_df in st.session_state.financial_years.items():
                metrics = calculate_overall_funnel(fy_df, fy)
                fy_metrics.append(metrics)
            
            fy_comparison = pd.DataFrame(fy_metrics)
            if 'Financial Year' in fy_comparison.columns:
                fy_comparison = fy_comparison.set_index('Financial Year')
            st.dataframe(fy_comparison, use_container_width=True)
            
            return "Here's the financial year comparison 📅"
        return "No financial year data available"
    
    # For all other queries, use conversational AI
    # Build a context string with key metrics
    metrics = calculate_overall_funnel(df)
    context = (f"Total leads: {metrics.get('Total Leads', 0)}, "
               f"Valid leads: {metrics.get('Valid Leads', 0)}, "
               f"Qualified leads: {metrics.get('Qualified Leads', 0)}, "
               f"Meetings booked: {metrics.get('Meetings Booked', 0)}, "
               f"Conversion rate: {metrics.get('Conversion Rate', 0)}%. ")
    prompt = (f"You are a real-estate analytics assistant. Current data context: {context} "
              f"The user asks: \"{query}\". Provide a detailed, insightful response, "
              f"including trends, strategic commentary, and any relevant observations from the data.")
    
    ai_answer = query_ibm_model(prompt)
    return ai_answer or "Sorry, I couldn't generate a response."

# --------------------------------------------------------------------------
# ↓↓↓ Here is the new helper that reads this entire file and asks your LLM ↓↓↓
# --------------------------------------------------------------------------

def ask_llm_to_read_file(question: str) -> str:
    """
    Read the entire current Python file (this .py script), 
    then ask the configured Watsonx LLM to analyze its schema 
    and generate a contextual answer to `question`.
    """
    try:
        # 1. Read this file's source code
        this_file_path = __file__  # Path to the currently running script
        with open(this_file_path, 'r', encoding='utf-8') as f:
            code_text = f.read()
    except Exception as e:
        return f"Error reading file: {e}"
    
    # 2. Build a single prompt that includes:
    #    a) the full code
    #    b) the specific question to answer
    full_prompt = (
        "Below is the complete Python source code for my Streamlit app. "
        "The schema and DataFrame creation logic are all in this code. "
        "Please read it carefully and then answer the question at the end.\n\n"
        "=== BEGIN FULL CODE ===\n"
        f"{code_text}\n"
        "=== END FULL CODE ===\n\n"
        "QUESTION:\n"
        f"{question}\n\n"
        "Please:\n"
        "  • Identify which column(s) in the code correspond to flat types (e.g. 2 BHK vs. 3 BHK).\n"
        "  • Explain how you locate qualified leads in the DataFrame (which 'Status' value counts as qualified).\n"
        "  • Compute (or state how you would compute) which flat type is most preferred among those qualified leads.\n"
        "  • Provide your final answer as: “Most preferred flat type among Qualified leads is ___ (count: __).”\n"
        "Use any internal reasoning you need, but respond succinctly."
    )
    
    # 3. Send that prompt to Watsonx Granite
    response = query_ibm_model(full_prompt)
    return response or "Error: No response from LLM."

# --------------------------------------------------------------------------
# ↑↑↑ End of the new LLM‐on‐code helper function ↑↑↑
# --------------------------------------------------------------------------

# Main Streamlit App
def main():
    st.set_page_config(
        page_title="Real Estate Analytics Assistant",
        page_icon="🏢",
        layout="wide"
    )
    
    st.title("🏢 Real Estate Analytics Assistant")
    st.caption("AI-powered Salesforce Analytics with Financial Year Insights")
    
    # Initialize session state
    init_session_state()
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # Data refresh button
        if st.button("🔄 Refresh Data", use_container_width=True):
            with st.spinner("Fetching latest data..."):
                leads_df = get_salesforce_data()
                if not leads_df.empty:
                    st.success(f"✅ Loaded {len(leads_df)} leads")
                    st.session_state.financial_years = categorize_by_financial_year(leads_df)
                    if st.session_state.financial_years:
                        st.success(f"📅 Financial years: {', '.join(st.session_state.financial_years.keys())}")
        
        st.divider()
        
        # Filters
        st.subheader("🔍 Data Filters")
        
        leads_df = st.session_state.leads_df
        if not leads_df.empty:
            # Status filter
            if 'Status' in leads_df.columns:
                status_options = leads_df['Status'].dropna().unique().tolist()
                selected_status = st.multiselect(
                    "Filter by Status",
                    options=status_options,
                    default=st.session_state.filters.get('status', [])
                )
                st.session_state.filters['status'] = selected_status
            
            # Date filters
            if 'CreatedDate' in leads_df.columns:
                date_col = leads_df['CreatedDate'].dropna()
                if not date_col.empty:
                    # Convert timezone-aware timestamps to dates for min/max
                    min_date = pd.to_datetime(date_col.min()).date()
                    max_date = pd.to_datetime(date_col.max()).date()
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        start_date = st.date_input(
                            "Start Date",
                            value=st.session_state.filters.get('start_date', min_date),
                            min_value=min_date,
                            max_value=max_date
                        )
                        st.session_state.filters['start_date'] = start_date
                    
                    with col2:
                        end_date = st.date_input(
                            "End Date",
                            value=st.session_state.filters.get('end_date', max_date),
                            min_value=min_date,
                            max_value=max_date
                        )
                        st.session_state.filters['end_date'] = end_date
            
            # Other filters
            filter_configs = [
                ('Project', 'Project__c', 'project'),
                ('Lead Source', 'LeadSource', 'lead_source'),
                ('Campaign', 'Campaign_Name__c', 'campaign')
            ]
            
            for label, column, key in filter_configs:
                if column in leads_df.columns:
                    options = leads_df[column].dropna().unique().tolist()
                    selected = st.selectbox(
                        f"Filter by {label}",
                        options=["All"] + options,
                        index=0
                    )
                    if selected != "All":
                        st.session_state.filters[key] = selected
                    elif key in st.session_state.filters:
                        del st.session_state.filters[key]
            
            # Clear filters button
            if st.button("🧹 Clear Filters", use_container_width=True):
                st.session_state.filters = {}
                st.rerun()
        
        st.divider()
        
        # Industry trends
        st.subheader("📊 Industry Trends")
        trends = st.session_state.industry_trends
        for key, value in trends.items():
            st.metric(key, value)
    
    # Main content
    leads_df = st.session_state.leads_df
    
    if leads_df.empty:
        st.info("📭 No data loaded. Click 'Refresh Data' to load leads data.")
        return
    
    # Apply filters
    filtered_df = apply_filters(leads_df, st.session_state.filters)
    
    # Display metrics
    if not filtered_df.empty:
        metrics = calculate_overall_funnel(filtered_df)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Leads", metrics.get('Total Leads', 0))
        with col2:
            st.metric("Valid Leads", metrics.get('Valid Leads', 0))
        with col3:
            st.metric("Qualified", metrics.get('Qualified Leads', 0))
        with col4:
            st.metric("Meetings", metrics.get('Meetings Booked', 0))
        
        # Conversion rate
        conv_rate = metrics.get('Conversion Rate', 0)
        st.metric("Conversion Rate", f"{conv_rate}%")
    
    st.divider()
    
    # Chat interface
    st.subheader("💬 Analytics Chat")
    st.caption("Ask about your leads, metrics, campaigns, or trends")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about leads, metrics, campaigns, or trends..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                response = get_ai_response(prompt, filtered_df)
                st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Quick action buttons
    st.subheader("🚀 Quick Actions")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("📊 Show Funnel", use_container_width=True):
            plot_funnel(filtered_df)
    
    with col2:
        if st.button("📈 Show Trends", use_container_width=True):
            plot_trends(filtered_df)
    
    with col3:
        if st.button("🎯 Lead Sources", use_container_width=True):
            plot_source_distribution(filtered_df)
    
    with col4:
        if st.button("📅 Financial Years", use_container_width=True):
            if st.session_state.financial_years:
                fy_metrics = []
                for fy, fy_df in st.session_state.financial_years.items():
                    metrics = calculate_overall_funnel(fy_df, fy)
                    fy_metrics.append(metrics)
                
                fy_comparison = pd.DataFrame(fy_metrics)
                if 'Financial Year' in fy_comparison.columns:
                    fy_comparison = fy_comparison.set_index('Financial Year')
                st.dataframe(fy_comparison, use_container_width=True)
    
    # ———————— New section: Show the LLM‐read‐file answer on demand ————————
    st.divider()
    st.subheader("🤖 Ask the LLM to Read the Entire Code File")
    question = "Which flat type—2 BHK or 3 BHK—is most preferred among Qualified leads?"
    if st.button("📚 Run LLM Analysis for Flat Preference"):
        with st.spinner("Reading file and querying LLM…"):
            llm_response = ask_llm_to_read_file(question)
            st.markdown(f"**LLM Answer:**\n\n{llm_response}")
    # ———————— End of LLM‐read‐file section ————————
    
    # Raw data view
    with st.expander("📋 View Raw Data"):
        st.dataframe(filtered_df, use_container_width=True)
        st.caption(f"Showing {len(filtered_df)} records")

if __name__ == "__main__":
    main()

