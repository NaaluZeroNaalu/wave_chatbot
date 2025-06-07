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
from main import *
import streamlit as st
import streamlit.components.v1 as components



def GetAccesstoken():
    auth_url = "https://iam.cloud.ibm.com/identity/token"
    
    headers = { 
        "Content-Type": "application/x-www-form-urlencoded",
        "Accept": "application/json"
    }
    
    data = {
        "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
        "apikey": API_KEY
    }
    response = requests.post(auth_url, headers=headers, data=data)
    
    if response.status_code != 200:
        st.write(f"Failed to get access token: {response.text}")
        return None
    else:
        token_info = response.json()
        return token_info['access_token']
    
    
def generatePrompt(question):
    # st.write(st.session_state.all_reports_veridia)
    body = {
        "input": f"""

        chat History:
        {st.session_state.chathistory}

user:{question}

You are a helpful and conversational Site engineer AI assistant. Respond to the user's input in a natural, human-like manner.


You have access to the following summaries:
veridia:
{st.session_state.veridiasummaries}

wavecity:
{st.session_state.wavecitysummaries}

Eligo:
{st.session_state.eligosummaries}

Eiden:
{st.session_state.eidensummaries}

EWSLIG
{st.session_state.ewsligsummaries}

### Instructions:

- If the user’s question contains a project keyword **and** the corresponding summary is available, return a **normal conversational reply** using that data.
- If the user’s question contains a project keyword **but no relevant data is found**, respond that the data is not available and return `"type": "project"`.
- If there is **no project keyword** in the user's question, respond with a **normal, general reply**.

{{
   "type":"normal / project",
   "reply":"normal conversation / project Mapping name"
}}

Keyword to Project Mapping:

- "wave city club" → WAVE CITY CLUB @ PSP 14A  
- "ews_lig" → EWS_LIG Veridia PH04  
- "eligo" → GH-8 Phase-2 (ELIGO) Wave City  
- "eden" → GH-8 Phase-3 (EDEN) Wave City  
- "veridia" → Wave Oakwood, Wave City 


NOTE: give only json format only for JSON loads, Dont add This <|eom_id|> to json

""" ,
        "parameters": {
            "decoding_method": "greedy",
            "max_new_tokens": 8100,
            "min_new_tokens": 0,
            "stop_sequences": [";"],
            "repetition_penalty": 1.05,
            "temperature": 0.5
        },
        "model_id": MODEL_ID,
        "project_id": PROJECT_ID
    }
    
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {GetAccesstoken()}"
    }
    
    if not headers["Authorization"]:
        return "Error: No valid access token."
    
    response = requests.post(WATSONX_API_URL, headers=headers, json=body)
    
    if response.status_code != 200:
        st.write(f"Failed to generate prompt: {response.text}")
        return "Error generating prompt"
    # st.write(json_datas)
    return response.json()['results'][0]['generated_text'].strip()

def chat_to_string(chat):
    chat_lines = []
    for msg in chat:
        role = "User" if msg["role"] == "user" else "Assistant"
        content = str(msg["content"]).strip()
        chat_lines.append(f"{role}: {content}  ")
    return "\n".join(chat_lines)

# Get the string


if "messages" not in st.session_state:
    st.session_state.messages = []

if "show_date_picker" not in st.session_state:
    st.session_state.show_date_picker = False

if "all_reports_veridia" not in st.session_state:
    st.session_state.all_reports_veridia = None

if "veridiasummaries" not in st.session_state:
    st.session_state.veridiasummaries = None

if "wavecitysummaries" not in st.session_state:
    st.session_state.wavecitysummaries = None

if "eligosummaries" not in st.session_state:
    st.session_state.eligosummaries = None

if "eidensummaries" not in st.session_state:
    st.session_state.eidensummaries = None

if "ewsligsummaries" not in st.session_state:
    st.session_state.ewsligsummaries = None

if "chathistory" not in st.session_state:
    st.session_state.chathistory = None

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Convert to readable chat string


user_input = st.chat_input("Enter your question")

    
# st.write(st.session_state.messages)

if user_input:
    # Display user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)


    # Generate and parse the response
    
    answer = generatePrompt(user_input)

    # st.write(answer)
    try:
        parsed_answer = json.loads(answer)
        message_type = parsed_answer.get("type")
        reply_content = parsed_answer.get("reply", "")

        if message_type == "normal":
            st.session_state.messages.append({"role": "assistant", "content": reply_content})
            with st.chat_message("assistant"):
                st.markdown(reply_content)

        elif message_type == "project":
            
            # display_text = f"**Project executed:** {reply_content}"
            st.write(reply_content)
            display_text = ProcessFiles(reply_content)
            st.info("Now Choose a Date")
            st.session_state.show_date_picker = True

            # Choosedate()
            st.session_state.messages.append({"role": "assistant", "content": "now Choose a Date"})
            with st.chat_message("assistant"):
                st.markdown("now Choose a Date")

        else:
            # Fallback for unknown types
            fallback_text = f"Unknown response type: {message_type}"
            st.session_state.messages.append({"role": "assistant", "content": fallback_text})
            with st.chat_message("assistant"):
                st.markdown(fallback_text)
        st.session_state.chathistory = chat_to_string(st.session_state.messages)
        # st.write(chat_string)

    except json.JSONDecodeError:
        # Fallback if answer is not valid JSON
        st.session_state.messages.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(answer)
            
if st.session_state.show_date_picker:
    if st.session_state["ncr_df"] is not None:
        ncr_df = st.session_state["ncr_df"]
        closed_start = st.sidebar.date_input("Closed Start Date", ncr_df['Created Date (WET)'].min().date(), key="ncr_closed_start")
        closed_end = st.sidebar.date_input("Closed End Date", ncr_df['Expected Close Date (WET)'].max().date(), key="ncr_closed_end")
        open_end = st.sidebar.date_input("Open Until Date", ncr_df['Created Date (WET)'].max().date(), key="ncr_open_end")
    else:
        closed_start = st.sidebar.date_input("Closed Start Date", key="ncr_closed_start")
        closed_end = st.sidebar.date_input("Closed End Date", key="ncr_closed_end")
        open_end = st.sidebar.date_input("Open Until Date", key="ncr_open_end")
    if st.sidebar.button("Generate File"):
        GenerateFile("Wave Oakwood, Wave City", closed_start, closed_end, open_end)
        # st.write(st.session_state.summaries)
        # st.write(st.session_state.all_reports)
        st.session_state.messages.append({"role": "assistant", "content":"file generated Successfully"})

