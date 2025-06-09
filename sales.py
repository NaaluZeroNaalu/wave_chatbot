import streamlit as st
import requests



SALESFORCE_CLIENT_ID = "3MVG9wt4IL4O5wvKmuWykzw13DGFOnjtd2q0MhKTvjQRdylQtrxmuTnEq4i2_.s6sQSQ5YJMl.1n_ScCpSDSP"
SALESFORCE_CLIENT_SECRET = "B7143F5B5BEA70B22F037608F6FDCD818AFEFDC88CD1588FB0608720471E9369"
SALESFORCE_USERNAME = "impwatson@gadieltechnologies.com"
SALESFORCE_PASSWORD = "Wave@#123456"
SALESFORCE_TOKEN_URL = "https://login.salesforce.com/services/oauth2/token"
SALESFORCE_API_URL = "https://waveinfratech.my.salesforce.com/services/apexrest/createCallTranscriptFromJson/"

url = f"https://login.salesforce.com/services/oauth2/token?grant_type=password&client_id={SALESFORCE_CLIENT_ID}&client_secret={SALESFORCE_CLIENT_ID}&username={SALESFORCE_USERNAME}&password={SALESFORCE_PASSWORD}"

payload = {}
headers = {
  'Cookie': 'BrowserId=ovdSOALsEfCxO0mpfSO-iQ; CookieConsentPolicy=0:0; LSKey-c$CookieConsentPolicy=0:0'
}

response = requests.request("POST", url, headers=headers, data=payload)

st.write(response.text)

