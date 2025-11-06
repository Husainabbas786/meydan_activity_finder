# app_v4.py

import streamlit as st
import pandas as pd
from matching_engine_v4 import ActivityMatcherV4 
import time
from openai import OpenAI
import os
import json
from datetime import datetime
import openpyxl  # ADD THIS
from openpyxl import Workbook

# --- Credential Handling ---
try:
    openai_client = OpenAI(api_key=st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY")))
except:
    pass # Error message will appear if connection fails later

# --- Helper Functions (Stubs - copy full logic from your original app.py if needed) ---
def generate_description(activity_name, category, code):
    """Generate AI description for activity"""
    try:
        prompt = f"""You are a Meydan Free Zone licensing expert in Dubai, UAE.

Generate a professional description for this business activity license:

Activity: {activity_name}
Code: {code}
Category: {category}

Write 2-3 sentences describing:
1. What business operations are permitted
2. What products/services can be offered
3. Any key characteristics of this activity

Be specific, professional, and actionable. No generic phrases."""

        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=150
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Description generation failed: {str(e)}"

def display_detail_view(activity):
    # Stub: Replace with your actual detail view display logic
    st.subheader(f"Full Details for {activity['activity_name']}")
    st.markdown(f"**Description:** {activity.get('description', 'Not Available')}")
    st.markdown(f"**Third Party Approval:** {activity.get('third_party', 'None')}")
    st.markdown(f"**Risk Rating:** **{activity.get('risk_rating', 'N/A')}**")
    st.markdown("---")

def save_feedback(query, activity_code, activity_name, feedback_type):
    """Save user feedback to CSV file"""
    feedback_file = "feedback_log.csv"
    
    # Prepare feedback data
    feedback_data = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'query': query,
        'activity_code': activity_code,
        'activity_name': activity_name,
        'feedback': feedback_type  # 'positive' or 'negative'
    }
    
    # Check if file exists
    file_exists = os.path.isfile(feedback_file)
    
    # Append to CSV
    df = pd.DataFrame([feedback_data])
    df.to_csv(feedback_file, mode='a', header=not file_exists, index=False)

def log_query_for_accuracy_evaluation(query, result):
    """Log query and all displayed activities to Excel for accuracy evaluation"""
    log_file = "query_accuracy_log.xlsx"
    
    # Get all displayed activities (initial + additional if viewed)
    all_activities = result.get('results_initial', [])
    if st.session_state.get('show_additional', False):
        all_activities += result.get('results_additional', [])
    
    # Prepare row data
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    from_cache = result.get('from_cache', False)
    cache_similarity = result.get('cache_similarity', 0.0)
    
    cache_status = f"Yes ({cache_similarity:.0%})" if from_cache else "No"
    
    # Build activity columns (up to 15 activities max for reasonable Excel width)
    row_data = {
        'Timestamp': timestamp,
        'Query': query,
        'From Cache': cache_status,
        'Total Activities': len(all_activities)
    }
    
    # Add each activity's data (code, name, scores)
    for idx, activity in enumerate(all_activities[:15], 1):  # Max 15 activities
        code = activity.get('code', 'N/A')
        name = activity.get('activity_name', 'N/A')
        hybrid = activity.get('final_score', 0)
        expert = activity.get('llm_match_score', 0)
        final = activity.get('ULTIMATE_SCORE', 0)
        
        row_data[f'Activity_{idx}'] = f"{code} - {name}"
        row_data[f'Hybrid_{idx}'] = f"{hybrid:.1f}%"
        row_data[f'Expert_{idx}'] = f"{expert:.1f}%"
        row_data[f'Final_{idx}'] = f"{final:.1f}%"
    
    # Convert to DataFrame
    df_new = pd.DataFrame([row_data])
    
    # Append to existing file or create new
    try:
        if os.path.exists(log_file):
            # Read existing data
            df_existing = pd.read_excel(log_file)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df_combined = df_new
        
        # Save to Excel
        df_combined.to_excel(log_file, index=False)
        print(f"📊 Logged query to {log_file}")
    except Exception as e:
        print(f"❌ Failed to log query: {e}")


# --- Streamlit Setup ---
st.set_page_config(layout="wide", page_title="Meydan Activity Finder V4")

@st.cache_resource
def load_matcher():
    """Load the new V4 matcher class."""
    return ActivityMatcherV4()

try:
    matcher = load_matcher()
except Exception as e:
    st.error(f"Failed to initialize Activity Matcher V4. Check API keys and file paths: {e}")
    st.stop()

# --- ADMIN: Download Logs ---
with st.sidebar:
    st.markdown("###  Admin Panel")
    
    log_file = "query_accuracy_log.xlsx"
    if os.path.exists(log_file):
        with open(log_file, "rb") as file:
            st.download_button(
                label=" Download Query Log",
                data=file,
                file_name=f"query_log_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        
        # Show stats
        df_log = pd.read_excel(log_file)
        st.metric("Total Queries Logged", len(df_log))
    else:
        st.info("No logs yet")

st.title(" Meydan Activity Finder")
st.markdown("---")
# --- Main App Logic ---


if 'show_detail' not in st.session_state:
    st.session_state.show_detail = None

if 'show_additional' not in st.session_state:
    st.session_state.show_additional = False

col_query, col_button = st.columns([5, 1])

with col_query:
    query = st.text_area(
        "Enter the Customer's Business Description:",
        height=100
    )
with col_button:
    st.markdown("<br>", unsafe_allow_html=True) 
    search_button = st.button("Find Activities", type="primary", use_container_width=True)

# --- Display Logic ---
def display_activity_card(activity, rank):
    
    # Activity Name Header
    st.markdown(f"### {rank}: {activity.get('activity_name', 'N/A')}")
    
    # --- GET ALL VALUES ---
    code = activity.get('code', 'N/A')
    category = activity.get('category', 'N/A')
    risk = activity.get('risk_rating', 'N/A')
    
    # Risk color coding
    if risk.lower() == 'low':
        risk_color = '#E8F5E9'
        risk_text_color = '#2E7D32'
    elif risk.lower() == 'medium':
        risk_color = '#FFF3E0'
        risk_text_color = '#E65100'
    elif risk.lower() == 'high':
        risk_color = '#FFEBEE'
        risk_text_color = '#C62828'
    else:
        risk_color = '#F5F5F5'
        risk_text_color = '#616161'
    
    # --- PREAPPROVAL PROCESSING ---
    when_value = activity.get('when', 'N/A')
    when_value_str = str(when_value).strip().lower()
    
    if when_value_str in ['', 'n/a', '-', 'nan', 'none']:
        preapproval_text = "Approval: Not Required"
        preapproval_color = '#E8F5E9'
        preapproval_text_color = '#2E7D32'
    elif when_value_str == 'post':
        preapproval_text = "Approval: Postapproval"
        preapproval_color = '#FFFDE7'
        preapproval_text_color = '#F57F17'
    elif when_value_str == 'pre':
        preapproval_text = "Approval: Preapproval"
        preapproval_color = '#FFFDE7'
        preapproval_text_color = '#F57F17'
    else:
        preapproval_text = f"Approval: {str(when_value).strip()}"
        preapproval_color = '#FFFDE7'
        preapproval_text_color = '#F57F17'
    
    # --- THIRD PARTY PROCESSING ---
    third_party = activity.get('third_party', '')
    third_party_str = str(third_party).strip().lower()
    
    if third_party_str in ['', 'n/a', '-', 'nan', 'none']:
        third_party_text = "3rd Party: Not Required"
        third_party_color = '#E8F5E9'
        third_party_text_color = '#2E7D32'
    else:
        third_party_text = f"3rd Party: {str(third_party).strip()}"
        third_party_color = '#E0F7FA'
        third_party_text_color = '#00838F'
    
    # --- GET SCORES ---
    hybrid_score = activity.get('final_score', 0)
    llm_score = activity.get('llm_match_score', 0)
    ultimate_score = activity.get('ULTIMATE_SCORE', 0)
    
    # --- ALL BADGES IN ONE ROW (8 badges total) ---
    st.markdown(f"""
    <div style="display: flex; gap: 8px; margin-bottom: 15px; flex-wrap: wrap; align-items: center;">
        <span style="background-color: #E3F2FD; color: #1565C0; padding: 6px 12px; border-radius: 12px; font-size: 13px; font-weight: 500; white-space: nowrap;">
            Code: {code}
        </span>
        <span style="background-color: #F3E5F5; color: #6A1B9A; padding: 6px 12px; border-radius: 12px; font-size: 13px; font-weight: 500; white-space: nowrap;">
            Cat: {category}
        </span>
        <span style="background-color: {risk_color}; color: {risk_text_color}; padding: 6px 12px; border-radius: 12px; font-size: 13px; font-weight: 500; white-space: nowrap;">
            Risk: {risk}
        </span>
        <span style="background-color: {preapproval_color}; color: {preapproval_text_color}; padding: 6px 14px; border-radius: 12px; font-size: 13px; font-weight: 500; white-space: nowrap; border-left: 3px solid {preapproval_text_color};">
            {preapproval_text}
        </span>
        <span style="background-color: {third_party_color}; color: {third_party_text_color}; padding: 6px 14px; border-radius: 12px; font-size: 13px; font-weight: 500; white-space: nowrap; border-left: 3px solid {third_party_text_color};">
            {third_party_text}
        </span>
        <span style="background-color: #F5F5F5; color: #616161; padding: 6px 12px; border-radius: 12px; font-size: 13px; font-weight: 500; white-space: nowrap;">
            Hybrid: {hybrid_score:.1f}%
        </span>
        <span style="background-color: #F5F5F5; color: #616161; padding: 6px 12px; border-radius: 12px; font-size: 13px; font-weight: 500; white-space: nowrap;">
            Expert: {llm_score:.1f}%
        </span>
        <span style="background-color: #F5F5F5; color: #616161; padding: 6px 12px; border-radius: 12px; font-size: 13px; font-weight: 500; white-space: nowrap;">
            Final: {ultimate_score:.1f}%
        </span>
    </div>
    """, unsafe_allow_html=True)
    
    # Removed Helpful / Not Helpful buttons per request.
    st.markdown("<br>", unsafe_allow_html=True)
    # --- DESCRIPTION SECTION ---
    with st.expander("📄 View Description"):
        description = activity.get('description', '')
        
        # Check if description exists and is not empty/placeholder
        if description and description not in ['', '-', 'N/A', 'nan', 'None']:
            st.markdown(f"**Description:** {description}")
        else:
            # Generate AI description
            with st.spinner("Generating description..."):
                ai_desc = generate_description(
                    activity.get('activity_name', 'N/A'),
                    activity.get('category', 'N/A'),
                    code
                )
                st.markdown(f"**AI-Generated Description:** {ai_desc}")
                st.caption("💡 This description was generated by AI as no official description is available.")
    
    st.markdown("---")
    
    
def display_results(result):
    # Check if we have ANY qualified results
    total_qualified = result.get('total_qualified', 0)
    
    if total_qualified == 0:
        st.error(" **No activities meet the quality threshold (LLM Score >= 60%)**")
        st.info(" **Suggestion:** Please refine your query with more specific details about your business.")
        return
    
    # Display initial results
    results_initial = result.get('results_initial', [])

    for activity in results_initial:
        display_activity_card(activity, activity['rank'])

    # "View More" section if there are additional results
    results_additional = result.get('results_additional', [])
    if results_additional:
        st.divider()
        remaining_count = len(results_additional)
        
        # Initialize session state for "View More" toggle
        if 'show_additional' not in st.session_state:
            st.session_state.show_additional = False
        
        # Toggle button
        if st.session_state.show_additional:
            if st.button(f" Hide {remaining_count} Additional Activities", use_container_width=True):
                st.session_state.show_additional = False
                st.rerun()
        else:
            if st.button(f" View {remaining_count} More Activities (LLM Score ≥ 60%)", type="primary", use_container_width=True):
                st.session_state.show_additional = True
                st.rerun()
        
        # Display additional results if toggled on
        if st.session_state.show_additional:
            st.divider()
            st.subheader("Additional Matching Activities")
            for activity in results_additional:
                display_activity_card(activity, activity['rank'])

    # Detail view (if user clicked "Show Details" on any activity)
    if st.session_state.show_detail:
        st.divider()
        display_detail_view(st.session_state.show_detail)


# Search button logic (outside the display_results function)
if search_button and query:
    st.session_state.show_detail = None
    st.session_state.current_query = query
    with st.spinner("Running Expert Hybrid Search V4 (LLM Reranking in progress)..."):
        start_time = time.time()
        try:
            result = matcher.search(query, top_k=10)
            end_time = time.time()
            st.success(f"Search complete in {end_time - start_time:.2f} seconds.")
            st.session_state.search_results = result
            
            # LOG QUERY FOR ACCURACY EVALUATION
            log_query_for_accuracy_evaluation(query, result)  # ADD THIS LINE
            
        except Exception as e:
            st.error(f"An error occurred during search: {e}. Check console for traceback.")

# Display results from session state (persists across reruns)
if 'search_results' in st.session_state:
    display_results(st.session_state.search_results)
