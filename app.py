import streamlit as st
import pandas as pd
from matching_engine_v3 import ActivityMatcherV3
import time
from openai import OpenAI
import base64

# Use Streamlit secrets for API key
openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

def generate_description(activity_name, category, code):
    """Generate description for activities that don't have one using AI"""
    try:
        prompt = f"""You are a business licensing expert. Generate a clear, professional description (3-4 sentences, ~80 words) for this business activity:

Activity Name: {activity_name}
Category: {category}
Activity Code: {code}

### Description Guidelines:
1. Begin with a general statement explaining what this activity involves.  
2. Clearly describe what type of work, services, or operations are permitted under this activity.  
3. If relevant, mention examples of typical business functions or industries that use this license.  
4. Avoid marketing tone, buzzwords, or repetition.  
5. The description must sound official and suitable for use in Meydan Free Zone's business activity list.

### Example of desired tone:
"This activity involves the provision of consultancy services focused on corporate strategy, organizational development, and business performance improvement. It includes advising companies on restructuring, process optimization, and market entry strategies, but excludes legal or financial auditing activities."

Write in a professional tone suitable for a business licensing document. Be specific and accurate."""
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert in business licensing and activity descriptions for free zones in Dubai."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=150
        )
        return response.choices[0].message.content.strip()
    except:
        return "Description not available."

# --- Page Config ---
st.set_page_config(
    page_title="Meydan Business Activity Finder",
    page_icon=":office:",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Meydan Branding CSS ---
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #ffffff 60%, #e0f2ef 90%);
        font-family: 'Inter', sans-serif;
        color: #003366;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1, h2, h3, h4 {
        color: #004b8d;
        font-weight: 600;
    }
    .stButton button {
        background-color: #008060;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    .stButton button:hover {
        background-color: #00664d;
    }
    .footer-text {
        text-align: center;
        color: #666;
        padding: 1rem;
        font-size: 0.9rem;
    }
    .meydan-logo {
        position: absolute;
        top: 15px;
        right: 35px;
        width: 180px;
    }
</style>
""", unsafe_allow_html=True)


# --- Initialize Matcher with Caching ---
@st.cache_resource(show_spinner="Loading AI matching engine...")
def load_matcher():
    """Load and cache the matching engine"""
    return ActivityMatcherV3()

# Get cached matcher
try:
    matcher = load_matcher()
except Exception as e:
    st.error(f"Error loading matching engine: {str(e)}")
    st.error("Please check the logs in 'Manage app' for more details.")
    st.stop()

# --- Initialize Session State ---
if 'results' not in st.session_state:
    st.session_state.results = None

if 'show_detail' not in st.session_state:
    st.session_state.show_detail = None

# --- Load CSV ---
@st.cache_data
def load_activities_data():
    return pd.read_csv('enhanced_activities_manual.csv', encoding='latin-1')

df_full = load_activities_data()

def get_group_info(code):
    """Get group information for an activity code"""
    try:
        activity_row = df_full[df_full['Code'] == code]
        if activity_row.empty:
            return None, None, None
        group_code = activity_row.iloc[0].get('Group', '')
        if pd.isna(group_code) or group_code == '':
            return None, None, None
        group_row = df_full[df_full['Code'] == str(group_code)]
        if not group_row.empty:
            group_activity_name = group_row.iloc[0]['Activity Name ']
            group_num = str(group_code).split('.')[0] if '.' in str(group_code) else str(group_code)
            return group_num, group_code, group_activity_name
        return None, None, None
    except:
        return None, None, None

def display_activity_card(activity, rank):
    """Display each activity card"""
    with st.container():
        col1, col2 = st.columns([1, 20])
        with col1:
            st.markdown(f"### #{rank}")
        with col2:
            st.markdown(f"## {activity['activity_name']}")
        st.markdown(f"**Activity Code:** `{activity['code']}`")

        col1, col2 = st.columns(2)
        with col1:
            third_party = activity.get('third_party', '')
            when = activity.get('when', '')
            if third_party and str(third_party) != 'nan' and third_party != '-':
                when_text = f" - {when}" if when and str(when) != 'nan' and when != 'N/A' else ""
                st.warning(f"⚠️ **Third Party Approval:** {third_party}{when_text}")
            else:
                st.success("✅ **No third party approval required**")

        with col2:
            risk = activity['risk_rating']
            if risk == 'Low':
                st.success(f"**Risk Rating:** {risk}")
            elif risk == 'Medium':
                st.warning(f"**Risk Rating:** {risk}")
            else:
                st.error(f"**Risk Rating:** {risk}")

        st.divider()

def display_detail_view(activity):
    """Show detailed info for selected activity"""
    st.markdown(f"# 📋 {activity['activity_name']}")
    st.markdown(f"### Activity Code: `{activity['code']}`")
    st.divider()

    # Description
    st.subheader("Description")
    description = activity.get('description', '')
    if description and str(description).lower() not in ['nan', '-', 'none', '']:
        st.write(description)
    else:
        with st.spinner("✨ Generating description..."):
            generated_desc = generate_description(
                activity.get('activity_name', ''),
                activity.get('category', ''),
                activity.get('code', '')
            )
            st.info(generated_desc)
            st.caption("ℹ️ AI-generated description")

    st.divider()

    # Category & Risk
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Category")
        st.info(activity['category'])
    with col2:
        st.subheader("Risk Rating")
        risk = activity['risk_rating']
        if risk == 'Low':
            st.success(risk)
        elif risk == 'Medium':
            st.warning(risk)
        else:
            st.error(risk)

    st.divider()

    # Third Party
    st.subheader("Third Party Approval")
    third_party = activity.get('third_party', '')
    when = activity.get('when', '')
    if third_party and str(third_party) != 'nan' and third_party != '-':
        when_text = f" - **{when}**" if when and str(when) != 'nan' and when != 'N/A' else ""
        st.warning(f"**Required:** {third_party}{when_text}")
    else:
        st.success("✅ No third party approval required")

    st.divider()

    # Keywords
    if activity.get('keywords') and str(activity['keywords']) != 'nan' and activity['keywords'] != '-':
        st.subheader("Keywords")
        st.write(activity['keywords'])
        st.divider()

    # Related Activities
    if activity.get('related_activities') and str(activity['related_activities']) != 'nan' and activity['related_activities'] != '-':
        st.subheader("Related Activities")
        st.write(activity['related_activities'])

# --- Header ---
st.title("Business Activity Finder")
st.divider()

# --- Query Input ---
query = st.text_input(
    "Describe your business activity:",
    placeholder="E.g., I want to sell mobile phones online and in retail stores...",
    key="query_input"
)
search_button = st.button("Find Activities", type="primary")

# --- Search logic ---
if search_button and query:
    with st.spinner("🔍 Searching for matching activities..."):
        result = matcher.search(query, top_k=5)
        st.session_state.results = result
        st.session_state.show_detail = None
        time.sleep(0.3)

# --- Results Display ---
if st.session_state.results:
    result = st.session_state.results
    confidence = result['confidence']
    st.divider()
    if confidence == 'high':
        st.success(f"✅ **Confidence: {confidence.upper()}**")
    elif confidence == 'medium':
        st.warning(f"⚠️ **Confidence: {confidence.upper()}**")
    else:
        st.error(f"🔴 **Confidence: {confidence.upper()}**")

    st.header("Top 5 Matching Activities")
    st.divider()

    for activity in result['results']:
        display_activity_card(activity, activity['rank'])

    st.divider()
    st.subheader("Need more details?")
    col1, col2 = st.columns([4, 1])
    with col1:
        selection = st.selectbox(
            "Select an activity to view full details:",
            options=[0] + list(range(1, len(result['results']) + 1)),
            format_func=lambda x: "Choose an activity..." if x == 0 else f"Activity #{x}: {result['results'][x-1]['activity_name']}",
            key="activity_selector"
        )
    with col2:
        show_detail_button = st.button("Show Details", use_container_width=True)

    if show_detail_button and selection > 0:
        st.session_state.show_detail = selection
        st.rerun()

    if st.session_state.show_detail:
        st.divider()
        selected_activity = result['results'][st.session_state.show_detail - 1]
        display_detail_view(selected_activity)
        st.divider()
        if st.button("⬅️ Back to Results"):
            st.session_state.show_detail = None
            st.rerun()

# --- Footer ---
st.markdown("""
<div class="footer-text">
    <p>© Meydan Free Zone – Dubai to the World</p>
</div>
""", unsafe_allow_html=True)