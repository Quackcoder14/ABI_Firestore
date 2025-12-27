# gapp.py
import streamlit as st
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types 
import json
import time
import random 
from gtools import (
    get_order_status, 
    query_business_analytics, 
    check_for_revenue_anomalies, 
    check_for_critical_delays,
    FIREBASE_INIT_STATUS
)
from datetime import datetime

# --- Configuration and Initialization ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# --- Streamlit Setup ---
st.set_page_config(
    layout="wide", 
    page_title="ABI Agent - Intelligent Portal",
    page_icon="🤖",
    initial_sidebar_state="expanded"
)

# --- Load Credentials ---
CREDENTIALS_FILE = "credentials.json"

def load_credentials():
    try:
        if not os.path.exists(CREDENTIALS_FILE):
            with open(CREDENTIALS_FILE, "w") as f:
                json.dump({}, f)
        with open(CREDENTIALS_FILE, "r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        st.error("Error: 'credentials.json' is improperly formatted. Resetting to empty.")
        return {}
    except Exception as e:
        st.error(f"Error loading credentials: {e}")
        return {}

def save_credentials(data):
    try:
        with open(CREDENTIALS_FILE, "w") as f:
            json.dump(data, f, indent=4)
        return True
    except Exception as e:
        st.error(f"Error saving credentials: {e}")
        return False

CREDENTIALS = load_credentials()

# --- Tool Definitions ---
REGISTERED_CUSTOMER_TOOLS = [get_order_status]
REGISTERED_BUSINESS_TOOLS_ALL = [
    get_order_status, 
    query_business_analytics, 
    check_for_revenue_anomalies, 
    check_for_critical_delays
]

# --- System Instructions ---
SYSTEM_INSTRUCTION_CUSTOMER = """You are the Customer Service Agent. Your persona is polite, concise, and focused on logistics.
Your primary role is to check the delivery status for specific orders using the available tool.
If a customer asks about sales, revenue, or internal data, politely state that you only handle order inquiries.
"""

SYSTEM_INSTRUCTION_BUSINESS = """You are an expert Autonomous Business Intelligence (ABI) Analyst.

Your role is to:
1. Analyze business data from orders, revenue, and products.
2. Provide accurate, data-driven insights.

CRITICAL TOOL SELECTION RULES:
- If the user asks for the status of a SINGLE specific order (e.g., "status of ORD1001"), you MUST use the tool `get_order_status`. Do NOT use `query_business_analytics` for single order lookups.
- For aggregate data (revenue, trends, count of orders), use `query_business_analytics`.
- For anomalies, use `check_for_revenue_anomalies`.
- For delay reports, use `check_for_critical_delays`.

Key Guidelines:
- NEVER fabricate data.
- Always interpret the tool results and provide meaningful business insights, not just raw data dumps.
"""

# --- STAR FIELD & LAYOUT CSS ---
STAR_FIELD_CSS = """
/* The space theme */
.stApp {
    background-color: #000000;
    background-attachment: fixed;
    background-image: 
        radial-gradient(circle at 80% 30%, #ffffff 0.5px, transparent 1.5px), 
        radial-gradient(circle at 15% 65%, #ffffff 0.7px, transparent 1.7px),
        radial-gradient(circle at 20% 70%, #fffacd 0.4px, transparent 1.2px), 
        radial-gradient(circle at 90% 40%, #fffacd 0.6px, transparent 1.6px),
        radial-gradient(circle at 50% 10%, rgba(255, 255, 255, 0.7) 0.5px, transparent 1px),
        radial-gradient(circle at 30% 90%, rgba(255, 255, 255, 0.5) 0.3px, transparent 0.8px),
        radial-gradient(circle at 70% 80%, rgba(255, 255, 255, 0.4) 0.2px, transparent 0.6px),
        radial-gradient(circle at 40% 20%, rgba(255, 255, 255, 0.3) 0.2px, transparent 0.5px);
    background-size: 600px 600px, 750px 750px, 900px 900px, 1100px 1100px, 1300px 1300px, 1500px 1500px, 1700px 1700px, 1900px 1900px;
    background-position: 0 0, 100px 100px, 200px 200px, 300px 300px, 400px 400px, 500px 500px, 600px 600px, 700px 700px;
    animation: move-stars 300s linear infinite;
}
@keyframes move-stars {
    from { background-position: 0 0, 100px 100px, 200px 200px, 300px 300px, 400px 400px, 500px 500px, 600px 600px, 700px 700px; }
    to { background-position: 600px 600px, 850px 850px, 1100px 1100px, 1400px 1400px, 1700px 1700px, 2000px 2000px, 2300px 2300px, 2600px 2600px; }
}
"""

st.markdown(f"""
<style>
    {STAR_FIELD_CSS}
    .main .block-container {{ padding-top: 0rem !important; padding-bottom: 2rem !important; max-width: 100%; }}
    .main .block-container > div:first-child:empty, .main .block-container > div[data-testid]:first-child {{ display: none !important; }}
    h1, h2, h3, h4, .stMarkdown, .stText, .stCaption {{ color: #FFFFFF !important; }}
    .stApp > div {{ color: #FFFFFF !important; }}
    .stButton>button {{ transition: all 0.2s ease-in-out; border: 1px solid #444444; }}
    .stButton>button:hover {{ border-color: #B3FF00 !important; color: #B3FF00 !important; box-shadow: 0 0 15px rgba(179, 255, 0, 0.7), 0 0 10px rgba(179, 255, 0, 0.5) !important; transform: translateY(-2px); }}
    .stTextInput>div>div>input:focus, .stTextInput>div>div>textarea:focus {{ border-color: #B3FF00 !important; box-shadow: 0 0 8px #B3FF00 !important; }}
    section[data-testid="stSidebar"] {{ background: rgba(13, 13, 13, 0.98) !important; backdrop-filter: blur(10px); border-right: 2px solid rgba(179, 255, 0, 0.3); }}
    section[data-testid="stSidebar"] > div {{ background: transparent !important; padding: 2rem 1rem !important; }}
    .role-card-container {{ position: relative; height: 250px; width: 100%; border-radius: 20px; background: rgba(255, 255, 255, 0.05); border: 1px solid rgba(255, 255, 255, 0.1); backdrop-filter: blur(10px); transition: transform 0.3s ease, box-shadow 0.3s ease; overflow: hidden; box-shadow: 0 4px 30px rgba(0, 0, 0, 0.5); margin-bottom: 20px; display: flex; flex-direction: column; justify-content: center; align-items: center; text-align: center; color: white; cursor: pointer; }}
    .card-customer {{ border-bottom: 4px solid #00FFFF; }}
    .card-customer:hover {{ transform: translateY(-5px); box-shadow: 0 0 30px rgba(0, 255, 255, 0.7); border-color: #00FFFF; }}
    .card-business {{ border-bottom: 4px solid #FFD700; }}
    .card-business:hover {{ transform: translateY(-5px); box-shadow: 0 0 30px rgba(255, 215, 0, 0.7); border-color: #FFD700; }}
    .card-icon {{ font-size: 4rem; margin-bottom: 10px; text-shadow: 0 0 10px rgba(255, 255, 255, 0.5); }}
    .card-title {{ font-size: 1.5rem; font-weight: 700; margin-bottom: 5px; }}
    .card-desc {{ font-size: 0.9rem; opacity: 0.8; padding: 0 20px; }}
    .login-container {{ padding: 30px; border-radius: 10px; background: rgba(255, 255, 255, 0.05); border: 1px solid rgba(255, 255, 255, 0.1); backdrop-filter: blur(8px); max-width: 100%; margin: 0 auto; box-shadow: 0 0 20px rgba(0, 0, 0, 0.5); }}
    .tool-call-box {{ background: rgba(33, 150, 243, 0.1); border-left: 4px solid #2196F3; padding: 15px; margin: 10px 0; border-radius: 5px; }}
    .tool-output-box {{ background: rgba(76, 175, 80, 0.1); border-left: 4px solid #4CAF50; padding: 10px; margin: 10px 0; border-radius: 5px; }}
    .stExpander {{ background: rgba(255, 255, 255, 0.03) !important; border: 1px solid rgba(255, 255, 255, 0.1) !important; }}
</style>
""", unsafe_allow_html=True)

# --- Session State Initialization ---
if "page" not in st.session_state:
    st.session_state.page = "selector" 
if "customer_history" not in st.session_state:
    st.session_state.customer_history = [] 
if "business_history" not in st.session_state:
    st.session_state.business_history = [] 
if "audit_log" not in st.session_state:
    st.session_state.audit_log = [] 
if "last_raw_response" not in st.session_state:
    st.session_state.last_raw_response = None
if 'revenue_alert_status' not in st.session_state:
    st.session_state.revenue_alert_status = "Pending (Run Audit)"
if 'delay_alert_status' not in st.session_state:
    st.session_state.delay_alert_status = "Pending (Run Audit)"
if 'auth_role_pending' not in st.session_state:
    st.session_state.auth_role_pending = None
if 'authenticated_user' not in st.session_state:
    st.session_state.authenticated_user = None

# --- Gemini Client Caching ---
@st.cache_resource(show_spinner=False)
def initialize_gemini_client():
    """Initializes and caches the Gemini Client connection."""
    if not GEMINI_API_KEY:
        st.error("GEMINI_API_KEY not found. Please set it in your .env file.")
        st.stop()
    try:
        return genai.Client(api_key=GEMINI_API_KEY)
    except Exception as e:
        st.error(f"Error initializing Gemini Client: {e}")
        st.stop()

client = initialize_gemini_client()

# --- Core Logic: Conversational Loop ---
def handle_chat_interaction(prompt, role):
    """Handles chat interaction with tool execution tracking and proper error handling"""
    if role == "customer":
        tools = REGISTERED_CUSTOMER_TOOLS
        history_key = "customer_history"
        system_instruction = SYSTEM_INSTRUCTION_CUSTOMER
    else: 
        tools = REGISTERED_BUSINESS_TOOLS_ALL 
        history_key = "business_history"
        system_instruction = SYSTEM_INSTRUCTION_BUSINESS
    
    # Build conversation history
    contents = [
        types.Content(
            role="model" if msg["role"] == "assistant" else msg["role"],
            parts=[types.Part.from_text(text=str(msg["content"]))]
        )
        for msg in st.session_state[history_key]
    ]
    contents.append(types.Content(role="user", parts=[types.Part.from_text(text=prompt)]))

    # Configure tool usage
    tool_config = types.ToolConfig(
        function_calling_config=types.FunctionCallingConfig(mode='AUTO')
    )

    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        tools=tools,
        tool_config=tool_config,
        temperature=0.1
    )
    
    # Generate response with retry logic for rate limits and 503 errors
    max_retries = 3
    base_delay = 2
    
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model='gemini-2.5-flash', # Corrected model name from 2.5 to 1.5 to fix 503
                contents=contents,
                config=config
            )
            break  # Success, exit retry loop
            
        except Exception as e:
            error_str = str(e)
            
            # Check for Rate limit (429) or Overloaded (503)
            if any(code in error_str for code in ["429", "503", "RESOURCE_EXHAUSTED", "UNAVAILABLE"]):
                if attempt < max_retries - 1:
                    wait_time = (base_delay * (2 ** attempt)) + random.uniform(0, 1)
                    st.warning(f"⏳ Server busy. Retrying in {wait_time:.1f}s... (Attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                    continue
                else:
                    return f"❌ Error: Service overloaded after {max_retries} attempts. Please try again later.\n\nDetails: {error_str}", []
            
            # Check for quota
            elif "quota" in error_str.lower():
                return f"❌ Error: API quota exceeded. Please check your Gemini API plan.\n\nDetails: {error_str}", []
            
            # Other errors
            else:
                return f"❌ Error generating response: {error_str}", []

    st.session_state.last_raw_response = str(response)
    current_tool_steps = []

    # Extract tool execution history
    if hasattr(response, 'automatic_function_calling_history') and response.automatic_function_calling_history:
        for content in response.automatic_function_calling_history:
            for part in content.parts:
                if part.function_call:
                    current_tool_steps.append({
                        "type": "Tool Call",
                        "name": part.function_call.name,
                        "args": dict(part.function_call.args)
                    })
                if part.function_response:
                    resp_dict = part.function_response.response
                    output_val = resp_dict.get('content') if resp_dict and isinstance(resp_dict, dict) else str(resp_dict)
                    current_tool_steps.append({
                        "type": "Tool Output",
                        "output": output_val
                    })

    return response.text, current_tool_steps

# --- Authentication Functions ---
def authenticate_user(username, password, expected_role):
    global CREDENTIALS
    CREDENTIALS = load_credentials()
    
    if username in CREDENTIALS:
        user_data = CREDENTIALS[username]
        if user_data["password"] == password:
            if user_data["role"] == expected_role:
                st.session_state.authenticated_user = username
                st.session_state.page = f"{expected_role}_chat"
                st.session_state.auth_role_pending = None 
                st.rerun()
            else:
                st.error("Authentication failed: Incorrect role for this user.")
        else:
            st.error("Authentication failed: Incorrect password.")
    else:
        st.error("Authentication failed: User ID not found.")

def create_new_user(new_username, new_password, role):
    global CREDENTIALS
    CREDENTIALS = load_credentials()
    
    if not new_username or not new_password:
        st.warning("User ID and Password cannot be empty.")
        return

    if new_username in CREDENTIALS:
        st.error(f"Registration failed: User ID '{new_username}' already exists. Please choose a different name.")
    else:
        if len(new_password) < 6:
            st.error("Password must be at least 6 characters long.")
            return

        CREDENTIALS[new_username] = {
            "password": new_password,
            "role": role 
        }
        
        if save_credentials(CREDENTIALS):
            st.success(f"Account for '{new_username}' created successfully! You can now log in.")
        else:
            st.error("Failed to save new user account.")

# --- Page Rendering Functions ---
def render_auth_page():
    """Renders the authentication page"""
    st.sidebar.empty()
    
    role = st.session_state.auth_role_pending
    display_role = "Customer" if role == "customer" else "Business Owner"
    
    st.markdown(f"<h1 style='text-align: center; margin-top: 80px; margin-bottom: 30px; font-size: 1.8rem;'>🔐 {display_role} Portal Access</h1>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("<div class='login-container'>", unsafe_allow_html=True)
        
        st.markdown("<h3>Login</h3>", unsafe_allow_html=True)
        
        with st.form("login_form"):
            login_username = st.text_input("User ID", key="login_username")
            login_password = st.text_input("Password", type="password", key="login_password")
            login_button = st.form_submit_button("Login", use_container_width=True)

            if login_button:
                authenticate_user(login_username, login_password, role)
        
        st.markdown("---")
        st.markdown("<h3>New User Registration</h3>", unsafe_allow_html=True)

        with st.form("register_form"):
            register_username = st.text_input("New User ID", key="register_username")
            register_password = st.text_input("New Password (Min 6 chars)", type="password", key="register_password")
            register_button = st.form_submit_button("Create Account", use_container_width=True, type="secondary")

            if register_button:
                create_new_user(register_username, register_password, role)
            
        st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<div style='text-align: center; margin-top: 30px;'>", unsafe_allow_html=True)
    col_back1, col_back2, col_back3 = st.columns([1, 2, 1])
    with col_back2:
        if st.button("⬅️ Cancel and Go Back", key="auth_back", use_container_width=True):
            st.session_state.auth_role_pending = None
            st.session_state.page = "selector"
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

def render_selector_page():
    """Renders the role selection page"""
    st.sidebar.empty()
    st.markdown("<div style='margin-top: 50px;'>", unsafe_allow_html=True) 
    st.markdown("<h1 style='text-align: center; margin-bottom: 40px;'>🤖 Autonomous Business Intelligence Agent</h1>", unsafe_allow_html=True)
    
    col_left, col_cust, col_mid, col_biz, col_right = st.columns([1, 3, 0.5, 3, 1])
    
    with col_cust:
        st.markdown("""<div class="role-card-container card-customer"><div class="card-icon">📦</div><div class="card-title">Customer Portal</div><div class="card-desc">Track Orders • Check Delivery • Shipping Updates</div></div>""", unsafe_allow_html=True)
        if st.button("Enter Customer Portal", use_container_width=True):
            st.session_state.auth_role_pending = "customer"
            st.session_state.page = "auth"
            st.rerun()
    
    with col_biz:
        st.markdown("""<div class="role-card-container card-business"><div class="card-icon">📊</div><div class="card-title">Business Command</div><div class="card-desc">Revenue Analytics • Anomaly Detection • Sales Data</div></div>""", unsafe_allow_html=True)
        if st.button("Enter Business Command", use_container_width=True):
            st.session_state.auth_role_pending = "business"
            st.session_state.page = "auth"
            st.rerun()
    
    st.markdown("</div>", unsafe_allow_html=True)

def render_chat_page(role):
    """Renders the chat interface for customer or business users"""
    if st.session_state.authenticated_user is None:
        st.warning("Access Denied. Please login.")
        st.session_state.page = "selector"
        st.rerun()
        return

    # --- SIDEBAR CONTENT ---
    with st.sidebar:
        st.header("Session")
        st.info(f"**Logged in as:** {st.session_state.authenticated_user}")
        
        # Firebase status
        if "SUCCESS" in FIREBASE_INIT_STATUS:
            st.success("✓ Firebase Connected")
        else:
            st.error("✗ Firebase Disconnected")
        
        if st.button("🚪 Logout & Return Home", key="sidebar_logout", use_container_width=True): 
            st.session_state.authenticated_user = None 
            st.session_state.auth_role_pending = None
            st.session_state.customer_history = []
            st.session_state.business_history = []
            st.session_state.audit_log = []
            st.session_state.page = "selector"
            st.rerun()
        
        st.markdown("---")
        
        if role == "business":
            st.header("⚡ System Actions")
            if st.button("🚨 Run System Audit", key="run_audit_sidebar", use_container_width=True):
                with st.spinner("Running diagnostic scans on Revenue & Logistics..."):
                    st.session_state.revenue_alert_status = check_for_revenue_anomalies()
                    st.session_state.delay_alert_status = check_for_critical_delays()
                st.rerun()
            
            st.markdown("---")
            st.header("📊 Live Status")
            
            if "CRITICAL" in st.session_state.revenue_alert_status or "⚠️" in st.session_state.revenue_alert_status:
                st.error(f"💰 **Revenue:**\n{st.session_state.revenue_alert_status[:100]}...")
            else:
                st.success(f"💰 **Revenue:**\nNormal")
            
            if "ALERT" in st.session_state.delay_alert_status or "🚨" in st.session_state.delay_alert_status:
                st.error(f"🚚 **Logistics:**\n{st.session_state.delay_alert_status[:100]}...")
            else:
                st.success(f"🚚 **Logistics:**\nNormal")

    # --- MAIN CHAT AREA ---
    st.markdown("<div style='margin-top: 30px;'>", unsafe_allow_html=True)
    
    if role == "customer":
        st.title("💬 Customer Support")
        st.caption("How can I help you with your order today?")
        history_key = "customer_history"
    else:
        st.title("📈 Business Intelligence Hub")
        st.caption("Ask anything about your data. I can calculate trends, totals, and more.")
        history_key = "business_history"

    col_chat, col_glass = st.columns([0.65, 0.35], gap="large")

    with col_chat:
        # Display chat history
        for message in st.session_state[history_key]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat input
        if prompt := st.chat_input("Type your query here..."):
            with st.chat_message("user"):
                st.markdown(prompt)
            st.session_state[history_key].append({"role": "user", "content": prompt})

            with st.spinner("Thinking..."):
                final_text, tool_steps = handle_chat_interaction(prompt, role)
            
            st.session_state.audit_log = tool_steps
            st.session_state[history_key].append({"role": "assistant", "content": final_text})
            st.rerun()

    # --- GLASS BOX (Tool Execution Viewer) ---
    with col_glass:
        st.subheader("🔍 Agent Thought Process")
        
        if st.session_state.audit_log:
            for step in st.session_state.audit_log:
                if step["type"] == "Tool Call":
                    with st.expander(f"🛠️ Executing: {step['name']}", expanded=False):
                        st.markdown('<div class="tool-call-box">', unsafe_allow_html=True)
                        st.markdown(f"**Function:** `{step['name']}`")
                        st.json(step['args'])
                        st.markdown('</div>', unsafe_allow_html=True)
                
                elif step["type"] == "Tool Output":
                    # Fix: Handle None types safely
                    output_raw = step.get('output')
                    res = str(output_raw)
                    
                    # 1. Skip if None
                    if output_raw is None or res.strip().lower() == "none":
                        continue

                    st.markdown('<div class="tool-output-box">', unsafe_allow_html=True)
                    
                    # 2. Check for Delay or Errors to show RED box
                    if any(word in res.upper() for word in ["DELAY", "ERROR", "CRITICAL", "ALERT", "🚨"]):
                        st.error(f"**Status Update:** {res}")
                    else:
                        st.success(f"**Data Retrieved:** {res}")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown("<p style='opacity: 0.6; font-style: italic;'>No external tools required for this response.</p>", unsafe_allow_html=True)
        
        # Developer debug section
        with st.expander("🛠️ Developer Debug", expanded=False):
            if st.session_state.last_raw_response:
                st.code(st.session_state.last_raw_response[:1000], language="text")
            else:
                st.caption("No raw response available yet.")
                
    st.markdown("</div>", unsafe_allow_html=True)

# --- ROUTER ---
if st.session_state.page == "selector":
    render_selector_page()
elif st.session_state.page == "auth":
    render_auth_page()
elif st.session_state.page == "customer_chat":
    render_chat_page("customer")
elif st.session_state.page == "business_chat":
    render_chat_page("business")