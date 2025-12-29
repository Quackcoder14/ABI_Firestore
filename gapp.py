# gapp.py - Complete Updated Version
import streamlit as st
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types 
import json
import time
import random 
from gtools import (
    get_customer_orders,
    execute_pandas_code_business,
    check_for_revenue_anomalies, 
    check_for_critical_delays,
    check_customer_order_status,
    FIREBASE_INIT_STATUS
)
from datetime import datetime

# --- Configuration and Initialization ---
load_dotenv()

# Try to get API key from Streamlit secrets first, then fall back to .env
try:
    if "GEMINI_API_KEY" in st.secrets:
        GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    else:
        GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
except:
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
REGISTERED_CUSTOMER_TOOLS = [get_customer_orders]
REGISTERED_BUSINESS_TOOLS_ALL = [execute_pandas_code_business]

# --- System Instructions ---
def get_customer_system_instruction():
    """Generate customer system instruction with current customer_id"""
    customer_id = st.session_state.get('customer_id', 'UNKNOWN')
    
    return f"""
You are the Customer Service Agent. Your persona is polite, concise, and focused on helping customers track their orders.

**IMPORTANT: The logged-in customer's ID is: {customer_id}**

You have access to the 'get_customer_orders' tool which shows all orders for the logged-in customer.

**CRITICAL INSTRUCTION FOR TOOL USAGE:**
- When the customer asks about their orders, ALWAYS call the tool with the exact customer_id: "{customer_id}"
- DO NOT ask the customer for their customer_id - you already know it
- DO NOT modify or concatenate the customer_id in any way
- Simply use: get_customer_orders(customer_id="{customer_id}")

The tool returns order details including:
- Order ID, Status, Order Date, Estimated Delivery
- Product Name, Category, Price
- Customer's own information

**INSTRUCTIONS:**
1. When a customer asks about their orders, immediately use 'get_customer_orders' with customer_id="{customer_id}"
2. Present the information in a friendly, easy-to-understand format.
3. If asked about specific order status, delays, or product details, extract from the tool's response.
4. NEVER discuss revenue, sales analytics, other customers' data, or business metrics.
5. If asked about business data, politely say: "I can only help with your order inquiries. For business analytics, please contact the business portal."

**Privacy Note:** You can only see the logged-in customer's own orders. This protects customer privacy.
"""

SYSTEM_INSTRUCTION_BUSINESS = """
You are the Autonomous Business Intelligence (ABI) Analyst. Your persona is professional, strategic, and highly analytical.

You have access to a powerful tool called 'execute_pandas_code_business' which allows you to run Python analysis on ALL company data across multiple tables with foreign key relationships.

**DATABASE SCHEMA:**
1. customers_df: customer_id (PK), name, email, region
2. orders_df: order_id (PK), customer_id (FK), product_id (FK), status, order_date, est_delivery
3. products_df: product_id (PK), name, category, price, stock_level
4. revenue_df: revenue_id (PK), order_id (FK), amount, date, payment_method

**DATABASE RELATIONSHIPS:**
- customers.customer_id ← orders.customer_id (one-to-many)
- products.product_id ← orders.product_id (one-to-many)
- orders.order_id ← revenue.order_id (one-to-one)

**CRITICAL: HANDLING DUPLICATE COLUMN NAMES**
Both 'customers_df' and 'products_df' have a column called 'name'. When joining these tables with orders_df, you MUST use the suffixes parameter to avoid conflicts.

**MANDATORY: YOU MUST USE THE TOOL FOR ALL DATA QUERIES**
For ANY question about data, orders, customers, products, revenue, delays, or analytics:
1. You MUST call the execute_pandas_code_business tool
2. Write Python Pandas code to answer the question
3. Never try to answer data questions without using the tool
4. The dataframes are pre-loaded as: customers_df, orders_df, products_df, revenue_df

**INSTRUCTIONS FOR USING THE TOOL:**
1. When asked ANY question about data, write Python Pandas code to answer it.
2. The dataframes are pre-loaded as: customers_df, orders_df, products_df, revenue_df.
3. Use pandas .merge() to combine tables using foreign keys.
4. ALWAYS use suffixes parameter when joining tables that share column names.
5. You MUST assign the final answer to a variable named result.
6. DO NOT use .to_string() method - just assign the DataFrame or Series directly to result.
7. After getting the tool result, interpret it and provide clear, business-focused insights to the user.

**COMMON QUERY PATTERNS:**

List all delayed orders:
```python
import pandas as pd
from datetime import datetime

today = pd.to_datetime(datetime.now().strftime('%Y-%m-%d'))
delayed = orders_df[
    (orders_df['est_delivery'] < today) & 
    (orders_df['status'] != 'Delivered') &
    (orders_df['status'] != 'Cancelled') &
    (pd.notna(orders_df['est_delivery']))
]
result = delayed
```

Most sold product (by order count):
```python
product_sales = orders_df.merge(products_df, on='product_id', suffixes=('_order', '_product'))
result = product_sales.groupby('name').size().sort_values(ascending=False).head(1)
```

Total revenue:
```python
result = revenue_df['amount'].sum()
```

Customer revenue analysis:
```python
customer_orders = orders_df.merge(customers_df, on='customer_id')
customer_revenue = customer_orders.merge(revenue_df, on='order_id')
result = customer_revenue.groupby('name')['amount'].sum().sort_values(ascending=False).head(5)
```

**Remember:** 
- ALWAYS use the tool for data queries - never guess or make up data
- 'name' exists in BOTH customers_df and products_df
- Use suffixes to distinguish between them after joining
- After the tool returns results, provide a natural language summary to the user
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
    
    /* Improved result box styling - smaller fonts */
    .element-container .stSuccess {{ 
        font-size: 0.75rem !important;
        line-height: 1.3 !important;
        padding: 0.6rem !important;
        border-radius: 8px !important;
        background: rgba(0, 255, 0, 0.1) !important;
        border-left: 3px solid #00ff00 !important;
    }}
    .element-container .stSuccess p {{ 
        font-size: 0.75rem !important;
        margin: 0 !important;
        font-family: 'Courier New', monospace !important;
        white-space: pre-wrap !important;
        word-break: break-word !important;
    }}
    .element-container .stError {{ 
        font-size: 0.75rem !important;
        line-height: 1.3 !important;
        padding: 0.6rem !important;
        border-radius: 8px !important;
        background: rgba(255, 0, 0, 0.1) !important;
        border-left: 3px solid #ff0000 !important;
    }}
    .element-container .stError p {{ 
        font-size: 0.75rem !important;
        margin: 0 !important;
        font-family: 'Courier New', monospace !important;
        white-space: pre-wrap !important;
        word-break: break-word !important;
    }}
    
    /* Thought process panel styling - much smaller fonts */
    div[data-testid="column"]:last-child .element-container h3 {{
        font-size: 0.9rem !important;
        margin-bottom: 0.5rem !important;
    }}
    div[data-testid="column"]:last-child .element-container {{
        font-size: 0.7rem !important;
    }}
    div[data-testid="column"]:last-child .stExpander {{
        font-size: 0.7rem !important;
        margin-bottom: 0.3rem !important;
    }}
    div[data-testid="column"]:last-child .stExpander summary {{
        font-size: 0.7rem !important;
        padding: 0.3rem 0.5rem !important;
    }}
    div[data-testid="column"]:last-child .stExpander div[role="button"] {{
        font-size: 0.7rem !important;
    }}
    div[data-testid="column"]:last-child .stSuccess {{
        font-size: 0.65rem !important;
        padding: 0.4rem !important;
        line-height: 1.2 !important;
    }}
    div[data-testid="column"]:last-child .stSuccess p {{
        font-size: 0.65rem !important;
        margin: 0.2rem 0 !important;
    }}
    div[data-testid="column"]:last-child .stSuccess strong {{
        font-size: 0.65rem !important;
        font-weight: 600 !important;
    }}
    div[data-testid="column"]:last-child .stError {{
        font-size: 0.65rem !important;
        padding: 0.4rem !important;
        line-height: 1.2 !important;
    }}
    div[data-testid="column"]:last-child .stError p {{
        font-size: 0.65rem !important;
        margin: 0.2rem 0 !important;
    }}
    div[data-testid="column"]:last-child pre {{
        font-size: 0.65rem !important;
        padding: 0.3rem !important;
        margin: 0.2rem 0 !important;
    }}
    div[data-testid="column"]:last-child code {{
        font-size: 0.65rem !important;
    }}
    
    /* Additional styling for customer result text */
    div[data-testid="column"]:last-child .stSuccess div {{
        font-size: 0.65rem !important;
    }}
    div[data-testid="column"]:last-child .stSuccess * {{
        font-size: 0.65rem !important;
        line-height: 1.3 !important;
    }}
    
    /* Welcome banner styling */
    .welcome-banner {{
        background: linear-gradient(135deg, rgba(179, 255, 0, 0.15) 0%, rgba(0, 255, 255, 0.15) 100%);
        border: 1px solid rgba(179, 255, 0, 0.3);
        border-radius: 12px;
        padding: 1rem 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 15px rgba(179, 255, 0, 0.2);
    }}
    .welcome-text {{
        font-size: 1.2rem;
        font-weight: 600;
        color: #B3FF00;
        margin: 0;
        text-shadow: 0 0 10px rgba(179, 255, 0, 0.5);
    }}
    .welcome-subtitle {{
        font-size: 0.9rem;
        color: rgba(255, 255, 255, 0.8);
        margin: 0.3rem 0 0 0;
    }}
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
if 'customer_id' not in st.session_state:
    st.session_state.customer_id = None
if 'notification_shown' not in st.session_state:
    st.session_state.notification_shown = False

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
        system_instruction = get_customer_system_instruction()
        # Create function declarations for customer tools
        tool_declarations = [
            types.FunctionDeclaration(
                name='get_customer_orders',
                description='Returns ALL order information for a specific customer',
                parameters={
                    'type': 'object',
                    'properties': {
                        'customer_id': {
                            'type': 'string',
                            'description': 'The customer\'s unique ID (e.g., CUST_001)'
                        }
                    },
                    'required': ['customer_id']
                }
            )
        ]
    else: 
        tools = REGISTERED_BUSINESS_TOOLS_ALL 
        history_key = "business_history"
        system_instruction = SYSTEM_INSTRUCTION_BUSINESS
        # Create function declarations for business tools
        tool_declarations = [
            types.FunctionDeclaration(
                name='execute_pandas_code_business',
                description='Executes Python code for business analytics with access to ALL tables',
                parameters={
                    'type': 'object',
                    'properties': {
                        'python_code': {
                            'type': 'string',
                            'description': 'Python code to execute. Must assign result to variable named "result"'
                        }
                    },
                    'required': ['python_code']
                }
            )
        ]
    
    # Build conversation history
    contents = [
        types.Content(
            role="model" if msg["role"] == "assistant" else msg["role"],
            parts=[types.Part.from_text(text=str(msg["content"]))]
        )
        for msg in st.session_state[history_key]
    ]
    contents.append(types.Content(role="user", parts=[types.Part.from_text(text=prompt)]))

    # Create tool with function declarations
    tool = types.Tool(function_declarations=tool_declarations)
    
    tool_config = types.ToolConfig(
        function_calling_config=types.FunctionCallingConfig(mode='AUTO')
    )

    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        tools=[tool],
        tool_config=tool_config,
        temperature=0
    )
    
    # Create function map for automatic function calling
    function_map = {}
    for func in tools:
        function_map[func.__name__] = func
    
    # Generate response with retry logic for rate limits and 503 errors
    max_retries = 3
    base_delay = 2
    
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model='gemini-2.0-flash-exp',
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
    
    # Handle function calling manually if needed
    current_tool_steps = []
    
    # Check if response has candidates and content
    if (hasattr(response, 'candidates') and 
        response.candidates and 
        len(response.candidates) > 0 and
        hasattr(response.candidates[0], 'content') and
        response.candidates[0].content and
        hasattr(response.candidates[0].content, 'parts') and
        response.candidates[0].content.parts):
        
        for part in response.candidates[0].content.parts:
            if hasattr(part, 'function_call') and part.function_call:
                func_call = part.function_call
                func_name = func_call.name
                func_args = dict(func_call.args)
                
                current_tool_steps.append({
                    "type": "Tool Call",
                    "name": func_name,
                    "args": func_args
                })
                
                # Execute the function
                if func_name in function_map:
                    try:
                        result = function_map[func_name](**func_args)
                        current_tool_steps.append({
                            "type": "Tool Output",
                            "output": result
                        })
                        
                        # Send the function response back to the model
                        function_response_content = types.Content(
                            role="user",
                            parts=[types.Part.from_function_response(
                                name=func_name,
                                response={'result': result}
                            )]
                        )
                        
                        contents.append(response.candidates[0].content)
                        contents.append(function_response_content)
                        
                        # Get final response
                        response = client.models.generate_content(
                            model='gemini-2.0-flash-exp',
                            contents=contents,
                            config=config
                        )
                    except Exception as e:
                        error_msg = f"Error executing {func_name}: {str(e)}"
                        current_tool_steps.append({
                            "type": "Tool Output",
                            "output": error_msg
                        })

    st.session_state.last_raw_response = str(response)
    
    # Extract text safely
    response_text = ""
    try:
        if hasattr(response, 'text') and response.text:
            response_text = response.text
        elif hasattr(response, 'candidates') and response.candidates:
            # Try to extract text from candidates
            for candidate in response.candidates:
                if hasattr(candidate, 'content') and candidate.content:
                    if hasattr(candidate.content, 'parts'):
                        for part in candidate.content.parts:
                            if hasattr(part, 'text') and part.text:
                                response_text += part.text
    except Exception as e:
        st.error(f"Error extracting response: {e}")
        response_text = "I encountered an error processing the response. Please try again."
    
    # If still no text, provide a default message
    if not response_text or response_text.strip() == "":
        response_text = "I apologize, but I couldn't generate a proper response. Please try rephrasing your question."
    
    return response_text, current_tool_steps

# --- Authentication Functions ---
def authenticate_user(username, password, expected_role):
    global CREDENTIALS
    CREDENTIALS = load_credentials()
    
    if username in CREDENTIALS:
        user_data = CREDENTIALS[username]
        if user_data["password"] == password:
            if user_data["role"] == expected_role:
                st.session_state.authenticated_user = username
                # Store customer_id for customer role
                if expected_role == "customer":
                    st.session_state.customer_id = user_data.get("customer_id")
                st.session_state.page = f"{expected_role}_chat"
                st.session_state.auth_role_pending = None 
                st.rerun()
            else:
                st.error("Authentication failed: Incorrect role for this user.")
        else:
            st.error("Authentication failed: Incorrect password.")
    else:
        st.error("Authentication failed: User ID not found.")

def create_new_user(new_username, new_password, role, customer_id=None):
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

        user_data = {
            "password": new_password,
            "role": role
        }
        
        # Add customer_id for customer accounts
        if role == "customer":
            if not customer_id:
                st.error("Customer ID is required for customer accounts.")
                return
            user_data["customer_id"] = customer_id.strip().upper()
        
        CREDENTIALS[new_username] = user_data
        
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
        
        st.
