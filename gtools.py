# gtools.py - FIXED VERSION with robust column handling
import firebase_admin
from firebase_admin import credentials, firestore
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
from sklearn.ensemble import IsolationForest

# --- Configuration ---
FIREBASE_CREDS_FILE = 'firebase_creds.json'

# --- Initialization ---
db = None
FIREBASE_INIT_STATUS = ""

# Try to initialize Firebase
try:
    if not firebase_admin._apps:
        cred = None
        
        # Try Streamlit secrets first (for cloud deployment)
        try:
            import streamlit as st
            if hasattr(st, 'secrets') and "firebase" in st.secrets:
                # Use Streamlit secrets (for deployed app)
                firebase_config = dict(st.secrets["firebase"])
                cred = credentials.Certificate(firebase_config)
                firebase_admin.initialize_app(cred)
                db = firestore.client()
                FIREBASE_INIT_STATUS = "SUCCESS (Streamlit Cloud)"
            else:
                raise KeyError("No Streamlit secrets found")
        except (ImportError, KeyError, AttributeError) as e:
            # Fall back to local credentials file
            if os.path.exists(FIREBASE_CREDS_FILE):
                cred = credentials.Certificate(FIREBASE_CREDS_FILE)
                firebase_admin.initialize_app(cred)
                db = firestore.client()
                FIREBASE_INIT_STATUS = "SUCCESS (Local File)"
            else:
                FIREBASE_INIT_STATUS = f"ERROR: Firebase credentials not found. Checked Streamlit secrets and local file ({FIREBASE_CREDS_FILE})"
                db = None
    else:
        db = firestore.client()
        FIREBASE_INIT_STATUS = "SUCCESS (Already Initialized)"
        
except Exception as e:
    FIREBASE_INIT_STATUS = f"ERROR: Firebase Initialization Failed. Details: {str(e)}"
    db = None
    
# Debug logging (will show in Streamlit Cloud logs)
if db is None:
    print(f"⚠️ Firebase initialization failed: {FIREBASE_INIT_STATUS}")
else:
    print(f"✅ Firebase initialized: {FIREBASE_INIT_STATUS}")


# --- Data Loading Helper Functions ---

def get_firestore_collection(collection_name: str) -> tuple[pd.DataFrame, str]:
    """Retrieves all data from a specific Firestore collection."""
    if db is None:
        return pd.DataFrame(), f"ERROR: Firebase not initialized. Status: {FIREBASE_INIT_STATUS}"
    
    try:
        docs = db.collection(collection_name).stream()
        data = [doc.to_dict() for doc in docs]
        df = pd.DataFrame(data)
        
        if df.empty:
            return df, f"ERROR: Collection '{collection_name}' is empty or does not exist."
        
        # Debug: Print actual column names
        print(f"✅ Loaded {collection_name}: {len(df)} rows, columns: {list(df.columns)}")
        
        return df, "SUCCESS"
        
    except Exception as e:
        return pd.DataFrame(), str(f"ERROR: Failed to retrieve collection '{collection_name}'. Details: {e}")


def load_data():
    """
    Loads all dataframes from Firestore with proper cleaning and type conversion.
    Returns: dict with dataframes or error message string
    """
    customers_df, customers_status = get_firestore_collection('customers')
    orders_df, orders_status = get_firestore_collection('orders')
    products_df, products_status = get_firestore_collection('products')
    revenue_df, revenue_status = get_firestore_collection('revenue')

    # Check for errors
    errors = []
    if 'ERROR' in customers_status:
        errors.append(f"Customers: {customers_status}")
    if 'ERROR' in orders_status:
        errors.append(f"Orders: {orders_status}")
    if 'ERROR' in products_status:
        errors.append(f"Products: {products_status}")
    if 'ERROR' in revenue_status:
        errors.append(f"Revenue: {revenue_status}")
    
    if errors:
        return f"Data Load Error: {'; '.join(errors)}"

    # --- Robust Date Converter (Timezone Naive) ---
    def to_datetime_clean(series):
        if series.empty: return series
        series = series.apply(lambda x: x.to_datetime() if hasattr(x, 'to_datetime') else x)
        series = pd.to_datetime(series, errors='coerce')
        if series.dt.tz is not None:
            series = series.dt.tz_localize(None)
        return series
    
    # --- Normalize Column Names (handle case sensitivity) ---
    def normalize_columns(df):
        """Normalize column names to lowercase with underscores"""
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
        return df
    
    # Normalize all dataframes
    if not customers_df.empty:
        customers_df = normalize_columns(customers_df)
    if not orders_df.empty:
        orders_df = normalize_columns(orders_df)
    if not products_df.empty:
        products_df = normalize_columns(products_df)
    if not revenue_df.empty:
        revenue_df = normalize_columns(revenue_df)
    
    # --- Process Customers ---
    if not customers_df.empty:
        if 'customer_id' in customers_df.columns:
            customers_df['customer_id'] = customers_df['customer_id'].astype(str).str.strip().str.upper()
    
    # --- Process Orders ---
    if not orders_df.empty:
        if 'order_id' in orders_df.columns:
            orders_df['order_id'] = orders_df['order_id'].astype(str)
        if 'customer_id' in orders_df.columns:
            orders_df['customer_id'] = orders_df['customer_id'].astype(str).str.strip().str.upper()
        if 'product_id' in orders_df.columns:
            orders_df['product_id'] = orders_df['product_id'].astype(str)
        
        # Date conversions - handle both possible column names
        for col in ['order_date', 'orderdate']:
            if col in orders_df.columns:
                orders_df['order_date'] = to_datetime_clean(orders_df[col])
                break
        
        for col in ['est_delivery', 'estdelivery', 'estimated_delivery']:
            if col in orders_df.columns:
                orders_df['est_delivery'] = to_datetime_clean(orders_df[col])
                break
    
    # --- Process Products ---
    if not products_df.empty:
        if 'product_id' in products_df.columns:
            products_df['product_id'] = products_df['product_id'].astype(str)
        if 'price' in products_df.columns:
            products_df['price'] = pd.to_numeric(products_df['price'], errors='coerce')
        if 'stock_level' in products_df.columns:
            products_df['stock_level'] = pd.to_numeric(products_df['stock_level'], errors='coerce')
    
    # --- Process Revenue ---
    if not revenue_df.empty:
        if 'order_id' in revenue_df.columns:
            revenue_df['order_id'] = revenue_df['order_id'].astype(str)
        if 'date' in revenue_df.columns:
            revenue_df['date'] = to_datetime_clean(revenue_df['date'])
        if 'amount' in revenue_df.columns:
            revenue_df['amount'] = pd.to_numeric(revenue_df['amount'], errors='coerce')
    
    return {
        "customers_df": customers_df,
        "orders_df": orders_df,
        "products_df": products_df,
        "revenue_df": revenue_df
    }


def format_dataframe_output(df, max_rows=10):
    """Formats a pandas DataFrame into a clean, readable string format."""
    if isinstance(df, pd.Series):
        df = df.to_frame()
    
    if not isinstance(df, pd.DataFrame):
        return str(df)
    
    total_rows = len(df)
    display_df = df.head(max_rows)
    
    output = f"📊 Total Records: {total_rows}\n"
    if total_rows > max_rows:
        output += f"📋 Showing first {max_rows} rows\n"
    output += "─" * 70 + "\n\n"
    
    for idx, row in display_df.iterrows():
        output += f"▸ Record {idx + 1 if isinstance(idx, int) else idx}:\n"
        for col in display_df.columns:
            value = row[col]
            if pd.api.types.is_datetime64_any_dtype(display_df[col]):
                value = pd.to_datetime(value).strftime('%Y-%m-%d') if pd.notna(value) else 'N/A'
            elif isinstance(value, float):
                value = f"${value:,.2f}" if 'amount' in col.lower() or 'price' in col.lower() or 'revenue' in col.lower() else f"{value:.2f}"
            elif pd.isna(value):
                value = 'N/A'
            
            output += f"  • {col}: {value}\n"
        output += "\n"
    
    if total_rows > max_rows:
        output += f"⋯ and {total_rows - max_rows} more records\n"
    
    return output


def format_series_output(series, name="Value"):
    """Formats a pandas Series into a clean, readable format."""
    output = f"📊 Analysis Results ({len(series)} items)\n"
    output += "─" * 70 + "\n\n"
    
    for idx, value in series.items():
        if isinstance(value, float):
            formatted_value = f"${value:,.2f}" if 'amount' in name.lower() or 'revenue' in name.lower() else f"{value:,.2f}"
        else:
            formatted_value = str(value)
        
        output += f"▸ {idx}: {formatted_value}\n"
    
    return output


def format_scalar_output(value, description="Result"):
    """Formats a scalar value nicely."""
    output = f"📊 {description}\n"
    output += "─" * 70 + "\n\n"
    
    if isinstance(value, (int, float)):
        if isinstance(value, float):
            formatted = f"${value:,.2f}"
        else:
            formatted = f"{value:,}"
        output += f"▸ {formatted}\n"
    else:
        output += f"▸ {value}\n"
    
    return output


# --- CUSTOMER TOOL (Privacy-Protected) ---

def get_customer_orders(customer_id: str) -> str:
    """
    Returns ALL order information for a specific customer, including product details.
    This tool filters data to show only the logged-in customer's orders.
    """
    data = load_data()
    if isinstance(data, str):
        return data
    
    customers_df = data['customers_df']
    orders_df = data['orders_df']
    products_df = data['products_df']
    
    clean_id = str(customer_id).strip().upper()
    
    customer = customers_df[customers_df['customer_id'] == clean_id]
    if customer.empty:
        return f"ERROR: Customer ID '{clean_id}' not found in the system."
    
    customer_orders = orders_df[orders_df['customer_id'] == clean_id].merge(
        products_df, on='product_id', how='left', suffixes=('_order', '_product')
    )
    
    if customer_orders.empty:
        customer_name = customer['name'].iloc[0]
        return f"Hello {customer_name}! You currently have no orders in the system."
    
    customer_name = customer['name'].iloc[0]
    customer_email = customer['email'].iloc[0]
    customer_region = customer['region'].iloc[0]
    
    result = f"👤 Customer: {customer_name} (ID: {clean_id})\n"
    result += f"📧 Email: {customer_email}\n"
    result += f"📍 Region: {customer_region}\n"
    result += f"📦 Total Orders: {len(customer_orders)}\n\n"
    result += "─" * 50 + "\n"
    
    for idx, row in customer_orders.iterrows():
        result += f"\n▸ Order ID: {row['order_id']}\n"
        result += f"  Product: {row['name']} ({row['category']})\n"
        result += f"  Price: ${row['price']:.2f}\n"
        result += f"  Status: {row['status']}\n"
        
        if 'order_date' in row and pd.notna(row['order_date']):
            result += f"  Order Date: {row['order_date'].strftime('%Y-%m-%d')}\n"
        
        if 'est_delivery' in row and pd.notna(row['est_delivery']):
            est_delivery = row['est_delivery']
            result += f"  Est. Delivery: {est_delivery.strftime('%Y-%m-%d')}\n"
            
            if row['status'] not in ['Delivered', 'Cancelled'] and datetime.now() > est_delivery:
                days_late = (datetime.now() - est_delivery).days
                result += f"  ⚠️ DELAYED: {days_late} day(s) overdue\n"
        else:
            result += f"  Est. Delivery: N/A\n"
        
        result += "─" * 50 + "\n"
    
    return result


def check_customer_order_status(customer_id: str) -> dict:
    """Checks if a customer has any delayed orders for notifications."""
    try:
        data = load_data()
        if isinstance(data, str):
            print(f"ERROR in check_customer_order_status: {data}")
            return {"status": "error", "message": "Unable to load data"}
        
        orders_df = data['orders_df']
        
        if orders_df.empty:
            return {"status": "normal", "message": "You have no active orders"}
        
        if 'customer_id' not in orders_df.columns:
            print(f"ERROR: 'customer_id' column not found in orders_df")
            print(f"Available columns: {list(orders_df.columns)}")
            return {"status": "error", "message": "Data structure error"}
        
        clean_id = str(customer_id).strip().upper()
        customer_orders = orders_df[orders_df['customer_id'] == clean_id]
        
        if customer_orders.empty:
            return {"status": "normal", "message": "You have no active orders"}
        
        # Check if est_delivery column exists
        if 'est_delivery' not in customer_orders.columns:
            # If no delivery dates, just show order count
            count = len(customer_orders)
            return {
                "status": "normal",
                "message": f"✅ You have {count} active order{'s' if count > 1 else ''}",
                "count": count
            }
        
        today = pd.to_datetime(datetime.now().strftime('%Y-%m-%d'))
        
        delayed = customer_orders[
            (customer_orders['est_delivery'] < today) & 
            (customer_orders['status'] != 'Delivered') &
            (customer_orders['status'] != 'Cancelled') &
            (pd.notna(customer_orders['est_delivery']))
        ]
        
        if not delayed.empty:
            count = len(delayed)
            return {
                "status": "delayed",
                "message": f"⚠️ You have {count} delayed order{'s' if count > 1 else ''}!",
                "count": count
            }
        else:
            on_time_count = len(customer_orders[customer_orders['status'].isin(['Pending', 'Shipped'])])
            delivered_count = len(customer_orders[customer_orders['status'] == 'Delivered'])
            
            if on_time_count > 0:
                return {
                    "status": "normal",
                    "message": f"✅ All {on_time_count} active order{'s are' if on_time_count > 1 else ' is'} on track!",
                    "count": on_time_count
                }
            else:
                return {
                    "status": "normal",
                    "message": f"✅ You have {delivered_count} completed order{'s' if delivered_count > 1 else ''}",
                    "count": delivered_count
                }
    except Exception as e:
        print(f"EXCEPTION in check_customer_order_status: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": "Error checking orders"}


# --- BUSINESS TOOL (Full Access) ---

def execute_pandas_code_business(python_code: str) -> str:
    """
    Executes Python code for business analytics with access to ALL tables.
    The code must calculate a result and store it in a variable named 'result'.
    """
    data_map = load_data()
    if isinstance(data_map, str):
        return data_map

    local_env = data_map.copy()
    local_env['pd'] = pd
    local_env['np'] = np
    local_env['datetime'] = datetime
    local_env['format_dataframe_output'] = format_dataframe_output
    local_env['format_series_output'] = format_series_output
    local_env['format_scalar_output'] = format_scalar_output

    try:
        exec(python_code, {}, local_env)
        
        if 'result' in local_env:
            result_value = local_env['result']
            
            if result_value is None:
                return "⚠️ Warning: The calculation returned None. Please check your code logic."
            
            if isinstance(result_value, pd.DataFrame):
                return format_dataframe_output(result_value)
            elif isinstance(result_value, pd.Series):
                return format_series_output(result_value)
            elif isinstance(result_value, (int, float, np.integer, np.floating)):
                return format_scalar_output(result_value)
            else:
                return str(result_value)
        else:
            return "❌ Error: The code ran, but no variable named 'result' was defined. Please assign the final answer to 'result'."
            
    except Exception as e:
        error_msg = f"Python Execution Error: {str(e)}\n\n"
        error_msg += "Common issues:\n"
        error_msg += "- Check column names match the schema exactly (case-sensitive)\n"
        error_msg += "- Ensure you're using correct merge keys (customer_id, product_id, order_id)\n"
        error_msg += "- Use suffixes when joining tables with duplicate column names\n"
        error_msg += "- Verify dataframe names: customers_df, orders_df, products_df, revenue_df"
        return error_msg


# --- AUDIT TOOLS ---

def check_for_revenue_anomalies() -> str:
    """Runs Isolation Forest on revenue data to detect unusual patterns."""
    data = load_data()
    if isinstance(data, str):
        return data
    
    revenue_df = data['revenue_df']
    
    if len(revenue_df) < 5:
        return "Not enough data points to run anomaly detection."

    if 'amount' not in revenue_df.columns:
        return "ERROR: 'amount' column not found in revenue data."

    X = revenue_df[['amount']].values
    model = IsolationForest(contamination=0.1, random_state=42)
    model.fit(X)
    revenue_df['anomaly'] = model.predict(X)
    anomalies = revenue_df[revenue_df['anomaly'] == -1]
    
    if not anomalies.empty:
        latest_anomaly = anomalies.iloc[-1]
        date_str = latest_anomaly['date'].strftime('%Y-%m-%d') if 'date' in latest_anomaly and pd.notna(latest_anomaly['date']) else 'Unknown'
        return f"CRITICAL REVENUE ANOMALY: Unusual pattern on {date_str} - Amount: ${latest_anomaly['amount']:,.2f} (Order: {latest_anomaly['order_id']})."
    else:
        return "SUCCESS: No significant revenue anomalies detected."


def check_for_critical_delays() -> str:
    """Checks for past-due orders across all customers."""
    data = load_data()
    if isinstance(data, str):
        return data
    
    orders_df = data['orders_df']
    customers_df = data['customers_df']
    
    # Check if est_delivery column exists
    if 'est_delivery' not in orders_df.columns:
        return "WARNING: Delivery date tracking not available. Cannot check for delays."
    
    today = pd.to_datetime(datetime.now().strftime('%Y-%m-%d'))
    
    critically_delayed_orders = orders_df[
        (orders_df['est_delivery'] < today) & 
        (orders_df['status'] != 'Delivered') &
        (orders_df['status'] != 'Cancelled') &
        (pd.notna(orders_df['est_delivery']))
    ]
    
    if not critically_delayed_orders.empty:
        count = len(critically_delayed_orders)
        
        delayed_with_customers = critically_delayed_orders.merge(
            customers_df, on='customer_id', how='left'
        )
        
        order_list = [str(oid) for oid in critically_delayed_orders['order_id'].tolist()[:3]]
        customer_names = [str(name) for name in delayed_with_customers['name'].tolist()[:3]]
        
        return f"ALERT: {count} orders are critically delayed!\nAffected Orders: {', '.join(order_list)}\nAffected Customers: {', '.join(customer_names)}"
    else:
        return "SUCCESS: No critical delivery delays found."


# Legacy functions for backward compatibility
def get_order_status(order_id: str) -> str:
    """Legacy function - redirects to get_customer_orders for single order lookup."""
    data = load_data()
    if isinstance(data, str):
        return data
    
    orders_df = data['orders_df']
    products_df = data['products_df']
    
    order = orders_df[orders_df['order_id'] == str(order_id)]
    
    if order.empty:
        return f"Order {order_id} not found."
    
    order_with_product = order.merge(products_df, on='product_id', how='left', suffixes=('', '_product'))
    
    return format_dataframe_output(order_with_product)


def query_business_analytics(query: str) -> str:
    """Legacy function - redirects to execute_pandas_code_business."""
    return f"Please use the execute_pandas_code_business tool for data analysis queries. Query: {query}"
