# gtools.py
import firebase_admin
from firebase_admin import credentials, firestore
import pandas as pd
from datetime import datetime, timedelta
import json
import numpy as np
import os

# --- Configuration ---
FIREBASE_CREDS_FILE = 'firebase_creds.json'

# --- Initialization ---
db = None
FIREBASE_INIT_STATUS = ""

if os.path.exists(FIREBASE_CREDS_FILE):
    try:
        if not firebase_admin._apps:
            cred = credentials.Certificate(FIREBASE_CREDS_FILE)
            firebase_admin.initialize_app(cred)
        
        db = firestore.client()
        FIREBASE_INIT_STATUS = "SUCCESS"
    except Exception as e:
        FIREBASE_INIT_STATUS = str(f"ERROR: Firebase Initialization Failed. Details: {e}")
        db = None
else:
    FIREBASE_INIT_STATUS = str(f"ERROR: Firebase credentials file not found: {FIREBASE_CREDS_FILE}")
    db = None


# --- Data Loading Helper Functions ---

def get_firestore_collection(collection_name: str) -> tuple[pd.DataFrame, str]:
    """Retrieves all data from a specific Firestore collection."""
    if db is None or FIREBASE_INIT_STATUS != "SUCCESS":
        return pd.DataFrame(), FIREBASE_INIT_STATUS
    
    try:
        docs = db.collection(collection_name).stream()
        data = [doc.to_dict() for doc in docs]
        df = pd.DataFrame(data)
        
        if df.empty:
            return df, f"ERROR: Collection '{collection_name}' is empty or does not exist."
            
        return df, "SUCCESS"
        
    except Exception as e:
        return pd.DataFrame(), str(f"ERROR: Failed to retrieve collection '{collection_name}'. Details: {e}")


def load_data():
    """
    Loads all dataframes from Firestore and performs robust data cleaning.
    Returns cleaned dataframes with timezone-naive dates and proper data types.
    """
    
    orders_df, orders_status = get_firestore_collection('orders')
    revenue_df, revenue_status = get_firestore_collection('revenue')
    products_df, products_status = get_firestore_collection('products')

    if 'ERROR' in orders_status or 'ERROR' in revenue_status or 'ERROR' in products_status:
        error_msg = f"Data Load Error: Orders ({orders_status}), Revenue ({revenue_status}), Products ({products_status})"
        return None, error_msg, None

    # --- Robust Date Converter (Timezone Naive) ---
    def to_datetime_clean(series):
        if series.empty: return series
        series = series.apply(lambda x: x.to_datetime() if hasattr(x, 'to_datetime') else x)
        series = pd.to_datetime(series, errors='coerce')
        if series.dt.tz is not None:
            series = series.dt.tz_localize(None)
        return series
        
    # --- Orders DF Cleanup ---
    if not orders_df.empty:
        if 'OrderID' in orders_df.columns:
            orders_df['OrderID'] = orders_df['OrderID'].astype(str)
        for col in ['OrderDate', 'ShipDate', 'EstDeliveryDate']:
            if col in orders_df.columns:
                orders_df[col] = to_datetime_clean(orders_df[col])
        
        # Add computed columns for analysis
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        if 'EstDeliveryDate' in orders_df.columns:
            orders_df['DaysUntilDelivery'] = (orders_df['EstDeliveryDate'] - today).dt.days
            orders_df['IsOverdue'] = (orders_df['DaysUntilDelivery'] < 0) & \
                                     (~orders_df['Status'].astype(str).str.lower().isin(['delivered', 'cancelled']))
        
        if 'ShipDate' in orders_df.columns and 'OrderDate' in orders_df.columns:
            orders_df['ProcessingTime'] = (orders_df['ShipDate'] - orders_df['OrderDate']).dt.days
        
        if 'EstDeliveryDate' in orders_df.columns and 'ShipDate' in orders_df.columns:
            orders_df['EstShippingDuration'] = (orders_df['EstDeliveryDate'] - orders_df['ShipDate']).dt.days
    
    # --- Revenue DF Cleanup ---
    if not revenue_df.empty:
        if 'Date' in revenue_df.columns:
            revenue_df['Date'] = to_datetime_clean(revenue_df['Date'])
        revenue_df['Amount'] = pd.to_numeric(revenue_df.get('Amount', pd.Series([])), errors='coerce')
        
        # Add time-based grouping columns
        if 'Date' in revenue_df.columns:
            revenue_df['Year'] = revenue_df['Date'].dt.year
            revenue_df['Month'] = revenue_df['Date'].dt.month
            revenue_df['Quarter'] = revenue_df['Date'].dt.quarter
            revenue_df['DayOfWeek'] = revenue_df['Date'].dt.day_name()
    
    # --- Products DF Cleanup ---
    if not products_df.empty:
        if 'ProductID' in products_df.columns:
            products_df['ProductID'] = products_df['ProductID'].astype(str)
        products_df['TotalSold'] = pd.to_numeric(products_df.get('TotalSold', pd.Series([])), errors='coerce') 
        products_df['Price'] = pd.to_numeric(products_df.get('Price', pd.Series([])), errors='coerce')
        
        # Add computed revenue column
        if 'TotalSold' in products_df.columns and 'Price' in products_df.columns:
            products_df['TotalRevenue'] = products_df['TotalSold'] * products_df['Price']
    
    return orders_df, revenue_df, products_df


def analyze_dataframe(df, df_name, query_keywords):
    """
    Performs intelligent analysis on a dataframe based on query keywords.
    Returns relevant insights and statistics.
    """
    if df is None or df.empty:
        return f"No data available in {df_name}"
    
    analysis = []
    query_lower = ' '.join(query_keywords).lower()
    
    # Numeric columns analysis
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Date columns analysis
    date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    
    # Categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # Check for aggregation keywords
    if any(word in query_lower for word in ['total', 'sum', 'aggregate']):
        for col in numeric_cols:
            analysis.append(f"Total {col}: {df[col].sum():,.2f}")
    
    if any(word in query_lower for word in ['average', 'mean', 'avg']):
        for col in numeric_cols:
            analysis.append(f"Average {col}: {df[col].mean():,.2f}")
    
    if any(word in query_lower for word in ['highest', 'maximum', 'max', 'top']):
        for col in numeric_cols:
            max_val = df[col].max()
            analysis.append(f"Highest {col}: {max_val:,.2f}")
    
    if any(word in query_lower for word in ['lowest', 'minimum', 'min', 'bottom']):
        for col in numeric_cols:
            min_val = df[col].min()
            analysis.append(f"Lowest {col}: {min_val:,.2f}")
    
    # Time-based analysis
    if date_cols and any(word in query_lower for word in ['trend', 'over time', 'monthly', 'daily', 'recent']):
        for date_col in date_cols:
            if not df[date_col].isna().all():
                analysis.append(f"Date range for {date_col}: {df[date_col].min()} to {df[date_col].max()}")
    
    # Grouping analysis
    if any(word in query_lower for word in ['by', 'per', 'each', 'breakdown']):
        for cat_col in categorical_cols[:3]:  # Limit to first 3 categorical columns
            if df[cat_col].nunique() <= 20:  # Only for reasonable number of categories
                counts = df[cat_col].value_counts().head(5)
                analysis.append(f"Top 5 {cat_col}:\n{counts.to_string()}")
    
    return '\n'.join(analysis) if analysis else f"Data shape: {df.shape[0]} rows, {df.shape[1]} columns"


def create_comprehensive_summary(orders_df, revenue_df, products_df, query):
    """
    Creates a comprehensive business summary with cross-dataset insights.
    """
    summary = []
    query_lower = query.lower()
    
    # --- REVENUE INSIGHTS ---
    if not revenue_df.empty and 'Amount' in revenue_df.columns:
        total_revenue = revenue_df['Amount'].sum()
        avg_revenue = revenue_df['Amount'].mean()
        
        summary.append(f"=== REVENUE INSIGHTS ===")
        summary.append(f"Total Revenue: ${total_revenue:,.2f}")
        summary.append(f"Average Transaction: ${avg_revenue:,.2f}")
        
        if 'Date' in revenue_df.columns:
            latest_date = revenue_df['Date'].max()
            last_30_days = revenue_df[revenue_df['Date'] >= (latest_date - pd.Timedelta(days=30))]
            if not last_30_days.empty:
                recent_revenue = last_30_days['Amount'].sum()
                summary.append(f"Last 30 Days Revenue: ${recent_revenue:,.2f}")
        
        # Monthly trend
        if 'Year' in revenue_df.columns and 'Month' in revenue_df.columns:
            monthly_revenue = revenue_df.groupby(['Year', 'Month'])['Amount'].sum().tail(6)
            summary.append(f"\nRecent Monthly Revenue:\n{monthly_revenue.to_string()}")
    
    # --- PRODUCT INSIGHTS ---
    if not products_df.empty:
        summary.append(f"\n=== PRODUCT INSIGHTS ===")
        
        if 'TotalSold' in products_df.columns and 'Name' in products_df.columns:
            top_products = products_df.nlargest(5, 'TotalSold')[['Name', 'TotalSold', 'Price']]
            summary.append(f"Top 5 Best-Selling Products:\n{top_products.to_markdown(index=False)}")
        
        if 'TotalRevenue' in products_df.columns:
            top_revenue_products = products_df.nlargest(5, 'TotalRevenue')[['Name', 'TotalRevenue']]
            summary.append(f"\nTop 5 Revenue-Generating Products:\n{top_revenue_products.to_markdown(index=False)}")
    
    # --- ORDER INSIGHTS ---
    if not orders_df.empty:
        summary.append(f"\n=== ORDER INSIGHTS ===")
        summary.append(f"Total Orders: {len(orders_df)}")
        
        if 'Status' in orders_df.columns:
            status_counts = orders_df['Status'].value_counts()
            summary.append(f"Order Status Breakdown:\n{status_counts.to_string()}")
        
        if 'IsOverdue' in orders_df.columns:
            overdue_count = orders_df['IsOverdue'].sum()
            if overdue_count > 0:
                summary.append(f"\n⚠️ ALERT: {overdue_count} orders are overdue!")
        
        if 'ProcessingTime' in orders_df.columns:
            avg_processing = orders_df['ProcessingTime'].mean()
            summary.append(f"Average Order Processing Time: {avg_processing:.1f} days")
        
        if 'ShippingMethod' in orders_df.columns:
            shipping_methods = orders_df['ShippingMethod'].value_counts()
            summary.append(f"\nShipping Methods Used:\n{shipping_methods.to_string()}")
    
    # --- CROSS-DATASET INSIGHTS ---
    if not orders_df.empty and not revenue_df.empty:
        summary.append(f"\n=== CROSS-ANALYSIS ===")
        if len(orders_df) > 0 and total_revenue > 0:
            avg_order_value = total_revenue / len(orders_df)
            summary.append(f"Average Order Value: ${avg_order_value:,.2f}")
    
    return '\n'.join(summary)


# --- Enhanced Tool Functions ---

def query_business_analytics(query: str) -> str:
    """
    Advanced analytical tool that can answer complex questions by analyzing relationships
    between orders, revenue, and products data. Supports aggregations, trends, comparisons,
    and multi-dimensional analysis.
    
    Examples of queries it can handle:
    - "What's our total revenue and how does it break down by month?"
    - "Which products generate the most revenue?"
    - "How many orders are delayed and by how much?"
    - "What's the average processing time for orders?"
    - "Compare revenue trends across different quarters"
    """
    orders_df, revenue_df, products_df = load_data()

    if orders_df is None:
        return revenue_df  # Contains the Firebase error message

    # Create comprehensive summary
    comprehensive_summary = create_comprehensive_summary(orders_df, revenue_df, products_df, query)
    
    # Extract keywords from query for targeted analysis
    query_keywords = query.lower().split()
    
    # Perform targeted analysis based on query
    targeted_insights = []
    
    # Orders-specific analysis
    if any(word in query_keywords for word in ['order', 'orders', 'delivery', 'shipping', 'delay']):
        orders_analysis = analyze_dataframe(orders_df, 'Orders', query_keywords)
        targeted_insights.append(f"--- ORDER ANALYSIS ---\n{orders_analysis}")
    
    # Revenue-specific analysis
    if any(word in query_keywords for word in ['revenue', 'sales', 'money', 'income', 'earnings']):
        revenue_analysis = analyze_dataframe(revenue_df, 'Revenue', query_keywords)
        targeted_insights.append(f"--- REVENUE ANALYSIS ---\n{revenue_analysis}")
    
    # Product-specific analysis
    if any(word in query_keywords for word in ['product', 'products', 'item', 'items', 'inventory']):
        products_analysis = analyze_dataframe(products_df, 'Products', query_keywords)
        targeted_insights.append(f"--- PRODUCT ANALYSIS ---\n{products_analysis}")
    
    # Add sample data for context
    data_samples = []
    if not orders_df.empty:
        relevant_cols = [col for col in ['OrderID', 'Status', 'OrderDate', 'ShipDate', 
                                          'EstDeliveryDate', 'IsOverdue', 'ProcessingTime'] 
                        if col in orders_df.columns]
        data_samples.append(f"--- ORDERS SAMPLE ---\n{orders_df[relevant_cols].head(3).to_markdown(index=False)}")
    
    if not revenue_df.empty:
        relevant_cols = [col for col in ['Date', 'Amount', 'Year', 'Month', 'Quarter'] 
                        if col in revenue_df.columns]
        data_samples.append(f"--- REVENUE SAMPLE ---\n{revenue_df[relevant_cols].head(3).to_markdown(index=False)}")
    
    if not products_df.empty:
        relevant_cols = [col for col in ['ProductID', 'Name', 'Price', 'TotalSold', 'TotalRevenue'] 
                        if col in products_df.columns]
        data_samples.append(f"--- PRODUCTS SAMPLE ---\n{products_df[relevant_cols].head(3).to_markdown(index=False)}")
    
    # Construct final response
    response_parts = [
        f"QUERY: {query}",
        "\n" + "="*80 + "\n",
        comprehensive_summary
    ]
    
    if targeted_insights:
        response_parts.append("\n" + "="*80 + "\n")
        response_parts.extend(targeted_insights)
    
    response_parts.append("\n" + "="*80 + "\n")
    response_parts.extend(data_samples)
    
    return '\n'.join(response_parts)


def get_order_status(order_id: str) -> str:
    """
    Retrieves comprehensive information about a specific order, including:
    - Current status and all relevant dates
    - Delay calculations and shipping information
    - Processing time metrics
    """
    orders_df, _, _ = load_data()

    if orders_df is None:
        return "Error: Could not load data from Firebase."
    
    if 'OrderID' not in orders_df.columns:
        return json.dumps({"status": "Error", "message": "The 'orders' data is missing the required 'OrderID' column."})

    # Robust matching
    orders_df['OrderID_str'] = orders_df['OrderID'].astype(str).str.strip().str.upper()
    target_id = str(order_id).strip().upper()
    
    order_data = orders_df[orders_df['OrderID_str'] == target_id]
    
    if order_data.empty:
        sample_ids = orders_df['OrderID_str'].unique()[:5]
        return json.dumps({
            "status": "Error", 
            "message": f"Order ID {order_id} not found.",
            "sample_available_ids": sample_ids.tolist() if sample_ids.size > 0 else []
        })

    # Prepare comprehensive result with all available information
    order = order_data.iloc[0]
    result = {
        "OrderID": order.get('OrderID'),
        "Status": order.get('Status'),
        "OrderDate": str(order.get('OrderDate')) if pd.notna(order.get('OrderDate')) else None,
        "ShipDate": str(order.get('ShipDate')) if pd.notna(order.get('ShipDate')) else None,
        "EstDeliveryDate": str(order.get('EstDeliveryDate')) if pd.notna(order.get('EstDeliveryDate')) else None,
        "ShippingMethod": order.get('ShippingMethod'),
        "CustomerID": order.get('CustomerID')
    }
    
    # Add computed insights
    if 'ProcessingTime' in order_data.columns and pd.notna(order.get('ProcessingTime')):
        result['ProcessingTime_Days'] = float(order.get('ProcessingTime'))
    
    if 'DaysUntilDelivery' in order_data.columns and pd.notna(order.get('DaysUntilDelivery')):
        days_until = float(order.get('DaysUntilDelivery'))
        result['DaysUntilDelivery'] = days_until
        
        if days_until < 0:
            result['DelayStatus'] = f"OVERDUE by {abs(days_until):.0f} days"
        elif days_until == 0:
            result['DelayStatus'] = "Due today"
        else:
            result['DelayStatus'] = f"On track, {days_until:.0f} days remaining"
    
    if 'IsOverdue' in order_data.columns:
        result['IsOverdue'] = bool(order.get('IsOverdue'))
    
    return json.dumps(result, indent=2, default=str)


def check_for_revenue_anomalies(days: int = 7, threshold: float = 2.0) -> str:
    """
    Analyzes revenue data to identify statistical anomalies using z-score analysis.
    Also provides trend analysis and comparison with historical averages.
    
    Args:
        days: Number of recent days to analyze
        threshold: Z-score threshold (default 2.0 = 2 standard deviations)
    """
    _, revenue_df, _ = load_data()

    if revenue_df is None:
        return "Error: Could not load data from Firebase."

    revenue_df = revenue_df.dropna(subset=['Date', 'Amount'])
    if revenue_df.empty:
        return "No valid revenue data available for analysis."

    latest_date = revenue_df['Date'].max()
    start_date = latest_date - pd.Timedelta(days=days)
    
    recent_revenue = revenue_df[(revenue_df['Date'] >= start_date) & 
                                (revenue_df['Date'] <= latest_date)].copy()
    
    if recent_revenue.empty:
        return f"No revenue data found in the last {days} days."

    # Calculate statistics
    overall_mean = revenue_df['Amount'].mean()
    overall_std = revenue_df['Amount'].std()
    recent_mean = recent_revenue['Amount'].mean()
    recent_total = recent_revenue['Amount'].sum()
    
    # Identify anomalies
    anomalies = []
    if overall_std > 0:
        for _, row in recent_revenue.iterrows():
            z_score = (row['Amount'] - overall_mean) / overall_std
            if abs(z_score) > threshold:
                anomalies.append({
                    "Date": row['Date'].strftime('%Y-%m-%d'),
                    "Amount": f"${row['Amount']:,.2f}",
                    "Deviation": f"{z_score:+.2f}σ",
                    "Type": "High" if z_score > 0 else "Low"
                })
    
    # Trend analysis
    if len(recent_revenue) >= 3:
        recent_revenue_sorted = recent_revenue.sort_values('Date')
        first_half = recent_revenue_sorted.head(len(recent_revenue_sorted)//2)['Amount'].mean()
        second_half = recent_revenue_sorted.tail(len(recent_revenue_sorted)//2)['Amount'].mean()
        
        if second_half > first_half * 1.1:
            trend = "INCREASING"
        elif second_half < first_half * 0.9:
            trend = "DECREASING"
        else:
            trend = "STABLE"
    else:
        trend = "INSUFFICIENT_DATA"
    
    result = {
        "analysis_period": f"Last {days} days ({start_date.strftime('%Y-%m-%d')} to {latest_date.strftime('%Y-%m-%d')})",
        "recent_total_revenue": f"${recent_total:,.2f}",
        "recent_average_daily": f"${recent_mean:,.2f}",
        "historical_average_daily": f"${overall_mean:,.2f}",
        "trend": trend,
        "anomalies_detected": len(anomalies),
        "anomalies": anomalies
    }

    if anomalies:
        return f"⚠️ REVENUE ANOMALIES DETECTED:\n{json.dumps(result, indent=2)}"
    else:
        return f"✓ No significant revenue anomalies detected.\n{json.dumps(result, indent=2)}"


def check_for_critical_delays(days_overdue: int = 0) -> str:
    """
    Comprehensive delay analysis that identifies:
    - Orders past their estimated delivery date
    - Orders at risk of being late
    - Processing bottlenecks
    - Shipping method performance
    
    Args:
        days_overdue: Minimum days overdue to flag (0 = any overdue order)
    """
    orders_df, _, _ = load_data()

    if orders_df is None:
        return "Error: Could not load data from Firebase."
    
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    
    # Filter for pending orders
    pending_orders = orders_df[
        ~orders_df['Status'].astype(str).str.lower().isin(['delivered', 'cancelled'])
    ].copy()

    if pending_orders.empty:
        return json.dumps({
            "status": "SUCCESS",
            "message": "No pending orders found. All orders are either delivered or cancelled."
        }, indent=2)

    # Calculate delays if not already present
    if 'DaysUntilDelivery' not in pending_orders.columns and 'EstDeliveryDate' in pending_orders.columns:
        pending_orders['DaysUntilDelivery'] = (pending_orders['EstDeliveryDate'] - today).dt.days
    
    # Categorize orders
    overdue_orders = pending_orders[pending_orders['DaysUntilDelivery'] < -days_overdue]
    at_risk_orders = pending_orders[(pending_orders['DaysUntilDelivery'] >= -days_overdue) & 
                                     (pending_orders['DaysUntilDelivery'] <= 2)]
    on_track_orders = pending_orders[pending_orders['DaysUntilDelivery'] > 2]
    
    result = {
        "analysis_date": today.strftime('%Y-%m-%d'),
        "total_pending_orders": len(pending_orders),
        "overdue_count": len(overdue_orders),
        "at_risk_count": len(at_risk_orders),
        "on_track_count": len(on_track_orders)
    }
    
    # Add overdue details
    if not overdue_orders.empty:
        overdue_details = []
        for _, order in overdue_orders.nlargest(10, 'DaysUntilDelivery', keep='all').iterrows():
            overdue_details.append({
                "OrderID": order['OrderID'],
                "Status": order.get('Status'),
                "EstDeliveryDate": str(order.get('EstDeliveryDate')),
                "DaysOverdue": abs(int(order['DaysUntilDelivery'])),
                "ShippingMethod": order.get('ShippingMethod')
            })
        result["overdue_orders"] = overdue_details
        
        # Shipping method analysis for overdue orders
        if 'ShippingMethod' in overdue_orders.columns:
            shipping_issues = overdue_orders['ShippingMethod'].value_counts().to_dict()
            result["shipping_method_issues"] = shipping_issues
    
    # Add at-risk details
    if not at_risk_orders.empty:
        at_risk_details = []
        for _, order in at_risk_orders.head(5).iterrows():
            at_risk_details.append({
                "OrderID": order['OrderID'],
                "Status": order.get('Status'),
                "EstDeliveryDate": str(order.get('EstDeliveryDate')),
                "DaysRemaining": int(order['DaysUntilDelivery'])
            })
        result["at_risk_orders"] = at_risk_details
    
    # Processing time analysis
    if 'ProcessingTime' in pending_orders.columns:
        avg_processing = pending_orders['ProcessingTime'].mean()
        if pd.notna(avg_processing):
            result["avg_processing_time_days"] = round(float(avg_processing), 1)
    
    if len(overdue_orders) > 0:
        return f"🚨 CRITICAL ALERT: {len(overdue_orders)} orders are overdue!\n{json.dumps(result, indent=2)}"
    elif len(at_risk_orders) > 0:
        return f"⚠️ WARNING: {len(at_risk_orders)} orders are at risk of delay.\n{json.dumps(result, indent=2)}"
    else:
        return f"✓ All pending orders are on track.\n{json.dumps(result, indent=2)}"