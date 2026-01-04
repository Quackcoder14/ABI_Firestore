# 🤖 ABI Agent - Autonomous Business Intelligence Platform

**An intelligent, dual-portal AI system powered by Google Gemini that revolutionizes customer service and business analytics.**

---

## 🌟 Overview

ABI Agent is a next-generation business intelligence platform that combines cutting-edge AI with enterprise-grade data analytics. Built with Google's Gemini AI and Firebase, it provides two specialized portals:

- **🛍️ Customer Portal**: AI-powered order tracking and support
- **📊 Business Command Center**: Advanced analytics, predictive insights, and lead tracking

### ✨ What Makes It Special?

- **🎯 Role-Based Intelligence**: Separate AI agents trained for customer service vs. business analytics
- **🔐 Privacy-First Design**: Customers only see their own data; businesses see everything
- **⚡ Real-Time Analytics**: Live anomaly detection and delivery delay monitoring
- **🔮 Predictive Supply Chain**: ML-powered inventory forecasting and stock-out predictions
- **💬 Natural Language Queries**: Ask questions in plain English, get data-driven answers
- **🎨 Stunning UI**: Cyberpunk-themed interface with smooth animations

---

## 🚀 Features

### 👥 Customer Portal

- **📦 Smart Order Tracking**: AI assistant that understands natural language queries
- **🚚 Delivery Status**: Real-time updates with automatic delay detection
- **🔔 Proactive Notifications**: Instant alerts for order issues
- **💬 Conversational AI**: Powered by Gemini 2.5 Flash for human-like interactions
- **🔒 Privacy Protected**: Strict data isolation per customer

### 📈 Business Command Center

#### 1. **AI Analytics Chat**
- Execute Python/Pandas code through natural language
- Cross-table data analysis across customers, orders, products, and revenue
- Automatic code generation for complex queries
- Real-time visualization of results

#### 2. **Customer Leads Dashboard**
- Track all customer interactions
- Engagement timestamps and message previews
- Lead scoring and prioritization

#### 3. **Predictive Supply Chain**
- ML-powered inventory forecasting using Isolation Forest
- Daily burn rate calculations
- Risk-level categorization (Critical, High, Moderate, Low)
- Stock-out prediction up to 30 days ahead

---

## 🛠️ Tech Stack

| Category | Technology |
|----------|-----------|
| **Frontend** | Streamlit 1.28+ |
| **AI/ML** | Google Gemini 2.5 Flash, scikit-learn |
| **Database** | Google Cloud Firestore |
| **Data Processing** | Pandas, NumPy |
| **Authentication** | Custom JSON-based system |
| **Deployment** | Streamlit Cloud |

---

## 📋 Prerequisites

- Python 3.9 or higher
- Google Cloud account with Firestore enabled
- Google AI Studio API key (Gemini)
- Git

---

## ⚡ Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/Quackcoder14/ABI_Firestore.git
cd ABI_Firestore
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set Up Firebase

1. Go to [Firebase Console](https://console.firebase.google.com/)
2. Create a new project (or use existing)
3. Enable **Firestore Database**
4. Generate a service account key:
   - Project Settings → Service Accounts → Generate New Private Key
5. Save as `firebase_creds.json` in the project root

### 4. Get Gemini API Key

1. Visit [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Create an API key
3. Copy the key

### 5. Initialize Firestore Collections

Create these collections in Firestore with the following structure:

**customers** collection:
```json
{
  "customer_id": "CUST_001",
  "name": "John Doe",
  "email": "john@example.com",
  "region": "North"
}
```

**orders** collection:
```json
{
  "order_id": "ORD_001",
  "customer_id": "CUST_001",
  "product_id": "PROD_001",
  "status": "Shipped",
  "order_date": "2024-01-15T10:30:00Z",
  "est_delivery": "2024-01-20T18:00:00Z"
}
```

**products** collection:
```json
{
  "product_id": "PROD_001",
  "name": "Laptop",
  "category": "Electronics",
  "price": 9000,
  "stock_level": 50
}
```

**revenue** collection:
```json
{
  "revenue_id": "REV_001",
  "order_id": "ORD_001",
  "amount": 9000,
  "date": "2024-01-15T10:30:00Z",
  "payment_method": "Credit Card"
}
```


## ☁️ Deploy to Streamlit Cloud

### 1. Push to GitHub

```bash
git add .
git commit -m "Initial commit"
git push origin main
```

### 2. Connect to Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io/)
2. Sign in with GitHub
3. Click "New app"
4. Select your repository
5. Set main file: `gapp.py`

### 3. Add Secrets

In Streamlit Cloud dashboard → App Settings → Secrets, add:

```toml
GEMINI_API_KEY = "your_gemini_api_key_here"

[firebase]
type = "service_account"
project_id = "your-project-id"
private_key_id = "your-private-key-id"
private_key = "-----BEGIN PRIVATE KEY-----\nYour-Key-Here\n-----END PRIVATE KEY-----\n"
client_email = "your-service-account@project.iam.gserviceaccount.com"
client_id = "your-client-id"
auth_uri = "https://accounts.google.com/o/oauth2/auth"
token_uri = "https://oauth2.googleapis.com/token"
auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
client_x509_cert_url = "your-cert-url"
```

### 4. Deploy!

Click "Deploy" and your app will be live in minutes! 🎉

---

## 📖 Usage Guide

### Creating User Accounts

#### Customer Account
1. Click "Enter Customer Portal"
2. Fill in registration form:
   - User ID: Any unique username
   - Password: Minimum 6 characters
   - Customer ID: Must match a `CustomerID` in Firestore (e.g., `CUST_001`)
3. Login and start tracking orders!

#### Business Account
1. Click "Enter Business Command"
2. Register with:
   - User ID: Any unique username
   - Password: Minimum 6 characters
3. Access the full analytics suite!

### Sample Customer Queries

```
"Show me all my orders"
"What's the status of order ORD_001?"
```

### Sample Business Queries

```
"Show me total revenue"
"List all delayed orders"
```

---

## 🏗️ Project Structure

```
abi-agent/
├── gapp.py                 # Main Streamlit application
├── gtools.py               # Backend tools & Firebase integration
├── requirements.txt        # Python dependencies
├── credentials.json        # User authentication
├── firebase_creds.example.json     # Firebase service account key format
├── LICENSE                # MIT License for this project
├── .gitignore             # Git ignore file
└── README.md              # This file
```

---

## 🔐 Security Features

- ✅ Role-based access control (RBAC)
- ✅ Data isolation between customers
- ✅ Password-protected accounts
- ✅ Secure Firebase authentication
- ✅ No sensitive data in code
- ✅ Environment-based configuration

---

## 🎨 UI Highlights

- **🌌 Animated Starfield Background**: Smooth, infinite space animation
- **🎭 Glassmorphism Cards**: Modern frosted-glass design
- **⚡ Hover Effects**: Interactive elements with glow effects
- **📱 Responsive Layout**: Works on desktop and mobile
- **🎨 Cyberpunk Theme**: Neon accents (cyan, gold, lime green)
- **🔔 Toast Notifications**: Elegant slide-in alerts

---

## 📊 Database Schema

```mermaid
erDiagram
    CUSTOMERS ||--o{ ORDERS : places
    PRODUCTS ||--o{ ORDERS : contains
    ORDERS ||--|| REVENUE : generates
    
    CUSTOMERS {
        string customer_id PK
        string name
        string email
        string region
    }
    
    ORDERS {
        string order_id PK
        string customer_id FK
        string product_id FK
        string status
        datetime order_date
        datetime est_delivery
    }
    
    PRODUCTS {
        string product_id PK
        string name
        string category
        float price
        int stock_level
    }
    
    REVENUE {
        string revenue_id PK
        string order_id FK
        float amount
        datetime date
        string payment_method
    }
```

---

## 🗺️ Roadmap

- [ ] **v2.0**: Multi-tenant support with organization accounts
- [ ] **v2.1**: Email notification system
- [ ] **v2.2**: Export reports to PDF/Excel
- [ ] **v2.3**: Advanced data visualizations (charts/graphs)
- [ ] **v2.4**: Integration with Shopify/WooCommerce
- [ ] **v2.5**: Mobile app (React Native)
- [ ] **v3.0**: Voice-based AI assistant

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Dhanush S*
- GitHub: [@Quackcoder14](https://github.com/Quackcoder14)
- Email: dhanushs123411@gmail.com

---
