import streamlit as st
import pandas as pd
import numpy as np
import re # For regular expressions in NLP-like tasks
from io import BytesIO

# Set Streamlit page configuration
st.set_page_config(layout="wide", page_title="Refund Abuse Detection App", page_icon="🕵️‍♀️")

# --- Helper Functions (Refactored from previous steps) ---

@st.cache_data # Cache data loading and preprocessing to avoid re-running on every interaction
def load_and_preprocess_retail(uploaded_file):
    """Loads and preprocesses the OnlineRetail.csv data."""
    try:
        df = pd.read_csv(uploaded_file, encoding='ISO-8859-1')
        st.success("OnlineRetail.csv loaded successfully.")

        # Drop rows with missing CustomerID
        original_rows = len(df)
        df.dropna(subset=['CustomerID'], inplace=True)
        st.info(f"Dropped {original_rows - len(df)} rows with missing CustomerID from retail data.")
        df['CustomerID'] = df['CustomerID'].astype(int)

        # Convert InvoiceDate to datetime, coercing errors
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
        original_rows_after_customer_id = len(df)
        df.dropna(subset=['InvoiceDate'], inplace=True)
        st.info(f"Dropped {original_rows_after_customer_id - len(df)} rows with unparsable InvoiceDate from retail data.")


        # Identify returns
        df['IsReturn'] = df['InvoiceNo'].astype(str).str.startswith('C') | (df['Quantity'] < 0)
        df['PurchaseQuantity'] = df['Quantity'].apply(lambda x: x if x > 0 else 0)
        df['ReturnQuantity'] = df['Quantity'].apply(lambda x: abs(x) if x < 0 else 0)
        df['LineTotal'] = df['Quantity'] * df['UnitPrice']

        return df
    except Exception as e:
        st.error(f"Error loading or preprocessing retail data: {e}")
        return None

@st.cache_data # Cache data loading and preprocessing
def load_and_preprocess_tickets(uploaded_file):
    """Loads and preprocesses the customer_support_tickets.csv data."""
    try:
        df = pd.read_csv(uploaded_file)
        st.success("customer_support_tickets.csv loaded successfully.")

        df['Date of Purchase'] = pd.to_datetime(df['Date of Purchase'])
        df['Ticket Description'] = df['Ticket Description'].astype(str).fillna('')
        return df
    except Exception as e:
        st.error(f"Error loading or preprocessing ticket data: {e}")
        return None

@st.cache_data # Cache feature engineering results
def engineer_features(df_retail_processed, df_tickets_processed):
    """
    Engineers structured and NLP features and combines them.
    Also defines the conceptual target variable.
    """
    st.subheader("Step 2: Feature Engineering")

    # --- Structured Features from df_retail_processed ---
    st.write("Engineering structured features from retail data...")
    customer_retail_summary = df_retail_processed.groupby('CustomerID').agg(
        TotalUniqueOrders=('InvoiceNo', 'nunique'),
        TotalItemsPurchased=('PurchaseQuantity', 'sum'),
        TotalItemsReturned=('ReturnQuantity', 'sum'),
        TotalPurchaseValue=('LineTotal', lambda x: x[x > 0].sum()),
        TotalRefundValue=('LineTotal', lambda x: x[x < 0].abs().sum()),
        FirstTransactionDate=('InvoiceDate', 'min'),
        LastTransactionDate=('InvoiceDate', 'max')
    ).reset_index()

    customer_retail_summary['RefundRate_by_Items'] = customer_retail_summary['TotalItemsReturned'] / customer_retail_summary['TotalItemsPurchased'].replace(0, np.nan)
    customer_retail_summary['RefundRate_by_Items'].fillna(0, inplace=True)

    customer_retail_summary['RefundRate_by_Value'] = customer_retail_summary['TotalRefundValue'] / customer_retail_summary['TotalPurchaseValue'].replace(0, np.nan)
    customer_retail_summary['RefundRate_by_Value'].fillna(0, inplace=True)

    customer_retail_summary['AccountAgeDays'] = (customer_retail_summary['LastTransactionDate'] - customer_retail_summary['FirstTransactionDate']).dt.days.fillna(0)

    st.write("Sample Customer Retail Summary (Structured Features):")
    st.dataframe(customer_retail_summary.head())


    # --- Simulated CustomerID Mapping for Linking ---
    st.write("Simulating CustomerID mapping for linking datasets...")
    unique_retail_customers = customer_retail_summary['CustomerID'].unique()
    unique_ticket_emails = df_tickets_processed['Customer Email'].unique()

    # Ensure unique_retail_customers is not empty before sampling
    if len(unique_retail_customers) == 0:
        st.error("No valid CustomerIDs found in retail data for linkage. Cannot proceed with feature engineering.")
        return None, None

    # Map a random retail CustomerID to each unique ticket email
    # This is for demonstration; a real system would have explicit links.
    np.random.seed(42) # For reproducibility
    email_to_simulated_customer_id_map = {
        email: np.random.choice(unique_retail_customers)
        for email in unique_ticket_emails
    }

    df_tickets_processed['Simulated_CustomerID'] = df_tickets_processed['Customer Email'].map(email_to_simulated_customer_id_map)
    df_tickets_processed.dropna(subset=['Simulated_CustomerID'], inplace=True)
    df_tickets_processed['Simulated_CustomerID'] = df_tickets_processed['Simulated_CustomerID'].astype(int)

    st.write("Sample Tickets with Simulated CustomerID:")
    st.dataframe(df_tickets_processed[['Customer Email', 'Simulated_CustomerID']].head())


    # --- NLP Features from df_tickets_processed ---
    st.write("Engineering NLP features from ticket data (simplified)...")

    # Keyword detection function
    def count_keywords(text, keywords_list):
        count = 0
        for keyword in keywords_list:
            count += len(re.findall(r'\b' + re.escape(keyword) + r'\b', text.lower()))
        return count

    # Basic sentiment approximation function
    def get_basic_sentiment_score(text):
        text_lower = text.lower()
        positive_sentiment_terms = ['happy', 'satisfied', 'resolved', 'excellent', 'great', 'thank you']
        negative_sentiment_terms = ['bad', 'poor', 'unhappy', 'frustrated', 'terrible', 'issue', 'problem', 'unresolved', 'damaged', 'broken', 'missing', 'refund', 'return']
        pos_count = count_keywords(text_lower, positive_sentiment_terms)
        neg_count = count_keywords(text_lower, negative_sentiment_terms)

        if (pos_count + neg_count) == 0:
            return 0.0
        return (pos_count - neg_count) / (pos_count + neg_count)

    # Define keywords for fraud-related themes
    refund_keywords = ['refund', 'return', 'cancel', 'money back', 'send back', 'credit']
    damage_keywords = ['damaged', 'broken', 'faulty', 'defective', 'not working', 'malfunction']
    missing_keywords = ['missing', 'never arrived', 'lost package', 'not received', 'where is my order']

    df_tickets_processed['Keywords_RefundCount'] = df_tickets_processed['Ticket Description'].apply(lambda x: count_keywords(x, refund_keywords))
    df_tickets_processed['Keywords_DamageCount'] = df_tickets_processed['Ticket Description'].apply(lambda x: count_keywords(x, damage_keywords))
    df_tickets_processed['Keywords_MissingCount'] = df_tickets_processed['Ticket Description'].apply(lambda x: count_keywords(x, missing_keywords))
    df_tickets_processed['TicketSentimentScore'] = df_tickets_processed['Ticket Description'].apply(get_basic_sentiment_score)

    # Aggregate NLP features per simulated CustomerID
    customer_ticket_summary = df_tickets_processed.groupby('Simulated_CustomerID').agg(
        TotalTickets=('Ticket ID', 'count'),
        Tickets_TypeRefundRequest=('Ticket Type', lambda x: (x == 'Refund request').sum()),
        Tickets_TypeDeliveryIssue=('Ticket Type', lambda x: (x == 'Delivery issue').sum()),
        Tickets_TypeProductInquiry=('Ticket Type', lambda x: (x == 'Product inquiry').sum()),
        Tickets_KeywordRefundCount=('Keywords_RefundCount', 'sum'),
        Tickets_KeywordDamageCount=('Keywords_DamageCount', 'sum'),
        Tickets_KeywordMissingCount=('Keywords_MissingCount', 'sum'),
        AvgTicketSentimentScore=('TicketSentimentScore', 'mean'),
        MinTicketSentimentScore=('TicketSentimentScore', 'min')
    ).reset_index()

    st.write("Sample Customer Ticket Summary (NLP-Derived Features):")
    st.dataframe(customer_ticket_summary.head())

    st.subheader("Step 3: Combining Multi-Modal Data")
    # Rename CustomerID in retail summary to match for merge
    customer_retail_summary.rename(columns={'CustomerID': 'Simulated_CustomerID'}, inplace=True)

    # Merge structured features with NLP-derived features
    final_combined_df = pd.merge(
        customer_retail_summary,
        customer_ticket_summary,
        on='Simulated_CustomerID',
        how='left'
    )

    # Fill NaN values for customers with no corresponding tickets
    final_combined_df.fillna({
        'TotalTickets': 0, 'Tickets_TypeRefundRequest': 0, 'Tickets_TypeDeliveryIssue': 0,
        'Tickets_TypeProductInquiry': 0, 'Tickets_KeywordRefundCount': 0,
        'Tickets_KeywordDamageCount': 0, 'Tickets_KeywordMissingCount': 0,
        'AvgTicketSentimentScore': 0.0, 'MinTicketSentimentScore': 0.0
    }, inplace=True)

    # Define the conceptual target variable (`is_refund_abuser`)
    # This simulates the model's 'prediction' based on our heuristic for demonstration.
    final_combined_df['is_refund_abuser'] = (
        (final_combined_df['RefundRate_by_Items'] > 0.3) &
        (final_combined_df['Tickets_TypeRefundRequest'] >= 2)
    ) | (
        (final_combined_df['RefundRate_by_Value'] > 0.2) &
        ( (final_combined_df['Tickets_KeywordMissingCount'] > 0) | (final_combined_df['Tickets_KeywordDamageCount'] > 0) )
    ) | (
        (final_combined_df['MinTicketSentimentScore'] < -0.5) &
        (final_combined_df['TotalItemsReturned'] > 5)
    )
    final_combined_df['is_refund_abuser'] = final_combined_df['is_refund_abuser'].astype(int)

    # Define the list of features that would go into a real ML model
    features = [col for col in final_combined_df.columns if col not in [
        'Simulated_CustomerID', 'FirstTransactionDate', 'LastTransactionDate', 'is_refund_abuser'
    ]]

    st.write("Sample of Final Combined Multi-Modal Dataset:")
    st.dataframe(final_combined_df[['Simulated_CustomerID', 'TotalItemsPurchased', 'TotalItemsReturned',
                                     'RefundRate_by_Items', 'TotalTickets',
                                     'Tickets_TypeRefundRequest', 'AvgTicketSentimentScore', 'is_refund_abuser']].head())

    st.success("Feature engineering and data combination complete.")
    return final_combined_df, features

# --- Streamlit App Layout ---

st.title("🕵️‍♀️ Multi-Modal Refund Abuse Detection")
st.markdown("""
This app demonstrates a conceptual pipeline for detecting refund abuse using both
**structured retail transaction data** and **unstructured customer support ticket text**.
It combines these data sources, engineers features, and applies a heuristic to flag potential abusers.

**Note:** This is a demonstration. A live production system would involve a trained Machine Learning model (e.g., LightGBM)
instead of a simple heuristic for prediction. Libraries for full ML model training/inference (like `scikit-learn`, `lightgbm`, `shap`)
are not available in this environment.
""")

st.sidebar.header("Upload Datasets")
retail_file = st.sidebar.file_uploader("Upload OnlineRetail.csv", type=["csv"])
tickets_file = st.sidebar.file_uploader("Upload customer_support_tickets.csv", type=["csv"])

df_retail_processed = None
df_tickets_processed = None

if retail_file:
    df_retail_processed = load_and_preprocess_retail(retail_file)

if tickets_file:
    df_tickets_processed = load_and_preprocess_tickets(tickets_file)

if df_retail_processed is not None and df_tickets_processed is not None:
    st.subheader("Step 1: Data Loaded & Preprocessed")
    st.write("Raw retail data (sample):")
    st.dataframe(df_retail_processed.head())
    st.write("Raw tickets data (sample):")
    st.dataframe(df_tickets_processed.head())


    final_combined_df, features = engineer_features(df_retail_processed, df_tickets_processed)

    if final_combined_df is not None:
        st.subheader("Step 4: Refund Abuse Prediction (Conceptual)")
        st.write("Applying the conceptual heuristic model to flag potential abusers:")

        num_total_customers = len(final_combined_df)
        num_abusers = final_combined_df['is_refund_abuser'].sum()
        percentage_abusers = (num_abusers / num_total_customers) * 100 if num_total_customers > 0 else 0

        st.success(f"**Prediction Complete!**")
        st.write(f"Out of **{num_total_customers}** customers, **{num_abusers}** ({percentage_abusers:.2f}%) are flagged as potential refund abusers based on the heuristic.")

        if num_abusers > 0:
            st.write("### Flagged Customers (Top 10)")
            flagged_customers = final_combined_df[final_combined_df['is_refund_abuser'] == 1].copy()
            st.dataframe(flagged_customers.head(10))

            st.write("### Conceptual Insights into Flagged Customers (Mimicking SHAP)")
            st.markdown("""
            **How to interpret this (if a real ML model were used with SHAP):**

            * **SHAP values** explain how each feature contributes to a customer being flagged as an abuser.
            * **Positive SHAP values (pushing towards 1/abuser)**: Features that make the customer *more likely* to be an abuser.
            * **Negative SHAP values (pushing towards 0/legitimate)**: Features that make the customer *less likely* to be an abuser.
            * The magnitude of the SHAP value indicates the strength of the contribution.
            """)

            # Display conceptual SHAP-like reasons for the first 5 flagged customers
            if not flagged_customers.empty:
                st.write("Below are *conceptual reasons* for some flagged customers, based on our heuristic:")
                for i, row in flagged_customers.head(5).iterrows():
                    st.markdown(f"#### Customer ID: {int(row['Simulated_CustomerID'])}")
                    st.write(f"Predicted as Abuser: **{bool(row['is_refund_abuser'])}**")
                    reasons = []

                    if row['RefundRate_by_Items'] > 0.3 and row['Tickets_TypeRefundRequest'] >= 2:
                        reasons.append(f"  - **High Item Refund Rate ({row['RefundRate_by_Items']:.2f})** combined with **multiple Refund Request Tickets ({int(row['Tickets_TypeRefundRequest'])})**.")
                    if row['RefundRate_by_Value'] > 0.2 and (row['Tickets_KeywordMissingCount'] > 0 or row['Tickets_KeywordDamageCount'] > 0):
                        reasons.append(f"  - **High Value Refund Rate ({row['RefundRate_by_Value']:.2f})** and **tickets mentioning missing/damaged items** (Missing: {int(row['Tickets_KeywordMissingCount'])}, Damaged: {int(row['Tickets_KeywordDamageCount'])}).")
                    if row['MinTicketSentimentScore'] < -0.5 and row['TotalItemsReturned'] > 5:
                        reasons.append(f"  - **Very Negative Ticket Sentiment ({row['MinTicketSentimentScore']:.2f})** and **significant items returned ({int(row['TotalItemsReturned'])})**.")

                    if reasons:
                        st.markdown("##### Key contributing factors (conceptual):")
                        for reason in reasons:
                            st.markdown(reason)
                    else:
                        st.markdown("##### Factors for flagging (based on heuristic): No specific conditions met for this customer's sample values to clearly define the exact heuristic path. (This would be more precise with real SHAP values).")

        else:
            st.info("No customers were flagged as potential refund abusers based on the current heuristic.")

st.sidebar.markdown("---")
st.sidebar.markdown("Built for demonstrating multi-modal data fusion.")