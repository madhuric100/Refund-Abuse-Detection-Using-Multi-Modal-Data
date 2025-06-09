import pandas as pd
import numpy as np
import re
import pickle
import os # For checking if files exist
import re # For regular expressions in NLP-like tasks
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import matplotlib.pyplot as plt


# --- Helper Functions (Copied from Streamlit app for consistency) ---

def load_data(file_path, encoding='utf-8'):
    """Loads a CSV file into a DataFrame."""
    try:
        df = pd.read_csv(file_path, encoding=encoding)
        print(f"'{os.path.basename(file_path)}' loaded successfully.")
        return df
    except FileNotFoundError:
        print(f"Error: '{os.path.basename(file_path)}' not found. Please ensure it's in the same directory.")
        return None
    except Exception as e:
        print(f"Error loading '{os.path.basename(file_path)}': {e}")
        return None

def preprocess_retail_data(df_retail_raw):
    """Preprocesses the raw OnlineRetail DataFrame."""
    if df_retail_raw is None:
        return None

    df = df_retail_raw.copy()
    original_rows = len(df)

    # Drop rows with missing CustomerID
    df.dropna(subset=['CustomerID'], inplace=True)
    print(f"  Retail preprocessing: Dropped {original_rows - len(df)} rows with missing CustomerID.")
    df['CustomerID'] = df['CustomerID'].astype(int)

    # Convert InvoiceDate to datetime, coercing errors
    original_rows_after_customer_id = len(df)
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
    df.dropna(subset=['InvoiceDate'], inplace=True)
    print(f"  Retail preprocessing: Dropped {original_rows_after_customer_id - len(df)} rows with unparsable InvoiceDate.")

    # Identify returns and calculate quantities/line totals
    df['IsReturn'] = df['InvoiceNo'].astype(str).str.startswith('C') | (df['Quantity'] < 0)
    df['PurchaseQuantity'] = df['Quantity'].apply(lambda x: x if x > 0 else 0)
    df['ReturnQuantity'] = df['Quantity'].apply(lambda x: abs(x) if x < 0 else 0)
    df['LineTotal'] = df['Quantity'] * df['UnitPrice']
    return df

def preprocess_tickets_data(df_tickets_raw):
    """Preprocesses the raw customer_support_tickets DataFrame."""
    if df_tickets_raw is None:
        return None

    df = df_tickets_raw.copy()
    df['Date of Purchase'] = pd.to_datetime(df['Date of Purchase'])
    df['Ticket Description'] = df['Ticket Description'].astype(str).fillna('')
    return df

def engineer_and_combine_features(df_retail_processed, df_tickets_processed):
    """
    Engineers structured and NLP features and combines them.
    Returns the final combined DataFrame and the list of features for the model.
    """
    if df_retail_processed is None or df_tickets_processed is None:
        print("  Skipping feature engineering as one or both processed DataFrames are None.")
        return None, None

    print("\n  Engineering & Combining Features...")

    # --- Structured Features from df_retail_processed ---
    print("    Extracting structured features from retail data...")
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

    # --- Simulated CustomerID Mapping for Linking ---
    print("    Simulating CustomerID linkage between datasets...")
    unique_retail_customers = customer_retail_summary['CustomerID'].unique()
    unique_ticket_emails = df_tickets_processed['Customer Email'].unique()

    if len(unique_retail_customers) == 0:
        print("    Warning: No valid CustomerIDs in retail data for linkage. Skipping ticket features.")
        # Create an empty customer_ticket_summary to avoid errors later
        customer_ticket_summary = pd.DataFrame(columns=['Simulated_CustomerID', 'TotalTickets', 'Tickets_TypeRefundRequest',
                                                         'Tickets_TypeDeliveryIssue', 'Tickets_TypeProductInquiry',
                                                         'Tickets_KeywordRefundCount', 'Tickets_KeywordDamageCount',
                                                         'Tickets_KeywordMissingCount', 'AvgTicketSentimentScore', 'MinTicketSentimentScore'])
    else:
        np.random.seed(42) # For reproducibility of random assignments
        email_to_simulated_customer_id_map = {
            email: np.random.choice(unique_retail_customers)
            for email in unique_ticket_emails
        }
        df_tickets_processed['Simulated_CustomerID'] = df_tickets_processed['Customer Email'].map(email_to_simulated_customer_id_map)
        df_tickets_processed.dropna(subset=['Simulated_CustomerID'], inplace=True)
        df_tickets_processed['Simulated_CustomerID'] = df_tickets_processed['Simulated_CustomerID'].astype(int)

        # --- NLP Features from df_tickets_processed ---
        print("    Extracting NLP features from ticket data (simplified)...")
        def count_keywords(text, keywords_list):
            count = 0
            for keyword in keywords_list:
                count += len(re.findall(r'\b' + re.escape(keyword) + r'\b', text.lower()))
            return count

        def get_basic_sentiment_score(text):
            text_lower = text.lower()
            positive_sentiment_terms = ['happy', 'satisfied', 'resolved', 'excellent', 'great', 'thank you']
            negative_sentiment_terms = ['bad', 'poor', 'unhappy', 'frustrated', 'terrible', 'issue', 'problem', 'unresolved', 'damaged', 'broken', 'missing', 'refund', 'return']
            pos_count = count_keywords(text_lower, positive_sentiment_terms)
            neg_count = count_keywords(text_lower, negative_sentiment_terms)
            if (pos_count + neg_count) == 0: return 0.0
            return (pos_count - neg_count) / (pos_count + neg_count)

        refund_keywords = ['refund', 'return', 'cancel', 'money back', 'send back', 'credit']
        damage_keywords = ['damaged', 'broken', 'faulty', 'defective', 'not working', 'malfunction']
        missing_keywords = ['missing', 'never arrived', 'lost package', 'not received', 'where is my order']

        df_tickets_processed['Keywords_RefundCount'] = df_tickets_processed['Ticket Description'].apply(lambda x: count_keywords(x, refund_keywords))
        df_tickets_processed['Keywords_DamageCount'] = df_tickets_processed['Ticket Description'].apply(lambda x: count_keywords(x, damage_keywords))
        df_tickets_processed['Keywords_MissingCount'] = df_tickets_processed['Ticket Description'].apply(lambda x: count_keywords(x, missing_keywords))
        df_tickets_processed['TicketSentimentScore'] = df_tickets_processed['Ticket Description'].apply(get_basic_sentiment_score)

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


    # Combine features
    customer_retail_summary.rename(columns={'CustomerID': 'Simulated_CustomerID'}, inplace=True)
    final_combined_df = pd.merge(
        customer_retail_summary,
        customer_ticket_summary,
        on='Simulated_CustomerID',
        how='left'
    )
    final_combined_df.fillna({
        'TotalTickets': 0, 'Tickets_TypeRefundRequest': 0, 'Tickets_TypeDeliveryIssue': 0,
        'Tickets_TypeProductInquiry': 0, 'Tickets_KeywordRefundCount': 0,
        'Tickets_KeywordDamageCount': 0, 'Tickets_KeywordMissingCount': 0,
        'AvgTicketSentimentScore': 0.0, 'MinTicketSentimentScore': 0.0
    }, inplace=True)

    # Define the list of features for the model
    features = [col for col in final_combined_df.columns if col not in [
        'Simulated_CustomerID', 'FirstTransactionDate', 'LastTransactionDate', 'is_refund_abuser'
    ]]

    print("  Feature engineering and data combination complete.")
    return final_combined_df, features

# --- Dummy Model Class (to be pickled) ---
# In a real scenario, this would be a trained LightGBM model
class DummyModel:
    def predict_proba(self, X):
        # Simulate probabilities based on some simple heuristic logic for demonstration
        # This should match how a real model would produce probabilities
        if not X.empty:
            # Example heuristic for dummy probability: higher refund rate and negative sentiment increase probability
            prob_abuser = (X['RefundRate_by_Items'] * 0.7) + (X['MinTicketSentimentScore'] * -0.2) + (X['Tickets_TypeRefundRequest'] * 0.1)
            # Clip probabilities to be within [0, 1]
            prob_abuser = np.clip(prob_abuser, 0.05, 0.95) # Ensure it's not exactly 0 or 1
            return np.column_stack((1 - prob_abuser, prob_abuser))
        return np.array([[0.5, 0.5]]) # Default if no data

    def predict(self, X):
        # Predict class based on a threshold (e.g., 0.5)
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)

# --- Main Script Execution for Model Training and Saving ---

if __name__ == "__main__":
    MODEL_PATH = "model.pkl"
    RETAIL_FILE = "OnlineRetail.csv"
    TICKETS_FILE = "customer_support_tickets.csv"

    print("--- Starting Conceptual Model Training Script ---")

    # Load and preprocess data
    df_retail_raw = load_data(RETAIL_FILE, encoding='ISO-8859-1')
    df_tickets_raw = load_data(TICKETS_FILE)

    if df_retail_raw is None or df_tickets_raw is None:
        print("Required data files not loaded. Exiting.")
        exit()

    df_retail_processed = preprocess_retail_data(df_retail_raw)
    df_tickets_processed = preprocess_tickets_data(df_tickets_raw)

    if df_retail_processed is None or df_tickets_processed is None:
        print("Data preprocessing failed. Exiting.")
        exit()

    # Engineer and combine features
    final_combined_df, features_list = engineer_and_combine_features(df_retail_processed, df_tickets_processed)

    if final_combined_df is None or features_list is None:
        print("Feature engineering and combination failed. Exiting.")
        exit()

    print(f"\nShape of final combined DataFrame: {final_combined_df.shape}")
    print(f"Number of features for model: {len(features_list)}")

    # Define a conceptual target variable (as used in Streamlit app's feature engineering)
    # This column is NOT passed to the model features, but is used to define 'training data'
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

    # --- Conceptual Model Training and Saving ---
    print("\n--- Training and Saving Conceptual Model ---")

    # In a real scenario, you would train your LightGBM model here:
    X = final_combined_df[features_list]
    y = final_combined_df['is_refund_abuser']
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = lgb.LGBMClassifier(objective='binary', random_state=42)
    model.fit(X_train, y_train)

    conceptual_model = DummyModel() # Instantiate the DummyModel

    try:
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump(conceptual_model, f)
        print(f"Conceptual model successfully saved to '{MODEL_PATH}'.")
        print("\n**Next Steps:** Place this 'model.pkl' file in the same directory as your Streamlit app.")
        print("Then run your Streamlit app: `streamlit run your_app_name.py`")
    except Exception as e:
        print(f"Error saving conceptual model: {e}")

    print("\n--- Conceptual Model Training Script Finished ---")
