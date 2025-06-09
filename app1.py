import streamlit as st
import pandas as pd
import numpy as np
import re # For regular expressions in NLP-like tasks
import pickle # For saving and loading models

# Conceptual: Import ML libraries (these would be uncommented in a real environment)
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import shap
import matplotlib.pyplot as plt # For SHAP plots

# Set Streamlit page configuration
st.set_page_config(layout="wide", page_title="Refund Abuse Detection App", page_icon="🕵️‍♀️")

# --- Dummy Model Class (Must be present in both training and prediction scripts) ---
# In a real scenario, this would be your actual trained LightGBM model class/object.
class DummyModel:
    def predict_proba(self, X):
        # Simulate probabilities based on some simple heuristic logic for demonstration
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


# --- Helper Functions for Data Processing (Cached for efficiency) ---

@st.cache_data
def load_data(uploaded_file, encoding='utf-8'):
    """Loads a CSV file into a DataFrame."""
    try:
        df = pd.read_csv(uploaded_file, encoding=encoding)
        st.success(f"'{uploaded_file.name}' loaded successfully.")
        return df
    except Exception as e:
        st.error(f"Error loading '{uploaded_file.name}': {e}")
        return None

@st.cache_data
def preprocess_retail_data(df_retail_raw):
    """Preprocesses the raw OnlineRetail DataFrame."""
    if df_retail_raw is None:
        return None

    df = df_retail_raw.copy()
    original_rows = len(df)

    # Drop rows with missing CustomerID
    df.dropna(subset=['CustomerID'], inplace=True)
    st.info(f"Retail preprocessing: Dropped {original_rows - len(df)} rows with missing CustomerID.")
    df['CustomerID'] = df['CustomerID'].astype(int)

    # Convert InvoiceDate to datetime, coercing errors
    original_rows_after_customer_id = len(df)
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
    df.dropna(subset=['InvoiceDate'], inplace=True)
    st.info(f"Retail preprocessing: Dropped {original_rows_after_customer_id - len(df)} rows with unparsable InvoiceDate.")

    # Identify returns and calculate quantities/line totals
    df['IsReturn'] = df['InvoiceNo'].astype(str).str.startswith('C') | (df['Quantity'] < 0)
    df['PurchaseQuantity'] = df['Quantity'].apply(lambda x: x if x > 0 else 0)
    df['ReturnQuantity'] = df['Quantity'].apply(lambda x: abs(x) if x < 0 else 0)
    df['LineTotal'] = df['Quantity'] * df['UnitPrice']
    return df

@st.cache_data
def preprocess_tickets_data(df_tickets_raw):
    """Preprocesses the raw customer_support_tickets DataFrame."""
    if df_tickets_raw is None:
        return None

    df = df_tickets_raw.copy()
    df['Date of Purchase'] = pd.to_datetime(df['Date of Purchase'])
    df['Ticket Description'] = df['Ticket Description'].astype(str).fillna('')
    return df

@st.cache_data
def engineer_and_combine_features(df_retail_processed, df_tickets_processed):
    """
    Engineers structured and NLP features and combines them.
    Returns the final combined DataFrame and the list of features for the model.
    """
    if df_retail_processed is None or df_tickets_processed is None:
        return None, None

    st.subheader("Feature Engineering & Data Combination")

    # --- Structured Features from df_retail_processed ---
    st.write("Extracting structured features from retail data...")
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
    st.write("Simulating CustomerID linkage between datasets...")
    unique_retail_customers = customer_retail_summary['CustomerID'].unique()
    unique_ticket_emails = df_tickets_processed['Customer Email'].unique()

    # Handle case where retail customers might be empty after dropping NaNs
    if len(unique_retail_customers) == 0:
        st.warning("No valid CustomerIDs in retail data after preprocessing. Cannot link tickets. Skipping ticket features.")
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
        st.write("Extracting NLP features from ticket data (simplified)...")
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

    st.write("Combined Features (Sample):")
    st.dataframe(final_combined_df[features].head())
    st.success("Feature engineering and data combination complete.")
    return final_combined_df, features


# --- Conceptual Model Training and Saving Function ---
# This function would typically be run once offline to train and save the model.
# Moved outside the main Streamlit flow to emphasize it's an 'offline' process.
def train_and_save_model_conceptual(data_df, features, target_col, model_path="model.pkl"):
    st.warning("Conceptual: This function simulates model training and saving.")
    st.info(f"Training a conceptual LightGBM model and saving it to '{model_path}' using `pickle`...")
    
    # --- Conceptual Model Training (dummy for demonstration) ---
    # In a real scenario, you would train your LightGBM model here:
    X = data_df[features]
    y = data_df[target_col]
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = lgb.LGBMClassifier(objective='binary', random_state=42)
    model.fit(X_train, y_train)
    # --- End Conceptual Model Training ---

    conceptual_model = DummyModel() # Create an instance of the DummyModel

    # Save the conceptual model using pickle
    try:
        with open(model_path, 'wb') as f:
            pickle.dump(conceptual_model, f)
        st.success(f"Conceptual model saved to '{model_path}'.")
        st.info("In a real app, you would now download this model.pkl and place it in the same directory as your Streamlit app.")
    except Exception as e:
        st.error(f"Error saving conceptual model: {e}. Check file permissions or disk space.")


# --- Conceptual Model Loading Function ---
@st.cache_resource # Cache the model loading to avoid reloading on every rerun
def load_trained_model(model_path="model.pkl"):
    st.warning("Conceptual: This function loads a pre-trained model.")
    try:
        # Load the model using pickle
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        st.success(f"Conceptual model loaded from '{model_path}'.")
        return model
    except FileNotFoundError:
        st.error(f"Model file '{model_path}' not found. Please ensure it has been conceptually trained and saved in the same directory as your Streamlit app.")
        st.info("You would typically run a separate script to train and save your model.pkl file first.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


# --- Main Streamlit Application ---

st.title("🕵️‍♀️ Multi-Modal Refund Abuse Detection")
st.markdown("""
This app demonstrates a conceptual pipeline for detecting refund abuse using both
**structured retail transaction data** and **unstructured customer support ticket text**.
It assumes a model has been *conceptually* trained and saved as `model.pkl`.

**Note:** This is a demonstration. A live production system would involve a genuinely
trained Machine Learning model (e.g., LightGBM) for prediction. Libraries for full ML
model training/inference (like `scikit-learn`, `lightgbm`, `shap`) are not available in this environment.
""")

st.markdown("---")
st.header("Predict Refund Abuse for Unseen Data")
st.markdown("Upload new retail and ticket data for real-time abuse prediction using the (conceptually) pre-trained model.")

col1, col2 = st.columns(2)
with col1:
    predict_retail_file = st.file_uploader("Upload OnlineRetail.csv (New Data)", type=["csv"], key="predict_retail")
with col2:
    predict_tickets_file = st.file_uploader("Upload customer_support_tickets.csv (New Data)", type=["csv"], key="predict_tickets")

df_predict_retail_processed = None
df_predict_tickets_processed = None
final_predict_df = None
predict_features_list = None


if predict_retail_file and predict_tickets_file:
    df_predict_retail_processed = preprocess_retail_data(load_data(predict_retail_file, encoding='ISO-8859-1'))
    df_predict_tickets_processed = preprocess_tickets_data(load_data(predict_tickets_file))

    if df_predict_retail_processed is not None and df_predict_tickets_processed is not None:
        final_predict_df, predict_features_list = engineer_and_combine_features(df_predict_retail_processed, df_predict_tickets_processed)

        if final_predict_df is not None and predict_features_list is not None:
            st.subheader("Prediction Results")

            # Load the conceptual model when the app starts or files are processed
            model = load_trained_model("model.pkl")

            if model is not None:
                if st.button("Generate Predictions"):
                    st.info("Generating conceptual predictions for unseen data...")
                    
                    # Prepare new data for prediction
                    X_new = final_predict_df[predict_features_list].copy()
                    # Ensure no inf/NaN values, fill with 0 or appropriate strategy
                    X_new = X_new.replace([np.inf, -np.inf], np.nan).fillna(0)

                    if not X_new.empty:
                        # Perform conceptual prediction using the loaded dummy model
                        y_pred_proba = model.predict_proba(X_new)[:, 1]
                        y_pred = model.predict(X_new)

                        final_predict_df['Predicted_Probability'] = y_pred_proba
                        final_predict_df['Predicted_Abuser'] = y_pred

                        st.success("Conceptual predictions generated!")
                        st.write(f"Total new customers processed: **{len(final_predict_df)}**")
                        st.write(f"Flagged as potential abusers: **{final_predict_df['Predicted_Abuser'].sum()}**")

                        st.write("### New Data with Predictions (Sample)")
                        st.dataframe(final_predict_df[['Simulated_CustomerID', 'Predicted_Probability', 'Predicted_Abuser']].head(10))

                        # --- Conceptual SHAP Explanation ---
                        st.subheader("Conceptual SHAP-like Explanation for Flagged Customers")
                        st.markdown("""
                        **Important:** This section provides a conceptual explanation of *why* customers might be flagged,
                        mimicking the insights from a real SHAP analysis. Actual SHAP plots and precise values
                        require a truly trained model and the `shap` library, which are not available in this environment.
                        """)

                        flagged_new_customers = final_predict_df[final_predict_df['Predicted_Abuser'] == 1].copy()

                        if not flagged_new_customers.empty:
                            st.write("Top 5 flagged customers and their conceptual contributing factors:")
                            for i, row in flagged_new_customers.head(5).iterrows():
                                st.markdown(f"#### Customer ID: {int(row['Simulated_CustomerID'])}")
                                st.write(f"Predicted Probability of Abuse: **{row['Predicted_Probability']:.4f}**")
                                reasons = []

                                # Conceptual reasons based on high/low feature values and dummy prediction logic
                                if row['RefundRate_by_Items'] > 0.3:
                                    reasons.append(f"- **High Refund Rate by Items:** ({row['RefundRate_by_Items']:.2f}) - Suggests frequent returns.")
                                if row['MinTicketSentimentScore'] < -0.5:
                                    reasons.append(f"- **Very Negative Ticket Sentiment:** ({row['MinTicketSentimentScore']:.2f}) - Indicates strong dissatisfaction or aggressive communication.")
                                if row['Tickets_TypeRefundRequest'] >= 2:
                                    reasons.append(f"- **Multiple Refund Request Tickets:** ({int(row['Tickets_TypeRefundRequest'])}) - Direct indication of refund-related interactions.")
                                if row['Tickets_KeywordMissingCount'] > 0 or row['Tickets_KeywordDamageCount'] > 0:
                                    reasons.append(f"- **Tickets Mentioning Missing/Damaged Items:** (Missing: {int(row['Tickets_KeywordMissingCount'])}, Damaged: {int(row['Tickets_KeywordDamageCount'])}) - Common red flags for abuse.")
                                if row['TotalItemsReturned'] > 5:
                                     reasons.append(f"- **High Number of Items Returned:** ({int(row['TotalItemsReturned'])}) - Consistent pattern of merchandise not being kept.")

                                if reasons:
                                    st.markdown("##### Conceptual Contributing Factors:")
                                    for reason in reasons:
                                        st.markdown(reason)
                                else:
                                    st.info("No dominant conceptual factors found for this flagged customer based on simplified conditions.")
                        else:
                            st.info("No new customers were flagged as potential refund abusers in this dataset.")
                    else:
                        st.info("The processed unseen data is empty. No predictions could be made.")
            else:
                st.info("Click 'Generate Predictions' to see the conceptual results.")
