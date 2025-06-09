import streamlit as st
import pandas as pd
import numpy as np
import re # For regular expressions in NLP-like tasks
import pickle # For saving and loading models
import shap # Required for SHAP plots
import matplotlib.pyplot as plt # Required for plotting, will save to file
import os # For path operations

# Set Streamlit page configuration
st.set_page_config(layout="wide", page_title="Refund Abuse Detection App", page_icon="🕵️‍♀️")

# --- Dummy Model Class (Must be present in both training and prediction scripts) ---
# In a real scenario, this would be your actual trained LightGBM model class/object.
class DummyModel:
    def predict_proba(self, X):
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
# These are kept for the conceptual "training" part.

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

    st.subheader("Feature Engineering & Data Combination (for Conceptual Training)")

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

    # Define the list of features for the model (this list is crucial for consistency)
    features = [
        'TotalUniqueOrders', 'TotalItemsPurchased', 'TotalItemsReturned',
        'TotalPurchaseValue', 'TotalRefundValue', 'RefundRate_by_Items',
        'RefundRate_by_Value', 'AccountAgeDays', 'TotalTickets',
        'Tickets_TypeRefundRequest', 'Tickets_TypeDeliveryIssue',
        'Tickets_TypeProductInquiry', 'Tickets_KeywordRefundCount',
        'Tickets_KeywordDamageCount', 'Tickets_KeywordMissingCount',
        'AvgTicketSentimentScore', 'MinTicketSentimentScore'
    ]


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
    # X = data_df[features]
    # y = data_df[target_col]
    # X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    # model = lgb.LGBMClassifier(objective='binary', random_state=42)
    # model.fit(X_train, y_train)
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
It allows for manual input of customer features for prediction.

**Note:** This is a demonstration. A live production system would involve a genuinely
trained Machine Learning model (e.g., LightGBM) for prediction. Libraries for full ML
model training/inference (like `scikit-learn`, `lightgbm`, `shap`) are not available in this environment.
""")

# Option to train/save conceptual model (for setup purposes)
with st.expander("Conceptual: Train & Save Model (Run Once Locally)"):
    st.markdown("Upload your historical retail and ticket data here to conceptually train the model and save `model.pkl`.")
    col_train1, col_train2 = st.columns(2)
    with col_train1:
        train_retail_file = st.file_uploader("OnlineRetail.csv (Training Data)", type=["csv"], key="train_retail")
    with col_train2:
        train_tickets_file = st.file_uploader("customer_support_tickets.csv (Training Data)", type=["csv"], key="train_tickets")

    df_train_retail_processed = None
    df_train_tickets_processed = None
    final_train_df = None
    features_list_for_training = None # Renamed to avoid confusion with prediction features

    if train_retail_file and train_tickets_file:
        df_train_retail_processed = preprocess_retail_data(load_data(train_retail_file, encoding='ISO-8859-1'))
        df_train_tickets_processed = preprocess_tickets_data(load_data(train_tickets_file))

        if df_train_retail_processed is not None and df_train_tickets_processed is not None:
            # Note: engineer_and_combine_features returns the features list too
            final_train_df, features_list_for_training = engineer_and_combine_features(df_train_retail_processed, df_train_tickets_processed)

            if final_train_df is not None and features_list_for_training is not None:
                st.subheader("Conceptual Model Training Action")
                # The 'is_refund_abuser' column is needed as a conceptual target for training
                if st.button("Train & Save Conceptual Model (model.pkl)"):
                    train_and_save_model_conceptual(final_train_df, features_list_for_training, 'is_refund_abuser', "model.pkl")
                    st.session_state['model_trained_flag'] = True # Set a flag

st.markdown("---")
st.header("Manual Feature Input for Prediction")
st.markdown("Enter values for a customer's features to get a real-time prediction.")

# Define the features list explicitly here for input field generation
# This should match the 'features' list generated by engineer_and_combine_features
predict_features_list = [
    'TotalUniqueOrders', 'TotalItemsPurchased', 'TotalItemsReturned',
    'TotalPurchaseValue', 'TotalRefundValue', 'RefundRate_by_Items',
    'RefundRate_by_Value', 'AccountAgeDays', 'TotalTickets',
    'Tickets_TypeRefundRequest', 'Tickets_TypeDeliveryIssue',
    'Tickets_TypeProductInquiry', 'Tickets_KeywordRefundCount',
    'Tickets_KeywordDamageCount', 'Tickets_KeywordMissingCount',
    'AvgTicketSentimentScore', 'MinTicketSentimentScore'
]

# Create input fields for each feature
input_values = {}
# Default inputs to show a refund abuser
customer_id_input = st.number_input("Customer ID (for display only):", value=10001, format="%d")

st.markdown("##### Retail/Transactional Features:")
col_r1, col_r2, col_r3 = st.columns(3)
with col_r1:
    input_values['TotalUniqueOrders'] = st.number_input("Total Unique Orders", value=15, min_value=0)
    input_values['TotalItemsPurchased'] = st.number_input("Total Items Purchased", value=150, min_value=0)
with col_r2:
    input_values['TotalItemsReturned'] = st.number_input("Total Items Returned", value=40, min_value=0) # High, contributes to high RefundRate_by_Items
    input_values['TotalPurchaseValue'] = st.number_input("Total Purchase Value", value=2500.0, min_value=0.0)
with col_r3:
    input_values['TotalRefundValue'] = st.number_input("Total Refund Value", value=600.0, min_value=0.0) # High, contributes to high RefundRate_by_Value
    input_values['AccountAgeDays'] = st.number_input("Account Age (Days)", value=450, min_value=0)

# Calculate refund rates based on user input for better coherence
calculated_refund_rate_items = input_values['TotalItemsReturned'] / (input_values['TotalItemsPurchased'] if input_values['TotalItemsPurchased'] > 0 else 1)
calculated_refund_rate_value = input_values['TotalRefundValue'] / (input_values['TotalPurchaseValue'] if input_values['TotalPurchaseValue'] > 0 else 1)
# Use calculated values as default, but allow manual override
input_values['RefundRate_by_Items'] = st.number_input(f"Refund Rate by Items (Calculated: {calculated_refund_rate_items:.2f})", value=float(f"{calculated_refund_rate_items:.2f}"), format="%.4f", step=0.01)
input_values['RefundRate_by_Value'] = st.number_input(f"Refund Rate by Value (Calculated: {calculated_refund_rate_value:.2f})", value=float(f"{calculated_refund_rate_value:.2f}"), format="%.4f", step=0.01)


st.markdown("##### Support Ticket/NLP Features:")
col_t1, col_t2, col_t3 = st.columns(3)
with col_t1:
    input_values['TotalTickets'] = st.number_input("Total Tickets", value=5, min_value=0)
    input_values['Tickets_TypeRefundRequest'] = st.number_input("Tickets: Refund Request Type", value=3, min_value=0) # High, strong indicator
with col_t2:
    input_values['Tickets_TypeDeliveryIssue'] = st.number_input("Tickets: Delivery Issue Type", value=0, min_value=0)
    input_values['Tickets_TypeProductInquiry'] = st.number_input("Tickets: Product Inquiry Type", value=0, min_value=0)
with col_t3:
    input_values['Tickets_KeywordRefundCount'] = st.number_input("Keywords: Refund Count", value=5, min_value=0) # High
    input_values['Tickets_KeywordDamageCount'] = st.number_input("Keywords: Damage Count", value=1, min_value=0) # Present
    input_values['Tickets_KeywordMissingCount'] = st.number_input("Keywords: Missing Count", value=2, min_value=0) # Present
    input_values['AvgTicketSentimentScore'] = st.number_input("Avg Ticket Sentiment Score", value=-0.6, min_value=-1.0, max_value=1.0, step=0.1, format="%.1f")
    input_values['MinTicketSentimentScore'] = st.number_input("Min Ticket Sentiment Score", value=-0.9, min_value=-1.0, max_value=1.0, step=0.1, format="%.1f") # Very Negative, strong indicator


# Load the conceptual model
model = load_trained_model("model.pkl")

if model is not None:
    if st.button("Predict Refund Abuse"):
        st.info("Generating conceptual prediction for entered data...")

        # Create DataFrame from input values
        # Ensure the order of columns matches the 'features' list expected by the model
        X_predict = pd.DataFrame([input_values], columns=predict_features_list)
        
        # Ensure no inf/NaN values, fill with 0 or appropriate strategy (as done during training)
        X_predict = X_predict.replace([np.inf, -np.inf], np.nan).fillna(0)

        if not X_predict.empty:
            # Perform conceptual prediction using the loaded dummy model
            y_pred_proba = model.predict_proba(X_predict)[:, 1][0] # Get single probability
            y_pred = model.predict(X_predict)[0] # Get single class prediction

            st.success("Conceptual prediction generated!")
            st.write(f"Customer ID: **{customer_id_input}**")
            st.write(f"Predicted Probability of Abuse: **{y_pred_proba:.4f}**")
            
            if y_pred == 1:
                st.error("Prediction: **FLAGGED** as potential refund abuser!")
            else:
                st.success("Prediction: Customer is **NOT FLAGGED** as a refund abuser.")

            # --- Conceptual SHAP Explanation for the single input ---
            st.subheader("Conceptual Contributing Factors")
            st.markdown("""
            **Important:** This section provides a conceptual explanation of *why* this customer might be flagged/not flagged,
            mimicking the insights from a real SHAP analysis. Actual SHAP plots and precise values
            require a truly trained model and the `shap` library, which are not available in this environment.
            """)

            reasons = []
            # Check the conditions from our conceptual heuristic (or the DummyModel's logic)
            # These explanations are based on the feature values entered by the user
            
            # Factors pushing towards abuse (based on DummyModel's calculation: positive impact on prob_abuser)
            if input_values['RefundRate_by_Items'] > 0.3:
                reasons.append(f"- **High Refund Rate by Items:** ({input_values['RefundRate_by_Items']:.2f}) is a strong indicator of frequent returns.")
            if input_values['MinTicketSentimentScore'] < -0.5:
                reasons.append(f"- **Very Negative Ticket Sentiment:** ({input_values['MinTicketSentimentScore']:.2f}) indicates dissatisfaction or potentially aggressive communication in support tickets.")
            if input_values['Tickets_TypeRefundRequest'] >= 2:
                reasons.append(f"- **Multiple Refund Request Tickets:** ({int(input_values['Tickets_TypeRefundRequest'])}) directly points to a history of seeking refunds.")

            # Additional conceptual factors (aligned with initial heuristic for target variable creation)
            if input_values['RefundRate_by_Value'] > 0.2:
                reasons.append(f"- **High Refund Rate by Value:** ({input_values['RefundRate_by_Value']:.2f}) suggests significant financial value of returns.")
            if input_values['TotalItemsReturned'] > 5:
                reasons.append(f"- **High Total Items Returned:** ({int(input_values['TotalItemsReturned'])}) indicates a consistent pattern of not keeping merchandise.")
            if input_values['Tickets_KeywordMissingCount'] > 0 or input_values['Tickets_KeywordDamageCount'] > 0:
                if input_values['Tickets_KeywordMissingCount'] > 0:
                    reasons.append(f"- **Tickets Mentioning 'Missing' Items:** ({int(input_values['Tickets_KeywordMissingCount'])} counts) can be a common red flag.")
                if input_values['Tickets_KeywordDamageCount'] > 0:
                    reasons.append(f"- **Tickets Mentioning 'Damaged' Items:** ({int(input_values['Tickets_KeywordDamageCount'])} counts) can also be a red flag for false claims.")


            # Categorize the explanation based on the prediction
            if y_pred == 1: # Flagged as abuser
                if reasons:
                    st.markdown("##### Reasons for Flagging as Potential Abuser:")
                    for reason in reasons:
                        st.markdown(reason)
                else: # Fallback if specific conditions weren't met but dummy model still predicted 1
                    st.info(f"Based on the combination of entered features, the conceptual model predicts a high likelihood of abuse (Probability: {y_pred_proba:.4f}), even if specific heuristic conditions weren't individually met strongly. (A real SHAP analysis would offer more nuanced explanations).")
            else: # Not flagged as abuser
                if not reasons: # If no strong "abuse" factors were found
                    st.info(f"Based on the entered values, the conceptual model predicts a low likelihood of abuse (Probability: {y_pred_proba:.4f}). Features indicate a legitimate customer profile.")
                else: # If some "abuse" factors were present but not enough to flag
                    st.info(f"While some factors might slightly push towards abuse, the overall combination of features for this customer suggests a low likelihood of abuse (Probability: {y_pred_proba:.4f}). (A real SHAP analysis would highlight counteracting features).")

            # --- SHAP Plot Generation (Saved to File AND Displayed via st.pyplot) ---
            st.subheader("Visual Explanation (SHAP Force Plot)")
            st.markdown("*(Note: This plot is also saved as a PNG file in your app's directory.)*")

            try:
                # Prepare background data for explainer (a sample of your training data)
                dummy_background_data = pd.DataFrame(np.random.rand(100, len(predict_features_list)), columns=predict_features_list)
                dummy_background_data['RefundRate_by_Items'] = np.random.uniform(0, 0.5, 100)
                dummy_background_data['MinTicketSentimentScore'] = np.random.uniform(-1.0, 0.0, 100)
                dummy_background_data['Tickets_TypeRefundRequest'] = np.random.randint(0, 5, 100)
                dummy_background_data['TotalItemsReturned'] = np.random.randint(0, 100, 100)
                dummy_background_data['TotalPurchaseValue'] = np.random.uniform(100, 5000, 100)
                dummy_background_data['TotalRefundValue'] = np.random.uniform(0, 1000, 100)


                # Explain only the probability of the positive class (abuser)
                explainer = shap.Explainer(lambda x: model.predict_proba(x)[:, 1], dummy_background_data)
                shap_values = explainer(X_predict) # Explain the single customer input

                # Create the plot
                # It's important to clear the current figure before creating a new one
                # to prevent plots from stacking or rendering issues.
                plt.figure(figsize=(10, 6))
                
                # Now pass the correct base_values for the force plot (index 0 for the single output explainer)
                shap.plots.force(shap_values.base_values[0], shap_values.values[0, :], shap_values.data[0, :], matplotlib=True, show=False)
                
                plt.title(f"SHAP Force Plot for Customer ID: {customer_id_input} (Prob: {y_pred_proba:.4f})")
                plt.tight_layout()

                # Save the plot to a file
                plot_filename = f"shap_force_plot_customer_{customer_id_input}.png"
                plt.savefig(plot_filename, bbox_inches='tight', dpi=300)
                st.info(f"SHAP Force Plot saved as '{plot_filename}' in your app's directory.")
                
                # Display the plot in Streamlit using st.pyplot
                st.pyplot(plt.gcf()) # Get current figure and pass to Streamlit
                plt.close('all') # Close all figures to free memory and prevent issues


            except Exception as e:
                st.error(f"Error generating or displaying SHAP plot: {e}. Ensure `shap` and `matplotlib` are correctly installed and you've run `train_model.py`.")
                st.warning("If the issue persists, direct interactive display might be limited; please check the saved PNG file.")

        else:
            st.error("Input data is empty. Please enter feature values.")
else:
    st.info("Model not loaded. Please ensure 'model.pkl' exists in the app's directory.")

