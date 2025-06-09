import pandas as pd
import numpy as np
import re # For regular expressions in NLP-like tasks
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import matplotlib.pyplot as plt


# --- Step 1: Data Loading and Preprocessing ---
print("--- Step 1: Data Loading and Preprocessing ---")

# Load Datasets
try:
    # OnlineRetail.csv often requires 'ISO-8859-1' encoding due to character issues
    df_retail = pd.read_csv('OnlineRetail.csv', encoding='ISO-8859-1')
    df_tickets = pd.read_csv('customer_support_tickets.csv')
    print("Datasets loaded successfully.")
except FileNotFoundError:
    print("One or more files not found. Please ensure 'OnlineRetail.csv' and 'customer_support_tickets.csv' are uploaded.")
    # In a live environment, you might exit or raise an error here.
    exit()

# Preprocessing for OnlineRetail.csv (Structured Data)
print("\nPreprocessing OnlineRetail.csv...")
# Drop rows with missing CustomerID as they cannot be linked to customer behavior
df_retail.dropna(subset=['CustomerID'], inplace=True)
df_retail['CustomerID'] = df_retail['CustomerID'].astype(int) # Convert CustomerID to integer

# Convert InvoiceDate to datetime objects for time-based analysis
# Use errors='coerce' to handle mixed or inconsistent date formats by converting invalid parses to NaT
df_retail['InvoiceDate'] = pd.to_datetime(df_retail['InvoiceDate'], errors='coerce')

# Drop rows where InvoiceDate could not be parsed (became NaT)
df_retail.dropna(subset=['InvoiceDate'], inplace=True)

# Identify returns.
# Transactions with InvoiceNo starting with 'C' explicitly denote a cancellation/return.
# Negative quantity also indicates a return. We'll use both for robustness.
df_retail['IsReturn'] = df_retail['InvoiceNo'].astype(str).str.startswith('C') | (df_retail['Quantity'] < 0)

# Calculate the actual quantity for purchases and absolute quantity for returns
df_retail['PurchaseQuantity'] = df_retail['Quantity'].apply(lambda x: x if x > 0 else 0)
df_retail['ReturnQuantity'] = df_retail['Quantity'].apply(lambda x: abs(x) if x < 0 else 0)

# Calculate line total (positive for purchases, negative for returns)
df_retail['LineTotal'] = df_retail['Quantity'] * df_retail['UnitPrice']

print("OnlineRetail.csv preprocessed. Sample:")
print(df_retail.head(2))

# Preprocessing for customer_support_tickets.csv (Unstructured Data)
print("\nPreprocessing customer_support_tickets.csv...")
# Convert 'Date of Purchase' to datetime if needed for any time-based linkage
df_tickets['Date of Purchase'] = pd.to_datetime(df_tickets['Date of Purchase'])

# Ensure Ticket Description is string type and fill any potential NaNs
df_tickets['Ticket Description'] = df_tickets['Ticket Description'].astype(str).fillna('')

print("customer_support_tickets.csv preprocessed. Sample:")
print(df_tickets.head(2))

print("\n--- Step 2: Feature Engineering ---")

# --- Structured Features from OnlineRetail.csv ---
print("\nEngineering structured features from retail data...")
customer_retail_summary = df_retail.groupby('CustomerID').agg(
    TotalUniqueOrders=('InvoiceNo', 'nunique'),
    TotalItemsPurchased=('PurchaseQuantity', 'sum'),
    TotalItemsReturned=('ReturnQuantity', 'sum'),
    TotalPurchaseValue=('LineTotal', lambda x: x[x > 0].sum()), # Sum of positive LineTotal
    TotalRefundValue=('LineTotal', lambda x: x[x < 0].abs().sum()), # Sum of absolute negative LineTotal
    FirstTransactionDate=('InvoiceDate', 'min'),
    LastTransactionDate=('InvoiceDate', 'max')
).reset_index()

# Calculate derived features for each customer
# Avoid division by zero by replacing 0 with NaN for division, then filling NaN with 0
customer_retail_summary['RefundRate_by_Items'] = customer_retail_summary['TotalItemsReturned'] / customer_retail_summary['TotalItemsPurchased'].replace(0, np.nan)
customer_retail_summary['RefundRate_by_Items'].fillna(0, inplace=True)

customer_retail_summary['RefundRate_by_Value'] = customer_retail_summary['TotalRefundValue'] / customer_retail_summary['TotalPurchaseValue'].replace(0, np.nan)
customer_retail_summary['RefundRate_by_Value'].fillna(0, inplace=True)

customer_retail_summary['AccountAgeDays'] = (customer_retail_summary['LastTransactionDate'] - customer_retail_summary['FirstTransactionDate']).dt.days.fillna(0)

print("Sample Customer Retail Summary (Structured Features):")
print(customer_retail_summary.head(2))

# --- Simulated CustomerID Mapping for Linking ---
# This is a crucial step to bridge the two datasets given no direct common ID.
# In a real application, you'd use a single, universal CustomerID present in both systems.
# Here, we simulate a linkage by mapping ticket emails to existing retail CustomerIDs.
# This approach is illustrative and its effectiveness depends on the overlap and distribution
# of customer identifiers in a real scenario.

print("\nSimulating CustomerID mapping for linking datasets...")
unique_retail_customers = customer_retail_summary['CustomerID'].unique()
unique_ticket_emails = df_tickets['Customer Email'].unique()

np.random.seed(42) # For reproducibility of random assignments

if len(unique_retail_customers) == 0:
    print("No valid CustomerIDs found in OnlineRetail.csv after preprocessing. Cannot simulate linkage.")
    exit()

email_to_simulated_customer_id_map = {
    email: np.random.choice(unique_retail_customers)
    for email in unique_ticket_emails
}

df_tickets['Simulated_CustomerID'] = df_tickets['Customer Email'].map(email_to_simulated_customer_id_map)

df_tickets.dropna(subset=['Simulated_CustomerID'], inplace=True)
df_tickets['Simulated_CustomerID'] = df_tickets['Simulated_CustomerID'].astype(int)

print("Sample Tickets with Simulated CustomerID:")
print(df_tickets[['Customer Email', 'Simulated_CustomerID']].head(2))


# --- NLP Features from customer_support_tickets.csv ---
print("\nEngineering NLP features from ticket data...")

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
    negative_sentiment_terms = ['bad', 'poor', 'unhappy', 'frustrated', 'terrible', 'issue', 'problem', 'unresolved', 'damaged', 'broken', 'missing', 'refund', 'return'] # Including some fraud-related keywords as negative
    pos_count = count_keywords(text_lower, positive_sentiment_terms)
    neg_count = count_keywords(text_lower, negative_sentiment_terms)

    if (pos_count + neg_count) == 0:
        return 0.0 # Neutral if no sentiment words found
    return (pos_count - neg_count) / (pos_count + neg_count) # Simple score -1 to 1

# Define keywords for fraud-related themes
refund_keywords = ['refund', 'return', 'cancel', 'money back', 'send back', 'credit']
damage_keywords = ['damaged', 'broken', 'faulty', 'defective', 'not working', 'malfunction']
missing_keywords = ['missing', 'never arrived', 'lost package', 'not received', 'where is my order']

df_tickets['Keywords_RefundCount'] = df_tickets['Ticket Description'].apply(lambda x: count_keywords(x, refund_keywords))
df_tickets['Keywords_DamageCount'] = df_tickets['Ticket Description'].apply(lambda x: count_keywords(x, damage_keywords))
df_tickets['Keywords_MissingCount'] = df_tickets['Ticket Description'].apply(lambda x: count_keywords(x, missing_keywords))
df_tickets['TicketSentimentScore'] = df_tickets['Ticket Description'].apply(get_basic_sentiment_score)

# Aggregate NLP features per simulated CustomerID
customer_ticket_summary = df_tickets.groupby('Simulated_CustomerID').agg(
    TotalTickets=('Ticket ID', 'count'),
    Tickets_TypeRefundRequest=('Ticket Type', lambda x: (x == 'Refund request').sum()),
    Tickets_TypeDeliveryIssue=('Ticket Type', lambda x: (x == 'Delivery issue').sum()),
    Tickets_TypeProductInquiry=('Ticket Type', lambda x: (x == 'Product inquiry').sum()),
    Tickets_KeywordRefundCount=('Keywords_RefundCount', 'sum'),
    Tickets_KeywordDamageCount=('Keywords_DamageCount', 'sum'),
    Tickets_KeywordMissingCount=('Keywords_MissingCount', 'sum'),
    AvgTicketSentimentScore=('TicketSentimentScore', 'mean'),
    MinTicketSentimentScore=('TicketSentimentScore', 'min') # Min score might indicate most negative interaction
).reset_index()

print("Sample Customer Ticket Summary (NLP-Derived Features):")
print(customer_ticket_summary.head(2))

print("\n--- Step 3: Combining Multi-Modal Data ---")

# Rename CustomerID in retail summary to match for merge
customer_retail_summary.rename(columns={'CustomerID': 'Simulated_CustomerID'}, inplace=True)

# Merge structured features with NLP-derived features on the Simulated_CustomerID
final_combined_df = pd.merge(
    customer_retail_summary,
    customer_ticket_summary,
    on='Simulated_CustomerID',
    how='left' # Keep all customers from retail summary, add ticket data if available
)

# Fill NaN values for customers who have no corresponding tickets in our simulated linkage
# These NaNs result from the 'left' merge when a retail customer has no mapped tickets.
final_combined_df.fillna({
    'TotalTickets': 0,
    'Tickets_TypeRefundRequest': 0,
    'Tickets_TypeDeliveryIssue': 0,
    'Tickets_TypeProductInquiry': 0,
    'Tickets_KeywordRefundCount': 0,
    'Tickets_KeywordDamageCount': 0,
    'Tickets_KeywordMissingCount': 0,
    'AvgTicketSentimentScore': 0.0, # Assume neutral sentiment if no tickets
    'MinTicketSentimentScore': 0.0  # Assume neutral sentiment if no tickets
}, inplace=True)

# Define a Conceptual Target Variable (`is_refund_abuser`)
# This is a simplified heuristic for demonstration purposes ONLY.
# In a real scenario, this would be based on detailed business rules,
# confirmed fraud labels, or expert review.

# Heuristic criteria for 'is_refund_abuser':
# (Condition A) High Refund Rate (by items) AND multiple explicit Refund Request tickets
# OR
# (Condition B) Moderately high Refund Rate (by value) AND high counts of missing/damage keywords in tickets
# OR
# (Condition C) Very low (negative) average sentiment score AND a significant number of items returned

final_combined_df['is_refund_abuser'] = (
    (final_combined_df['RefundRate_by_Items'] > 0.3) & # E.g., more than 30% of items returned
    (final_combined_df['Tickets_TypeRefundRequest'] >= 2) # At least 2 explicit refund request tickets
) | (
    (final_combined_df['RefundRate_by_Value'] > 0.2) & # E.g., more than 20% value refunded
    ( (final_combined_df['Tickets_KeywordMissingCount'] > 0) | (final_combined_df['Tickets_KeywordDamageCount'] > 0) ) # Has tickets with missing or damage claims
) | (
    (final_combined_df['MinTicketSentimentScore'] < -0.5) & # At least one very negative interaction
    (final_combined_df['TotalItemsReturned'] > 5) # And a significant number of items returned
)


# Convert boolean target variable to integer (0 for legitimate, 1 for abuser)
final_combined_df['is_refund_abuser'] = final_combined_df['is_refund_abuser'].astype(int)

print("Sample of Final Combined Multi-Modal Dataset with Target Variable:")
print(final_combined_df[['Simulated_CustomerID', 'TotalItemsPurchased', 'TotalItemsReturned',
                         'RefundRate_by_Items', 'RefundRate_by_Value', 'TotalTickets',
                         'Tickets_TypeRefundRequest', 'Tickets_KeywordMissingCount',
                         'AvgTicketSentimentScore', 'is_refund_abuser']].head())
print(f"\nTotal customers in combined dataset: {len(final_combined_df)}")
print(f"Number of simulated refund abusers: {final_combined_df['is_refund_abuser'].sum()}")
print("\nDistribution of 'is_refund_abuser':")
print(final_combined_df['is_refund_abuser'].value_counts())
print("\nProportion of 'is_refund_abuser':")
print(final_combined_df['is_refund_abuser'].value_counts(normalize=True))

# --- Step 4: Model Training (Conceptual Code) ---
print("\n--- Step 4: Model Training (Conceptual Code) ---")
print("This step requires 'scikit-learn' and 'lightgbm' libraries, which are not available in this environment.")
print("The code below is illustrative of what you would implement in a compatible Python environment.")

# Define features (X) and target (y)
# Exclude identifier columns and date columns from features
features = [col for col in final_combined_df.columns if col not in ['Simulated_CustomerID', 'FirstTransactionDate', 'LastTransactionDate', 'is_refund_abuser']]
print(f"\nExample 'features' list for model training (conceptual):")
print(features)
print(f"Number of features: {len(features)}")


X = final_combined_df[features]
y = final_combined_df['is_refund_abuser']

# # Handle potential infinite or NaN values in features (e.g., from division by zero in real data)
X = X.replace([np.inf, -np.inf], np.nan).fillna(0) # Replace inf with NaN then fill NaNs with 0, or a mean/median

# # Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# # Initialize and train LightGBM Classifier
lgb_clf = lgb.LGBMClassifier(objective='binary', random_state=42)
print("\nTraining LightGBM model...")
lgb_clf.fit(X_train, y_train)
print("LightGBM model training complete.")

# --- Step 5: Model Evaluation (Conceptual Code) ---
print("\n--- Step 5: Model Evaluation (Conceptual Code) ---")
print("This step requires 'scikit-learn' library, which is not available in this environment.")
print("The code below is illustrative.")

y_pred = lgb_clf.predict(X_test)
y_pred_proba = lgb_clf.predict_proba(X_test)[:, 1] # Probability of being the positive class (abuser)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print(f"ROC-AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")

# --- Step 6: Model Explainability (Conceptual Code) ---
print("\n--- Step 6: Model Explainability (Conceptual Code) ---")
print("This step requires 'shap' library, which is not available in this environment.")
print("The code below is illustrative.")

import shap

# # Create a SHAP explainer object for the LightGBM model
explainer = shap.TreeExplainer(lgb_clf)

# # Calculate SHAP values for the test set (or a subset for faster computation)
shap_values = explainer.shap_values(X_test)

# # Visualize global feature importance (e.g., summary plot)
print("\nSHAP Global Feature Importance (Summary Plot - requires matplotlib):")
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False) # For mean absolute SHAP value
#shap.summary_plot(shap_values, X_test, show=False) # For individual SHAP values

# # Visualize individual prediction explanations (e.g., force plot)
print("\nSHAP Explanation for a Single Prediction (Force Plot - requires matplotlib):")
shap.initjs() # For JS visualization in notebooks
shap.force_plot(
    explainer.expected_value, 
    shap_values[0], 
    X_test.iloc[[0]]  # <- DOUBLE brackets to keep it as a DataFrame
)
plt.savefig("force_plot_example.png", bbox_inches='tight')
plt.clf() # Clear the current figure to prevent it from displaying twice

# For a summary plot:
shap.summary_plot(shap_values, X_test, show=False)
plt.savefig("summary_plot_example.png", bbox_inches='tight')
plt.clf()

# --- Step 7: Prediction and Flagging Accounts (Conceptual Code) ---
print("\n--- Step 7: Prediction and Flagging Accounts (Conceptual Code) ---")
print("This step assumes a trained LightGBM model is available.")

# # Example: Simulate new incoming data for a customer
# # Now, based on the 'features' list, here's an example of how you would define 'new_customer_data'
# # with dummy values. Replace these dummy values with actual, computed features for a new customer.

customer_id_for_prediction = 99999 # Assign a dummy customer ID for this example

example_new_customer_values = [
     10,      # TotalUniqueOrders (dummy)
    100,     # TotalItemsPurchased (dummy)
     5,       # TotalItemsReturned (dummy)
     1000.0,  # TotalPurchaseValue (dummy)
     50.0,    # TotalRefundValue (dummy)
     0.05,    # RefundRate_by_Items (dummy)
     0.05,    # RefundRate_by_Value (dummy)
    300,     # AccountAgeDays (dummy)
     3,       # TotalTickets (dummy)
     1,       # Tickets_TypeRefundRequest (dummy)
     0,       # Tickets_TypeDeliveryIssue (dummy)
     0,       # Tickets_TypeProductInquiry (dummy)
     2,       # Tickets_KeywordRefundCount (dummy)
     0,       # Tickets_KeywordDamageCount (dummy)
    1,       # Tickets_KeywordMissingCount (dummy)
     -0.2,    # AvgTicketSentimentScore (dummy)
     -0.8     # MinTicketSentimentScore (dummy)
 ]

# # Create a list of all columns that will be in the new_customer_data DataFrame
# # This includes 'Simulated_CustomerID' plus all the 'features'.
all_columns_for_new_data = ['Simulated_CustomerID'] + features

# # Create a list of all values for the new customer, starting with the ID
all_values_for_new_data = [customer_id_for_prediction] + example_new_customer_values

# # Now create the DataFrame with all necessary columns
new_customer_data = pd.DataFrame([all_values_for_new_data], columns=all_columns_for_new_data)


# # Ensure new_customer_data goes through the exact same feature engineering pipeline
# # as the training data, including the NLP features and structured aggregates.

# # Predict probability of refund abuse
new_customer_pred_proba = lgb_clf.predict_proba(new_customer_data[features])[:, 1]

# # Define a flagging threshold (e.g., 0.7 or determined by business tolerance for false positives/negatives)
flagging_threshold = 0.7

if True: # Replace with `new_customer_pred_proba[0] > flagging_threshold:` in a real scenario
    print(f"\nCustomer ID: {new_customer_data['Simulated_CustomerID'].iloc[0]} - FLAGGED for potential refund abuse.")
    print(f"Predicted probability: {new_customer_pred_proba[0]:.4f}")
else:
    print(f"\nCustomer ID: {new_customer_data['Simulated_CustomerID'].iloc[0]} - Not flagged.")
    print(f"Predicted probability: {new_customer_pred_proba[0]:.4f}")

print("\n--- End of Code Demonstration ---")