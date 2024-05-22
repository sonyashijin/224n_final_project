from sklearn.metrics import f1_score
import pandas as pd

# Load the uploaded CSV files
ft_no_rag_path = '/mnt/data/ft_no_rag_results.csv'
base_no_rag_path = '/mnt/data/base_no_rag_results (1).csv'

ft_no_rag_df = pd.read_csv(ft_no_rag_path)
base_no_rag_df = pd.read_csv(base_no_rag_path)

# Display the first few rows of each dataframe to understand their structure
ft_no_rag_df.head(), base_no_rag_df.head()
def calculate_em_and_f1(expected, generated):
    # Calculate Exact Match
    em = int(expected.lower() == generated.lower())
    
    # Calculate F1 Score
    expected_tokens = expected.lower().split()
    generated_tokens = generated.lower().split()
    
    common_tokens = set(expected_tokens) & set(generated_tokens)
    
    if len(common_tokens) == 0:
        f1 = 0.0
    else:
        precision = len(common_tokens) / len(generated_tokens)
        recall = len(common_tokens) / len(expected_tokens)
        f1 = 2 * (precision * recall) / (precision + recall)
    
    return em, f1

# Apply the calculation to both dataframes
ft_no_rag_df['EM'], ft_no_rag_df['F1'] = zip(*ft_no_rag_df.apply(lambda row: calculate_em_and_f1(row['Expected Answer'], row['Generated Answer']), axis=1))
base_no_rag_df['EM'], base_no_rag_df['F1'] = zip(*base_no_rag_df.apply(lambda row: calculate_em_and_f1(row['Expected Answer'], row['Generated Answer']), axis=1))

# Calculate the mean EM and F1 scores for both dataframes
ft_no_rag_em_mean = ft_no_rag_df['EM'].mean() * 100
ft_no_rag_f1_mean = ft_no_rag_df['F1'].mean() * 100

base_no_rag_em_mean = base_no_rag_df['EM'].mean() * 100
base_no_rag_f1_mean = base_no_rag_df['F1'].mean() * 100

ft_no_rag_em_mean, ft_no_rag_f1_mean, base_no_rag_em_mean, base_no_rag_f1_mean
