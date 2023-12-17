import numpy as np

# Calculate percentiles with variable granularity
def calculate_percentiles(data, column_name, granularity, interpolation='linear'):
    """
    Calculate percentiles with variable granularity, including sub-integer granularity.

    :param data: DataFrame containing the data.
    :param column_name: The name of the column to calculate the percentiles for.
    :param granularity: The granularity for the percentiles (e.g., 0.1 for 0.1%, 5 for 5%).
    :return: Dictionary of percentiles.
    """
    if column_name not in data.columns:
        raise ValueError(f"Column {column_name} does not exist in the data.")
    
    if granularity <= 0 or granularity > 100:
        raise ValueError("Granularity must be between 0 (exclusive) and 100 (inclusive).")
    
    # Sort the data
    sorted_values = data[column_name].sort_values().reset_index(drop=True)
    
    percentiles = {
        p: np.percentile(sorted_values, p, interpolation=interpolation)
        for p in np.arange(0, 100 + granularity, granularity)
    }
    return percentiles

def ssd(A,B):
    squares = (A - B) ** 2
    return np.sum(squares)