import numpy as np
def rmspe(y_true, y_pred):
    """
    Calculate Root Mean Squared Percentage Error (RMSPE).
    
    Parameters:
    y_true (array-like): Array of true target values.
    y_pred (array-like): Array of predicted target values.
    
    Returns:
    float: RMSPE value.
    """
    assert len(y_true) == len(y_pred), "Length of y_true and y_pred must be the same."
    
    # Exclude cases where y_true is zero to avoid division by zero
    mask = y_true != 0
    
    # Calculate percentage error
    pe = (y_true - y_pred) / y_true
    pe = np.abs(pe[mask])
    
    # Calculate squared percentage error
    spe = pe ** 2
    
    # Calculate mean squared percentage error
    mspe = np.mean(spe)
    
    # Calculate RMSPE
    rmspe = np.sqrt(mspe)
    
    return rmspe