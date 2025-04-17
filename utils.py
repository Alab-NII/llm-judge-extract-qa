import re
import numpy as np
from scipy.stats import pearsonr

def extract_answer(text):
    match = re.search(r"<ans>(.*?)</ans>", text, re.DOTALL)
    return match.group(1).strip() if match else text


def extract_answer_mistral(text):
    # 
    matches = re.findall(r'<ans>\s*([\s\S]*?)\s*</ans>', text, re.IGNORECASE)
    # 
    return matches[1].strip() if len(matches) > 1 else ""


def extract_answer_mistral_judge(text):
    # 
    match = re.findall(r'<ans>(.*?)</ans>', text.strip())
    if match:
        return match[-1].strip()  
    return ""  


def convert_list2dict(data, key="_id"):
    """
        Converts a list of dictionaries into a dictionary indexed by a specified key.
    """
    return {item[key]: item for item in data if key in item}


def pearson_correlation(list1, list2):
    """
    Calculate the Pearson correlation coefficient between two lists.

    Parameters:
    list1 (list): First list of numerical values
    list2 (list): Second list of numerical values

    Returns:
    float: Pearson correlation coefficient
    """
    if len(list1) != len(list2):
        raise ValueError("Both lists must have the same length.")
    
    # Convert lists to numpy arrays to handle calculations
    list1 = np.array(list1)
    list2 = np.array(list2)
    
    # Calculate Pearson correlation using scipy's pearsonr
    correlation, _ = pearsonr(list1, list2)
    
    return correlation

