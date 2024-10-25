import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
import logging

# Configure standard logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load the dataset
df = pd.read_csv("D:/Usha/Prefect/Assignment1_trial/data/earthquakest.csv")
logger.info("Dataset loaded successfully.")
logger.info(f"DataFrame head:\n{df.head()}")

# Convert date columns to datetime format
df['date'] = pd.to_datetime(df['date'], errors='coerce', dayfirst=True)
logger.info("Converted date columns to datetime format.")

# Summary statistics
logger.info("Summary Statistics:")
logger.info(f"\n{df.describe(include='all')}")
## , datetime_is_numeric=True

# Checking for missing values
logger.info("Missing Values:")
logger.info(f"\n{df.isnull().sum()}")

# Data type information
logger.info("Data Types:")
logger.info(f"\n{df.dtypes}")