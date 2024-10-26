import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

class EarthquakeDataStatistics:
    def __init__(self, data):
        self.data = data
        
    def numerical_statistics(self):
        """
        Calculate basic statistical measures for numerical columns
        """
        # Relevant numerical columns for earthquake analysis
        numerical_columns = [
            'magnitude',    # Earthquake magnitude
            'depth',        # Depth of the earthquake
            'latitude',     # Latitude coordinate
            'longitude',    # Longitude coordinate
            'felt',         # Number of felt reports
            'cdi',          # Community Decimal Intensity
            'mmi',          # Modified Mercalli Intensity
            'sig',          # Significance of the event
            'gap',          # Azimuthal gap
            'rms',          # Root mean square travel time residual
            'dmin',         # Minimum distance to stations
            'distanceKM'    # Distance in kilometers
        ]
        
        # Calculate basic statistics
        basic_stats = self.data[numerical_columns].describe()
        
        # Add additional statistical measures
        basic_stats.loc['variance'] = self.data[numerical_columns].var()
        basic_stats.loc['mode'] = self.data[numerical_columns].mode().iloc[0]
        basic_stats.loc['range'] = self.data[numerical_columns].max() - self.data[numerical_columns].min()
        
        return basic_stats

    def skewness_analysis(self):
        """
        Analyze skewness of numerical columns
        """
        numerical_columns = [
            'magnitude', 'depth', 'felt', 'cdi', 'mmi', 'sig', 
            'gap', 'rms', 'dmin', 'distanceKM'
        ]
        
        skewness_stats = pd.DataFrame()
        
        for column in numerical_columns:
            skewness = stats.skew(self.data[column].dropna())
            skewness_interpretation = self.interpret_skewness(skewness)
            
            skewness_stats.loc[column, 'Skewness'] = skewness
            skewness_stats.loc[column, 'Interpretation'] = skewness_interpretation
            
        return skewness_stats

    def kurtosis_analysis(self):
        """
        Analyze kurtosis of numerical columns
        """
        numerical_columns = [
            'magnitude', 'depth', 'felt', 'cdi', 'mmi', 'sig', 
            'gap', 'rms', 'dmin', 'distanceKM'
        ]
        
        kurtosis_stats = pd.DataFrame()
        
        for column in numerical_columns:
            kurt = stats.kurtosis(self.data[column].dropna())
            kurtosis_interpretation = self.interpret_kurtosis(kurt)
            
            kurtosis_stats.loc[column, 'Kurtosis'] = kurt
            kurtosis_stats.loc[column, 'Interpretation'] = kurtosis_interpretation
            
        return kurtosis_stats

    def missing_value_analysis(self):
        """
        Analyze missing values in all columns
        """
        missing_stats = pd.DataFrame()
        
        # Calculate missing values
        missing_count = self.data.isnull().sum()
        missing_percentage = (missing_count / len(self.data)) * 100
        
        missing_stats['Missing Count'] = missing_count
        missing_stats['Missing Percentage'] = missing_percentage
        missing_stats['Data Type'] = self.data.dtypes
        
        return missing_stats.sort_values('Missing Percentage', ascending=False)

    def visualize_distributions(self):
        """
        Visualize distributions of key numerical columns
        """
        numerical_columns = [
            'magnitude', 'depth', 'felt', 'cdi', 'mmi', 'sig'
        ]
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Distribution of Key Earthquake Metrics')
        
        for idx, column in enumerate(numerical_columns):
            row = idx // 3
            col = idx % 3
            
            # Histogram with KDE
            sns.histplot(data=self.data, x=column, kde=True, ax=axes[row, col])
            axes[row, col].set_title(f'{column} Distribution')
            
        plt.tight_layout()
        plt.show()

    @staticmethod
    def interpret_skewness(skewness):
        """
        Interpret skewness values
        """
        if skewness < -1:
            return 'Highly Negative Skewed'
        elif skewness < -0.5:
            return 'Moderately Negative Skewed'
        elif skewness < 0.5:
            return 'Approximately Symmetric'
        elif skewness < 1:
            return 'Moderately Positive Skewed'
        else:
            return 'Highly Positive Skewed'

    @staticmethod
    def interpret_kurtosis(kurtosis):
        """
        Interpret kurtosis values
        """
        if kurtosis < -1:
            return 'Platykurtic (Light-tailed)'
        elif kurtosis < 1:
            return 'Mesokurtic (Normal-like)'
        else:
            return 'Leptokurtic (Heavy-tailed)'

    def generate_summary_report(self):
        """
        Generate a comprehensive statistical summary report
        """
        report = {
            'basic_statistics': self.numerical_statistics(),
            'skewness_analysis': self.skewness_analysis(),
            'kurtosis_analysis': self.kurtosis_analysis(),
            'missing_value_analysis': self.missing_value_analysis()
        }
        
        return report

# Example usage:
def main():
    # Load your earthquake data
    data = pd.read_csv("D:/Usha/Prefect/Assignment1_trial/data/earthquakes.csv")
    
    # Initialize the statistics analyzer
    analyzer = EarthquakeDataStatistics(data)
    
    # Generate comprehensive report
    report = analyzer.generate_summary_report()
    
    # Print results
    print("Basic Statistical Measures:")
    print(report['basic_statistics'])
    print("\nSkewness Analysis:")
    print(report['skewness_analysis'])
    print("\nKurtosis Analysis:")
    print(report['kurtosis_analysis'])
    print("\nMissing Value Analysis:")
    print(report['missing_value_analysis'])
    
    # Visualize distributions
    analyzer.visualize_distributions()

if __name__ == "__main__":
    main()