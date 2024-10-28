import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import matplotlib

# Use the TkAgg backend for Matplotlib
matplotlib.use('TkAgg')

class EarthquakeCorrelationAnalysis:
    def __init__(self, data):
        self.data = data
        
    def prepare_correlation_columns(self):
        """
        Define column groups for correlation analysis
        """
        return {
            'primary_metrics': [
                'magnitude',    # Earthquake magnitude
                'depth',        # Depth in kilometers
                'sig',          # Significance of the event
                'mmi'           # Modified Mercalli Intensity
            ],
            
            'intensity_metrics': [
                'cdi',         # Community Decimal Intensity
                'felt',        # Number of felt reports
                'dmin',        # Minimum distance to stations
                'distanceKM'   # Distance from epicenter
            ],
            
            'technical_metrics': [
                'gap',         # Azimuthal gap
                'rms',         # Root mean square travel time residual
                'latitude',    # Latitude coordinate
                'longitude'    # Longitude coordinate
            ]
        }

    def calculate_correlation_matrix(self):
        """
        Calculate correlation matrix for all relevant columns
        """
        columns = sum(self.prepare_correlation_columns().values(), [])
        correlation_matrix = self.data[columns].corr(method='pearson')
        return correlation_matrix

    def calculate_detailed_correlations(self):
        """
        Calculate detailed correlation statistics including p-values
        """
        columns = sum(self.prepare_correlation_columns().values(), [])
        n = len(columns)
        correlations = pd.DataFrame(index=columns, columns=columns)
        p_values = pd.DataFrame(index=columns, columns=columns)
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    corr, p_val = stats.pearsonr(
                        self.data[columns[i]].dropna(),
                        self.data[columns[j]].dropna()
                    )
                    correlations.iloc[i, j] = corr
                    p_values.iloc[i, j] = p_val
                else:
                    correlations.iloc[i, j] = 1.0
                    p_values.iloc[i, j] = 0.0
                    
        return correlations, p_values

    def visualize_correlation_matrix(self, correlation_matrix):
        """
        Create heatmap visualization of correlation matrix
        """
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            correlation_matrix,
            annot=True,
            cmap='coolwarm',
            vmin=-1,
            vmax=1,
            center=0,
            fmt='.2f'
        )
        plt.title('Correlation Matrix of Earthquake Features')
        plt.tight_layout()
        plt.show()  # Show the plot

    def analyze_strong_correlations(self, correlation_matrix, threshold=0.5):
        """
        Identify and analyze strong correlations
        """
        strong_correlations = []
        
        for i in range(len(correlation_matrix.columns)):
            for j in range(i + 1, len(correlation_matrix.columns)):
                corr = correlation_matrix.iloc[i, j]
                if abs(corr) >= threshold:
                    strong_correlations.append({
                        'variable1': correlation_matrix.columns[i],
                        'variable2': correlation_matrix.columns[j],
                        'correlation': corr
                    })
        
        return pd.DataFrame(strong_correlations)

    def correlation_by_group(self):
        """
        Analyze correlations within and between feature groups
        """
        column_groups = self.prepare_correlation_columns()
        group_correlations = {}
        
        # Within-group correlations
        for group_name, columns in column_groups.items():
            group_correlations[f'{group_name}_internal'] = (
                self.data[columns].corr()
            )
        
        # Between-group correlations
        for g1 in column_groups.keys():
            for g2 in column_groups.keys():
                if g1 < g2:
                    cols1 = column_groups[g1]
                    cols2 = column_groups[g2]
                    cross_corr = pd.DataFrame(
                        np.corrcoef(
                            self.data[cols1].values.T,
                            self.data[cols2].values.T
                        )[:len(cols1), len(cols1):],
                        index=cols1,
                        columns=cols2
                    )
                    group_correlations[f'{g1}_vs_{g2}'] = cross_corr
        
        return group_correlations

    def visualize_correlation_by_group(self, group_correlations):
        """
        Create visualization for group-wise correlations
        """
        for group_name, correlation_matrix in group_correlations.items():
            plt.figure(figsize=(10, 8))
            sns.heatmap(
                correlation_matrix,
                annot=True,
                cmap='coolwarm',
                vmin=-1,
                vmax=1,
                center=0,
                fmt='.2f'
            )
            plt.title(f'Correlation Matrix: {group_name}')
            plt.tight_layout()
            plt.show()  # Show each group correlation plot

    def generate_correlation_report(self):
        """
        Generate comprehensive correlation analysis report
        """
        # Calculate basic correlation matrix
        correlation_matrix = self.calculate_correlation_matrix()
        
        # Calculate detailed correlations with p-values
        detailed_corr, p_values = self.calculate_detailed_correlations()
        
        # Identify strong correlations
        strong_correlations = self.analyze_strong_correlations(correlation_matrix)
        
        # Calculate group-wise correlations
        group_correlations = self.correlation_by_group()
        
        report = {
            'correlation_matrix': correlation_matrix,
            'p_values': p_values,
            'strong_correlations': strong_correlations,
            'group_correlations': group_correlations
        }
        
        return report

def main():
    # Load earthquake data
    data = pd.read_csv("D:/Usha/Prefect/Assignment1_trial/data/earthquakes.csv")
    
    # Initialize analyzer
    analyzer = EarthquakeCorrelationAnalysis(data)
    
    # Generate correlation report
    report = analyzer.generate_correlation_report()
    
    # Print strong correlations
    print("Strong Correlations:")
    print(report['strong_correlations'])
    
    # Visualize correlation matrix
    analyzer.visualize_correlation_matrix(report['correlation_matrix'])
    
    # Visualize group correlations
    analyzer.visualize_correlation_by_group(report['group_correlations'])

if __name__ == "__main__":
    main()
