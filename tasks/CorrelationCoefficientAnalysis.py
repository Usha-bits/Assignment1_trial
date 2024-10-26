import pandas as pd
# import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
# import seaborn as sns

class EarthquakeCorrelationCoefficient:
    def __init__(self, data):
        self.data = data
        
    def prepare_analysis_columns(self):
        """
        Define columns for correlation coefficient analysis
        """
        return {
            'target_variables': [
                'magnitude',    # Primary target - earthquake magnitude
                'mmi',         # Secondary target - Modified Mercalli Intensity
                'sig'          # Secondary target - earthquake significance
            ],
            
            'predictor_variables': {
                'ground_motion': [
                    'depth',        # Depth of earthquake
                    'cdi',          # Community Decimal Intensity
                    'felt'          # Number of felt reports
                ],
                
                'location_metrics': [
                    'latitude',     # Geographic latitude
                    'longitude',    # Geographic longitude
                    'distanceKM',   # Distance from epicenter
                    'dmin'          # Minimum distance to stations
                ],
                
                'technical_measurements': [
                    'gap',          # Azimuthal gap
                    'rms',          # Root mean square travel time residual
                    'nst'           # Number of seismic stations
                ]
            }
        }

    def calculate_correlation_coefficients(self, target_variable):
        """
        Calculate correlation coefficients and p-values for a target variable
        """
        columns = self.prepare_analysis_columns()
        predictor_vars = sum(columns['predictor_variables'].values(), [])
        
        correlation_results = []
        
        for predictor in predictor_vars:
            # Remove missing values
            valid_data = self.data[[target_variable, predictor]].dropna()
            
            if len(valid_data) > 1:  # Check if we have enough data
                # Calculate Pearson correlation coefficient and p-value
                corr_coef, p_value = stats.pearsonr(
                    valid_data[target_variable],
                    valid_data[predictor]
                )
                
                # Calculate Spearman correlation for non-linear relationships
                spearman_coef, spearman_p = stats.spearmanr(
                    valid_data[target_variable],
                    valid_data[predictor]
                )
                
                # Calculate R-squared value
                r_squared = corr_coef ** 2
                
                correlation_results.append({
                    'predictor': predictor,
                    'pearson_correlation': corr_coef,
                    'pearson_p_value': p_value,
                    'spearman_correlation': spearman_coef,
                    'spearman_p_value': spearman_p,
                    'r_squared': r_squared,
                    'sample_size': len(valid_data)
                })
        
        return pd.DataFrame(correlation_results)

    def sort_correlations_by_strength(self, correlation_results):
        """
        Sort correlations by absolute strength
        """
        correlation_results['abs_correlation'] = abs(
            correlation_results['pearson_correlation']
        )
        
        sorted_results = correlation_results.sort_values(
            'abs_correlation', 
            ascending=False
        ).drop('abs_correlation', axis=1)
        
        return sorted_results

    def analyze_significance_levels(self, correlation_results):
        """
        Analyze statistical significance at different levels
        """
        significance_levels = {
            'highly_significant': correlation_results[
                correlation_results['pearson_p_value'] < 0.01
            ],
            'significant': correlation_results[
                (correlation_results['pearson_p_value'] >= 0.01) & 
                (correlation_results['pearson_p_value'] < 0.05)
            ],
            'marginally_significant': correlation_results[
                (correlation_results['pearson_p_value'] >= 0.05) & 
                (correlation_results['pearson_p_value'] < 0.1)
            ],
            'not_significant': correlation_results[
                correlation_results['pearson_p_value'] >= 0.1
            ]
        }
        
        return significance_levels

    def visualize_correlation_strengths(self, correlation_results, target_variable):
        """
        Create visualization of correlation strengths
        """
        plt.figure(figsize=(12, 6))
        
        # Plot correlation coefficients
        bars = plt.barh(
            correlation_results['predictor'],
            correlation_results['pearson_correlation']
        )
        
        # Color bars based on significance
        for i, bar in enumerate(bars):
            if correlation_results.iloc[i]['pearson_p_value'] < 0.01:
                bar.set_color('darkred')
            elif correlation_results.iloc[i]['pearson_p_value'] < 0.05:
                bar.set_color('red')
            else:
                bar.set_color('lightgray')
        
        plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        plt.title(f'Correlation Coefficients with {target_variable}')
        plt.xlabel('Correlation Coefficient')
        
        # Add significance level annotations
        for i, row in correlation_results.iterrows():
            plt.text(
                row['pearson_correlation'],
                i,
                f"p={row['pearson_p_value']:.3f}",
                va='center'
            )
        
        plt.tight_layout()
        return plt.gcf()

    def generate_detailed_report(self):
        """
        Generate detailed correlation coefficient analysis report
        """
        columns = self.prepare_analysis_columns()
        report = {}
        
        for target in columns['target_variables']:
            # Calculate correlations
            correlations = self.calculate_correlation_coefficients(target)
            
            # Sort by strength
            sorted_correlations = self.sort_correlations_by_strength(correlations)
            
            # Analyze significance
            significance_analysis = self.analyze_significance_levels(
                sorted_correlations
            )
            
            report[target] = {
                'correlations': sorted_correlations,
                'significance_analysis': significance_analysis
            }
            
            # Create visualization
            self.visualize_correlation_strengths(sorted_correlations, target)
            plt.show()
        
        return report

def main():
    # Load earthquake data
    data = pd.read_csv("D:/Usha/Prefect/Assignment1_trial/data/earthquakes.csv")
    
    # Initialize analyzer
    analyzer = EarthquakeCorrelationCoefficient(data)
    
    # Generate detailed report
    report = analyzer.generate_detailed_report()
    
    # Print results for magnitude correlations
    print("\nCorrelations with Magnitude:")
    print(report['magnitude']['correlations'])
    
    # Print significant correlations
    print("\nHighly Significant Correlations:")
    print(report['magnitude']['significance_analysis']['highly_significant'])

if __name__ == "__main__":
    main()