import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# from sklearn.preprocessing import KBinsDiscretizer

class EarthquakeBinningAnalysis:
    def __init__(self, data):
        self.data = data
        self.binned_data = {}
        
    def perform_binning(self, column, n_bins=5, method='equal_width'):
        """
        Perform binning on specified columns using different methods
        """
        if method == 'equal_width':
            return pd.cut(self.data[column], bins=n_bins)
        elif method == 'equal_frequency':
            return pd.qcut(self.data[column], q=n_bins)
        else:
            raise ValueError("Method must be 'equal_width' or 'equal_frequency'")

    def magnitude_binning(self, n_bins=5):
        """
        Bin earthquake magnitudes
        Typical ranges: <2 (micro), 2-4 (minor), 4-6 (light), 6-7 (moderate), >7 (major)
        """
        bins = [0, 2, 4, 6, 7, np.inf]
        labels = ['Micro', 'Minor', 'Light', 'Moderate', 'Major']
        
        self.binned_data['magnitude'] = pd.cut(
            self.data['magnitude'],
            bins=bins,
            labels=labels,
            include_lowest=True
        )
        
        return self.calculate_bin_statistics('magnitude')

    def depth_binning(self, n_bins=5):
        """
        Bin earthquake depths
        Shallow: 0-70 km
        Intermediate: 70-300 km
        Deep: >300 km
        """
        bins = [0, 70, 300, np.inf]
        labels = ['Shallow', 'Intermediate', 'Deep']
        
        self.binned_data['depth'] = pd.cut(
            self.data['depth'],
            bins=bins,
            labels=labels,
            include_lowest=True
        )
        
        return self.calculate_bin_statistics('depth')

    def intensity_binning(self):
        """
        Bin earthquake intensities (MMI scale)
        """
        # Modified Mercalli Intensity (MMI) scale bins
        bins = range(0, 13)  # MMI scale goes from I to XII
        labels = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X', 'XI', 'XII']
        
        self.binned_data['mmi'] = pd.cut(
            self.data['mmi'],
            bins=bins,
            labels=labels,
            include_lowest=True
        )
        
        return self.calculate_bin_statistics('mmi')

    def significance_binning(self, n_bins=5):
        """
        Bin significance scores
        """
        return self.perform_binning('sig', n_bins)

    def distance_binning(self, n_bins=5):
        """
        Bin distances to nearest location
        """
        return self.perform_binning('distanceKM', n_bins)

    def calculate_bin_statistics(self, column):
        """
        Calculate statistics for each bin
        """
        stats = pd.DataFrame()
        
        # Basic statistics
        stats['count'] = self.data.groupby(self.binned_data[column]).size()
        stats['percentage'] = (stats['count'] / len(self.data)) * 100
        
        # Additional statistics based on the original values
        stats['mean'] = self.data.groupby(self.binned_data[column])[column].mean()
        stats['std'] = self.data.groupby(self.binned_data[column])[column].std()
        stats['min'] = self.data.groupby(self.binned_data[column])[column].min()
        stats['max'] = self.data.groupby(self.binned_data[column])[column].max()
        
        return stats

    def visualize_binned_data(self, column):
        """
        Create visualizations for binned data
        """
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
        
        # Histogram of bin frequencies
        sns.histplot(
            data=self.binned_data[column],
            ax=ax1
        )
        ax1.set_title(f'Distribution of {column} Bins')
        ax1.set_xlabel('Bins')
        ax1.set_ylabel('Frequency')
        
        # Box plot
        sns.boxplot(
            x=self.binned_data[column],
            y=self.data[column],
            ax=ax2
        )
        ax2.set_title(f'Box Plot of {column} by Bins')
        
        # Violin plot
        sns.violinplot(
            x=self.binned_data[column],
            y=self.data[column],
            ax=ax3
        )
        ax3.set_title(f'Violin Plot of {column} by Bins')
        
        plt.tight_layout()
        plt.show()

    def analyze_relationships(self, column1, column2):
        """
        Analyze relationships between binned variables
        """
        # Create contingency table
        contingency_table = pd.crosstab(
            self.binned_data[column1],
            self.binned_data[column2],
            normalize='index'
        )
        
        # Visualize relationship
        plt.figure(figsize=(10, 6))
        sns.heatmap(contingency_table, annot=True, cmap='YlOrRd')
        plt.title(f'Relationship between {column1} and {column2} Bins')
        plt.show()
        
        return contingency_table

def main():
    # Load earthquake data
    #data = pd.read_csv('earthquakes.csv')
    data = pd.read_csv("D:/Usha/Prefect/Assignment1_trial/data/earthquakes.csv")
    
    # Initialize binning analyzer
    analyzer = EarthquakeBinningAnalysis(data)
    
    # Perform binning analysis for different columns
    print("Magnitude Binning Statistics:")
    magnitude_stats = analyzer.magnitude_binning()
    print(magnitude_stats)
    
    print("\nDepth Binning Statistics:")
    depth_stats = analyzer.depth_binning()
    print(depth_stats)
    
    print("\nIntensity Binning Statistics:")
    intensity_stats = analyzer.intensity_binning()
    print(intensity_stats)
    
    # Visualize binned data
    analyzer.visualize_binned_data('magnitude')
    analyzer.visualize_binned_data('depth')
    
    # Analyze relationships between binned variables
    relationship = analyzer.analyze_relationships('magnitude', 'depth')
    print("\nRelationship between Magnitude and Depth Bins:")
    print(relationship)

if __name__ == "__main__":
    main()