import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

class EarthquakeFeatureImportance:
    def __init__(self, data):
        self.data = data
        
    def prepare_feature_columns(self):
        """
        Define feature groups for importance analysis
        """
        return {
            'target_variables': [
                'magnitude',    # Primary target - earthquake magnitude
                'mmi',         # Modified Mercalli Intensity
                'sig'          # Earthquake significance
            ],
            
            'seismic_features': [
                'depth',       # Depth of earthquake
                'nst',         # Number of seismic stations
                'gap',         # Azimuthal gap
                'dmin',        # Minimum distance to stations
                'rms'          # Root mean square travel time residual
            ],
            
            'impact_features': [
                'cdi',         # Community Decimal Intensity
                'felt',        # Number of felt reports
                'tsunami',     # Tsunami flag
                'alert'        # Alert level
            ],
            
            'location_features': [
                'latitude',    # Geographic latitude
                'longitude',   # Geographic longitude
                'distanceKM'   # Distance from epicenter
            ],
            
            'temporal_features': [
                'time',        # Time of event
                'updated'      # Last update time
            ]
        }

    def preprocess_features(self, target_variable):
        """
        Preprocess features for importance analysis
        """
        columns = self.prepare_feature_columns()
        
        # Combine all feature columns except target variables
        feature_columns = sum(
            [cols for name, cols in columns.items() if name != 'target_variables'],
            []
        )
        
        # Prepare feature matrix
        X = self.data[feature_columns].copy()
        y = self.data[target_variable]
        
        # Handle categorical variables
        categorical_features = ['alert', 'tsunami']
        X = pd.get_dummies(X, columns=categorical_features)
        
        # Handle temporal features
        X['time'] = pd.to_datetime(X['time']).astype(np.int64)
        X['updated'] = pd.to_datetime(X['updated']).astype(np.int64)
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
        
        return X_scaled, y

    def calculate_feature_importance(self, target_variable, n_estimators=100):
        """
        Calculate feature importance using Random Forest
        """
        # Preprocess features
        X, y = self.preprocess_features(target_variable)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train Random Forest model
        rf_model = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=42
        )
        rf_model.fit(X_train, y_train)
        
        # Calculate feature importance
        importance = pd.DataFrame({
            'feature': X.columns,
            'importance': rf_model.feature_importances_
        })
        
        # Calculate additional metrics
        importance['cumulative_importance'] = importance['importance'].cumsum()
        importance['importance_rank'] = importance['importance'].rank(
            ascending=False
        )
        
        return importance, rf_model

    def rank_features(self, importance_df):
        """
        Rank features by importance
        """
        return importance_df.sort_values('importance', ascending=False)

    def analyze_feature_groups(self, importance_df):
        """
        Analyze importance by feature groups
        """
        columns = self.prepare_feature_columns()
        group_importance = {}
        
        for group_name, features in columns.items():
            if group_name != 'target_variables':
                # Get importance values for features in this group
                group_features = [
                    col for col in importance_df['feature'] 
                    if any(feat in col for feat in features)
                ]
                group_importance[group_name] = importance_df[
                    importance_df['feature'].isin(group_features)
                ]['importance'].sum()
        
        return pd.Series(group_importance)

    def visualize_importance(self, importance_df, top_n=15):
        """
        Create visualizations of feature importance
        """
        # Feature importance bar plot
        plt.figure(figsize=(12, 6))
        importance_df = importance_df.sort_values('importance', ascending=True)
        plt.barh(
            importance_df['feature'].tail(top_n),
            importance_df['importance'].tail(top_n)
        )
        plt.title('Top Feature Importance')
        plt.xlabel('Importance Score')
        plt.tight_layout()
        
        # Cumulative importance plot
        plt.figure(figsize=(10, 6))
        plt.plot(
            range(len(importance_df)),
            importance_df['cumulative_importance'].sort_values(ascending=True)
        )
        plt.axhline(y=0.95, color='r', linestyle='--')
        plt.title('Cumulative Feature Importance')
        plt.xlabel('Number of Features')
        plt.ylabel('Cumulative Importance')
        plt.tight_layout()
        
        return plt.gcf()

    def generate_importance_report(self):
        """
        Generate comprehensive feature importance report
        """
        columns = self.prepare_feature_columns()
        report = {}
        
        for target in columns['target_variables']:
            # Calculate importance
            importance_df, model = self.calculate_feature_importance(target)
            
            # Rank features
            ranked_features = self.rank_features(importance_df)
            
            # Analyze feature groups
            group_importance = self.analyze_feature_groups(importance_df)
            
            # Store results
            report[target] = {
                'feature_importance': ranked_features,
                'group_importance': group_importance,
                'model': model
            }
            
            # Visualize importance
            self.visualize_importance(ranked_features)
            plt.show()
        
        return report

def main():
    # Load earthquake data
    data = pd.read_csv("D:/Usha/Prefect/Assignment1_trial/data/earthquakes.csv")
    
    # Initialize analyzer
    analyzer = EarthquakeFeatureImportance(data)
    
    # Generate importance report
    report = analyzer.generate_importance_report()
    
    # Print results for magnitude prediction
    print("\nFeature Importance for Magnitude Prediction:")
    print(report['magnitude']['feature_importance'])
    
    print("\nFeature Group Importance:")
    print(report['magnitude']['group_importance'])

if __name__ == "__main__":
    main()