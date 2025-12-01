import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import os
from database_config import CollegeDataManager

class CollegePredictorModel:
    
    def __init__(self):
        self.model = None
        self.label_encoders = {}
        self.data_manager = CollegeDataManager()
        self.category_mapping = {
            'GEN': 0, 'GEN-EWS': 1, 'GEN-EWS-PWD': 2, 'GEN-PWD': 3,
            'OBC-NCL': 4, 'OBC-NCL-PWD': 5, 'SC': 6, 'SC-PWD': 7,
            'ST': 8, 'ST-PWD': 9
        }
        self.quota_mapping = {
            'AI': 0, 'AP': 1, 'GO': 2, 'HS': 3, 'JK': 4, 'LA': 5, 'OS': 6
        }
        self.pool_mapping = {
            'Gender-Neutral': 0, 'Female-only': 1
        }
        self.institute_type_mapping = {
            'IIT': 0, 'NIT': 1
        }
        self.df = None  # Store loaded data for filtering
    
    def load_data_from_database(self, filters=None):
        df = self.data_manager.get_college_data(filters)
        if df.empty:
            raise Exception("No data found in database.")
        self.df = df  # Store for later use
        return df
    
    def preprocess_data(self, df):
        processed_df = df.copy()
        
        processed_df['institute_type_encoded'] = processed_df['institute_type'].map(self.institute_type_mapping)
        processed_df['category_encoded'] = processed_df['category'].map(self.category_mapping)
        processed_df['quota_encoded'] = processed_df['quota'].map(self.quota_mapping)
        processed_df['pool_encoded'] = processed_df['pool'].map(self.pool_mapping)
        
        target_columns = ['institute_short', 'degree_short', 'program_name']
        for col in target_columns:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
            processed_df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(processed_df[col])
        
        return processed_df
    
    def prepare_features_and_targets(self, processed_df):
        feature_columns = [
            'category_encoded', 'quota_encoded', 'pool_encoded', 
            'institute_type_encoded', 'round_no', 'opening_rank', 'closing_rank'
        ]
        target_columns = ['institute_short_encoded', 'degree_short_encoded', 'program_name_encoded']
        
        X = processed_df[feature_columns]
        y = processed_df[target_columns]
        
        return X, y
    
    def train_model(self, test_size=0.2, random_state=42):
        df = self.load_data_from_database()
        processed_df = self.preprocess_data(df)
        X, y = self.prepare_features_and_targets(processed_df)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        self.model = KNeighborsClassifier(n_neighbors=5)
        self.model.fit(X_train, y_train)
        
        y_pred = self.model.predict(X_test)
        
        accuracies = {}
        for i, col in enumerate(['institute_short', 'degree_short', 'program_name']):
            accuracy = accuracy_score(y_test.iloc[:, i], y_pred[:, i])
            accuracies[col] = accuracy
        
        mean_accuracy = np.mean(list(accuracies.values()))
        return accuracies
    
    def predict(self, input_data):
        """
        DEPRECATED: Use get_eligible_colleges() instead for better results.
        This method is kept for backward compatibility.
        """
        if self.model is None:
            raise Exception("Model not trained.")
        
        feature_vector = np.array([
            self.category_mapping.get(input_data['category'], 0),
            self.quota_mapping.get(input_data['quota'], 0),
            self.pool_mapping.get(input_data['pool'], 0),
            self.institute_type_mapping.get(input_data['institute_type'], 0),
            input_data['round_no'],
            input_data['opening_rank'],
            input_data['closing_rank']
        ]).reshape(1, -1)
        
        prediction_encoded = self.model.predict(feature_vector)[0]
        
        prediction = {
            'college': self.label_encoders['institute_short'].inverse_transform([prediction_encoded[0]])[0],
            'degree': self.label_encoders['degree_short'].inverse_transform([prediction_encoded[1]])[0],
            'course': self.label_encoders['program_name'].inverse_transform([prediction_encoded[2]])[0]
        }
        
        return prediction
    
    def get_eligible_colleges(self, student_data, top_n=20):
        """
        NEW METHOD: Find colleges where student is eligible based on their rank.
        This is the RECOMMENDED method to use instead of predict().
        
        Parameters:
        - student_data: dict with keys:
            - rank: int (student's rank)
            - category: str (GEN, OBC-NCL, SC, ST, etc.)
            - quota: str (AI, HS, OS, etc.)
            - pool: str (Gender-Neutral, Female-only)
            - institute_type: str (IIT, NIT) - IMPORTANT: This will be respected
            - round_no: int (default: 6 for final round)
        - top_n: int, number of top results to return
        
        Returns:
        - DataFrame with eligible colleges sorted by closing rank
        """
        if self.df is None:
            self.load_data_from_database()
        
        # Extract student information
        student_rank = student_data['rank']
        category = student_data['category']
        quota = student_data.get('quota', 'AI')
        pool = student_data.get('pool', 'Gender-Neutral')
        institute_type = student_data.get('institute_type', None)
        round_no = student_data.get('round_no', 6)
        
        # Filter data based on student criteria
        filtered_df = self.df[
            (self.df['category'] == category) &
            (self.df['quota'] == quota) &
            (self.df['pool'] == pool) &
            (self.df['round_no'] == round_no)
        ].copy()
        
        # CRITICAL FIX: Filter by institute type if specified
        if institute_type:
            filtered_df = filtered_df[filtered_df['institute_type'] == institute_type]
        
        # Find colleges where student's rank falls within opening and closing ranks
        eligible_df = filtered_df[
            (filtered_df['opening_rank'] <= student_rank) &
            (filtered_df['closing_rank'] >= student_rank)
        ].copy()
        
        # Sort by closing rank (lower closing rank = more competitive/better)
        eligible_df = eligible_df.sort_values('closing_rank')
        
        # Select relevant columns for output
        result_columns = [
            'institute_short', 'institute_type', 'program_name', 
            'degree_short', 'opening_rank', 'closing_rank',
            'category', 'quota', 'pool', 'round_no'
        ]
        
        return eligible_df[result_columns].head(top_n)
    
    def get_safe_colleges(self, student_data, safety_margin=500, top_n=10):
        """
        Find 'safe' colleges where student's rank is well below closing rank
        """
        if self.df is None:
            self.load_data_from_database()
        
        student_rank = student_data['rank']
        category = student_data['category']
        quota = student_data.get('quota', 'AI')
        pool = student_data.get('pool', 'Gender-Neutral')
        institute_type = student_data.get('institute_type', None)
        round_no = student_data.get('round_no', 6)
        
        filtered_df = self.df[
            (self.df['category'] == category) &
            (self.df['quota'] == quota) &
            (self.df['pool'] == pool) &
            (self.df['round_no'] == round_no)
        ].copy()
        
        if institute_type:
            filtered_df = filtered_df[filtered_df['institute_type'] == institute_type]
        
        safe_df = filtered_df[
            (filtered_df['closing_rank'] >= student_rank + safety_margin)
        ].copy()
        
        safe_df = safe_df.sort_values('closing_rank')
        
        result_columns = [
            'institute_short', 'institute_type', 'program_name', 
            'degree_short', 'opening_rank', 'closing_rank',
            'category', 'quota', 'pool'
        ]
        
        return safe_df[result_columns].head(top_n)
    
    def get_dream_colleges(self, student_data, stretch_margin=500, top_n=10):
        """
        Find 'dream' colleges where student's rank is slightly above opening rank
        """
        if self.df is None:
            self.load_data_from_database()
        
        student_rank = student_data['rank']
        category = student_data['category']
        quota = student_data.get('quota', 'AI')
        pool = student_data.get('pool', 'Gender-Neutral')
        institute_type = student_data.get('institute_type', None)
        round_no = student_data.get('round_no', 6)
        
        filtered_df = self.df[
            (self.df['category'] == category) &
            (self.df['quota'] == quota) &
            (self.df['pool'] == pool) &
            (self.df['round_no'] == round_no)
        ].copy()
        
        if institute_type:
            filtered_df = filtered_df[filtered_df['institute_type'] == institute_type]
        
        dream_df = filtered_df[
            (filtered_df['opening_rank'] - stretch_margin <= student_rank) &
            (filtered_df['opening_rank'] >= student_rank)
        ].copy()
        
        dream_df = dream_df.sort_values('opening_rank')
        
        result_columns = [
            'institute_short', 'institute_type', 'program_name', 
            'degree_short', 'opening_rank', 'closing_rank',
            'category', 'quota', 'pool'
        ]
        
        return dream_df[result_columns].head(top_n)
    
    def get_recommendations(self, student_data):
        """
        Get comprehensive recommendations with dream, eligible, and safe colleges
        """
        results = {
            'dream_colleges': self.get_dream_colleges(student_data),
            'eligible_colleges': self.get_eligible_colleges(student_data),
            'safe_colleges': self.get_safe_colleges(student_data)
        }
        
        return results
    
    def save_model(self, filepath='model_clean.pkl'):
        if self.model is None:
            raise Exception("No model to save.")
        
        model_data = {
            'model': self.model,
            'label_encoders': self.label_encoders,
            'category_mapping': self.category_mapping,
            'quota_mapping': self.quota_mapping,
            'pool_mapping': self.pool_mapping,
            'institute_type_mapping': self.institute_type_mapping
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath='model_clean.pkl'):
        if not os.path.exists(filepath):
            raise Exception(f"Model file not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.label_encoders = model_data['label_encoders']
        self.category_mapping = model_data['category_mapping']
        self.quota_mapping = model_data['quota_mapping']
        self.pool_mapping = model_data['pool_mapping']
        self.institute_type_mapping = model_data['institute_type_mapping']

def retrain_model():
    predictor = CollegePredictorModel()
    accuracies = predictor.train_model()
    predictor.save_model('model_clean.pkl')
    return accuracies

def test_prediction():
    """Old test function - kept for backward compatibility"""
    predictor = CollegePredictorModel()
    predictor.load_model('model_clean.pkl')
    
    test_input = {
        'category': 'GEN',
        'quota': 'AI',
        'pool': 'Gender-Neutral',
        'institute_type': 'IIT',
        'round_no': 6,
        'opening_rank': 1000,
        'closing_rank': 2000
    }
    
    prediction = predictor.predict(test_input)
    return prediction

def test_new_method():
    """New test function showing the recommended approach"""
    predictor = CollegePredictorModel()
    
    # Student information - notice we only need RANK, not opening/closing ranks
    student_info = {
        'rank': 5000,
        'category': 'GEN',
        'quota': 'AI',
        'pool': 'Gender-Neutral',
        'institute_type': 'NIT',  # This will now be respected!
        'round_no': 6
    }
    
    print("=" * 80)
    print("ELIGIBLE COLLEGES FOR NIT ONLY")
    print("=" * 80)
    eligible = predictor.get_eligible_colleges(student_info, top_n=10)
    print(eligible.to_string() if not eligible.empty else "No eligible colleges found")
    print()
    
    # Test with IIT
    student_info['institute_type'] = 'IIT'
    print("=" * 80)
    print("ELIGIBLE COLLEGES FOR IIT ONLY")
    print("=" * 80)
    eligible_iit = predictor.get_eligible_colleges(student_info, top_n=10)
    print(eligible_iit.to_string() if not eligible_iit.empty else "No eligible colleges found")

if __name__ == "__main__":
    # Old method still works
    print("Testing old method (predict)...")
    accuracies = retrain_model()
    print("Model Accuracies:", accuracies)
    prediction = test_prediction()
    print("Test Prediction:", prediction)
    print()
    
    # New recommended method
    print("\nTesting NEW method (get_eligible_colleges)...")
    test_new_method()