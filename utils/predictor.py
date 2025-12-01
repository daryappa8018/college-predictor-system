import pandas as pd
import joblib
import os
from config import Config

class CollegePredictor:
    """Handle college predictions"""
    
    def __init__(self, db_manager):
        self.db = db_manager
        self.model = None
        self.encoders = None
        self.load_model()
    
    def load_model(self):
        """Load trained model and encoders"""
        try:
            if os.path.exists(Config.MODEL_PATH):
                self.model = joblib.load(Config.MODEL_PATH)
                print("Model loaded successfully")
            else:
                print("Warning: Model file not found. Please train the model first.")
            
            if os.path.exists(Config.ENCODER_PATH):
                self.encoders = joblib.load(Config.ENCODER_PATH)
                print("Encoders loaded successfully")
            else:
                print("Warning: Encoder file not found.")
        except Exception as e:
            print(f"Error loading model: {e}")
    
    def predict_single_college(self, rank, category, year=2024, quota='AI', 
                              pool='Gender-Neutral', institute_type=None):
        """Predict single best college for guest users"""
        
        # Get colleges from database
        filters = {
            'rank': rank,
            'category': category
        }
        
        if institute_type:
            filters['institute_type'] = institute_type
        
        colleges = self.db.search_colleges(filters)
        
        if not colleges:
            return None, "No colleges found matching your criteria"
        
        # Return the best match (first result)
        best_college = colleges[0]
        
        return {
            'college': best_college['institute_short'],
            'program': best_college['program_name'],
            'degree': best_college['degree_short'],
            'opening_rank': best_college['opening_rank'],
            'closing_rank': best_college['closing_rank'],
            'institute_type': best_college['institute_type']
        }, None
    
    def predict_top_colleges(self, rank, category, year=2024, quota='AI',
                            pool='Gender-Neutral', institute_type=None, 
                            round_no=6, limit=30):
        """Predict top N colleges with 5-step smart fallback"""
        
        # Step 1: Try with exact parameters
        colleges = self._search_colleges(rank, category, quota, pool, institute_type, round_no, limit)
        
        if colleges:
            return self._format_results(colleges, rank, limit), None
        
        # Step 2: Try Round 6 if other round selected
        if round_no != 6:
            colleges = self._search_colleges(rank, category, quota, pool, institute_type, 6, limit)
            if colleges:
                return self._format_results(colleges, rank, limit), None
        
        # Step 3: Try OS quota if regional quota fails
        if quota not in ['AI', 'OS']:
            colleges = self._search_colleges(rank, category, 'OS', pool, institute_type, round_no, limit)
            if colleges:
                return self._format_results(colleges, rank, limit), None
        
        # Step 4: Try AI quota if still no results
        if quota != 'AI':
            colleges = self._search_colleges(rank, category, 'AI', pool, institute_type, round_no, limit)
            if colleges:
                return self._format_results(colleges, rank, limit), None
        
        # Step 5: Try Gender-Neutral pool if Female-Only fails
        if pool != 'Gender-Neutral':
            colleges = self._search_colleges(rank, category, quota, 'Gender-Neutral', institute_type, round_no, limit)
            if colleges:
                return self._format_results(colleges, rank, limit), None
        
        # Step 6: Try All institutes if specific type fails
        if institute_type:
            colleges = self._search_colleges(rank, category, quota, pool, None, round_no, limit)
            if colleges:
                return self._format_results(colleges, rank, limit), None
        
        return [], "No colleges found matching your criteria. Try adjusting your parameters."
    
    def _search_colleges(self, rank, category, quota, pool, institute_type, round_no, limit):
        """Internal method to search colleges"""
        conn = self.db.get_connection()
        cursor = conn.cursor()
        
        query = '''
            SELECT * FROM colleges 
            WHERE category = ? 
            AND opening_rank <= ? 
            AND closing_rank >= ?
        '''
        params = [category, rank + 5000, rank - 5000]
        
        # Add optional filters
        if quota and quota != 'All':
            query += " AND quota = ?"
            params.append(quota)
        
        if pool and pool != 'All':
            query += " AND pool = ?"
            params.append(pool)
        
        if institute_type and institute_type != 'All':
            query += " AND institute_type = ?"
            params.append(institute_type)
        
        if round_no:
            query += " AND round_no = ?"
            params.append(round_no)
        
        query += " ORDER BY closing_rank ASC LIMIT ?"
        params.append(limit * 2)  # Get more for better filtering
        
        cursor.execute(query, params)
        colleges = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        return colleges
    
    def _format_results(self, colleges, rank, limit):
        """Format college results with match scores"""
        if not colleges:
            return []
        
        results = []
        for college in colleges:
            match_score = self.calculate_match_score(rank, college)
            
            results.append({
                'college': college['institute_short'],
                'program': college['program_name'],
                'degree': college['degree_short'],
                'opening_rank': college['opening_rank'],
                'closing_rank': college['closing_rank'],
                'institute_type': college['institute_type'],
                'match_score': match_score,
                'year': college['year'],
                'duration': college['program_duration'],
                'quota': college['quota'],
                'pool': college['pool'],
                'round_no': college.get('round_no', 6)
            })
        
        # Sort by match score
        results.sort(key=lambda x: x['match_score'], reverse=True)
        
        return results[:limit]
    
    def calculate_match_score(self, user_rank, college):
        """Calculate how well a college matches the user's rank"""
        opening = college['opening_rank']
        closing = college['closing_rank']
        
        if user_rank < opening:
            # Rank is better than opening - high confidence
            score = 100
        elif opening <= user_rank <= closing:
            # Rank is within range - calculate position
            range_size = closing - opening
            position = user_rank - opening
            score = 100 - (position / range_size * 30)  # 70-100% match
        else:
            # Rank is worse than closing - lower confidence
            difference = user_rank - closing
            score = max(0, 70 - (difference / 1000) * 10)  # Decrease score
        
        return round(score, 2)
    
    def get_iit_colleges(self, rank, category, year=2024, limit=30):
        """Get IIT colleges separately"""
        results, error = self.predict_top_colleges(
            rank, category, year, 
            institute_type='IIT', 
            limit=limit
        )
        return results, error
    
    def get_nit_colleges(self, rank, category, year=2024, limit=30):
        """Get NIT colleges separately"""
        results, error = self.predict_top_colleges(
            rank, category, year,
            institute_type='NIT',
            limit=limit
        )
        return results, error
    
    def get_all_institute_types(self):
        """Get list of all institute types"""
        conn = self.db.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT institute_type FROM colleges ORDER BY institute_type")
        types = [row['institute_type'] for row in cursor.fetchall()]
        conn.close()
        return types
    
    def get_all_categories(self):
        """Get list of all categories"""
        conn = self.db.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT category FROM colleges ORDER BY category")
        categories = [row['category'] for row in cursor.fetchall()]
        conn.close()
        return categories
    
    def get_all_quotas(self):
        """Get list of all quotas"""
        # Standard quotas
        return ['AI', 'HS', 'OS', 'AP', 'GO', 'JK', 'LA']
    
    def get_all_pools(self):
        """Get list of all pools"""
        return ['Gender-Neutral', 'Female-Only']
    
    def get_all_rounds(self):
        """Get list of counseling rounds"""
        return list(range(1, 8))  # Rounds 1-7
    
    def validate_input(self, rank, category):
        """Validate user input"""
        errors = []
        
        if not rank or rank <= 0:
            errors.append("Please enter a valid rank greater than 0")
        
        if rank and rank > 1000000:
            errors.append("Rank seems too high. Please check your input")
        
        if not category:
            errors.append("Please select a category")
        
        valid_categories = self.get_all_categories()
        if category and category not in valid_categories:
            errors.append(f"Invalid category. Must be one of: {', '.join(valid_categories)}")
        
        return len(errors) == 0, errors