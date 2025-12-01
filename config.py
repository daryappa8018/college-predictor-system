import os
from datetime import timedelta

class Config:
    """Application configuration class"""
    
    # Base directory
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    
    # Secret key for sessions
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your-secret-key-change-in-production-2024'
    
    # Database configuration
    DATABASE_PATH = os.path.join(BASE_DIR, 'database', 'college_predictor.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Session configuration
    PERMANENT_SESSION_LIFETIME = timedelta(days=7)
    SESSION_COOKIE_SECURE = False  # Set to True in production with HTTPS
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    
    # Model paths
    MODEL_DIR = os.path.join(BASE_DIR, 'models')
    MODEL_PATH = os.path.join(MODEL_DIR, 'college_predictor_model.pkl')
    ENCODER_PATH = os.path.join(MODEL_DIR, 'label_encoders.pkl')
    
    # Data paths
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    TRAINING_DATA_PATH = os.path.join(DATA_DIR, 'data.csv')
    
    # Upload configuration
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    
    # Export configuration
    EXPORT_DIR = os.path.join(BASE_DIR, 'exports')
    
    # Pagination
    RESULTS_PER_PAGE = 30
    
    # Admin credentials (change in production)
    ADMIN_EMAIL = 'admin@college.com'
    ADMIN_PASSWORD = 'Admin@123'  # This will be hashed
    
    # ML Model parameters
    MODEL_PARAMS = {
        'n_estimators': 100,
        'max_depth': 20,
        'random_state': 42,
        'min_samples_split': 5,
        'min_samples_leaf': 2
    }
    
    # Prediction settings
    TOP_N_PREDICTIONS = 30
    CONFIDENCE_THRESHOLD = 0.7
    
    @staticmethod
    def init_app(app):
        """Initialize application with config"""
        # Create necessary directories
        os.makedirs(Config.MODEL_DIR, exist_ok=True)
        os.makedirs(Config.DATA_DIR, exist_ok=True)
        os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
        os.makedirs(Config.EXPORT_DIR, exist_ok=True)
        os.makedirs(os.path.dirname(Config.DATABASE_PATH), exist_ok=True)