import sqlite3
import hashlib
from datetime import datetime
from config import Config
import os

class DatabaseManager:
    """Handles all database operations"""
    
    def __init__(self):
        self.db_path = Config.DATABASE_PATH
        self.init_db()
    
    def get_connection(self):
        """Get database connection"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def init_db(self):
        """Initialize database with tables"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                phone TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP,
                is_active INTEGER DEFAULT 1
            )
        ''')
        
        # Admin table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS admins (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                name TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP
            )
        ''')
        
        # Predictions table (search history)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                rank INTEGER NOT NULL,
                category TEXT NOT NULL,
                year INTEGER NOT NULL,
                quota TEXT DEFAULT 'AI',
                pool TEXT DEFAULT 'Gender-Neutral',
                institute_type TEXT,
                round_no INTEGER DEFAULT 6,
                predicted_colleges TEXT,
                prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        ''')
        
        # Dataset versions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS dataset_versions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                version TEXT NOT NULL,
                uploaded_by INTEGER,
                upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                record_count INTEGER,
                file_path TEXT,
                FOREIGN KEY (uploaded_by) REFERENCES admins(id)
            )
        ''')
        
        # Model metrics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                total_predictions INTEGER DEFAULT 0,
                guest_predictions INTEGER DEFAULT 0,
                user_predictions INTEGER DEFAULT 0,
                last_retrain_date TIMESTAMP,
                dataset_version TEXT,
                model_version TEXT,
                accuracy_score REAL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Colleges table (from CSV data)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS colleges (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                year INTEGER,
                institute_type TEXT,
                round_no INTEGER,
                quota TEXT,
                pool TEXT,
                institute_short TEXT,
                program_name TEXT,
                program_duration TEXT,
                degree_short TEXT,
                category TEXT,
                opening_rank INTEGER,
                closing_rank INTEGER,
                is_preparatory INTEGER
            )
        ''')
        
        # System logs table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS system_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                action TEXT NOT NULL,
                details TEXT,
                ip_address TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        
        # Create default admin if not exists
        self.create_default_admin(cursor, conn)
        
        conn.close()
        print("Database initialized successfully")
    
    def create_default_admin(self, cursor, conn):
        """Create default admin account"""
        cursor.execute("SELECT * FROM admins WHERE email = ?", (Config.ADMIN_EMAIL,))
        if not cursor.fetchone():
            hashed_password = self.hash_password(Config.ADMIN_PASSWORD)
            cursor.execute('''
                INSERT INTO admins (email, password, name)
                VALUES (?, ?, ?)
            ''', (Config.ADMIN_EMAIL, hashed_password, 'System Admin'))
            conn.commit()
            print(f"Default admin created: {Config.ADMIN_EMAIL}")
    
    @staticmethod
    def hash_password(password):
        """Hash password using SHA-256"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def verify_password(self, password, hashed_password):
        """Verify password"""
        return self.hash_password(password) == hashed_password
    
    # User operations
    def create_user(self, name, email, password, phone=None):
        """Create new user"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            hashed_password = self.hash_password(password)
            cursor.execute('''
                INSERT INTO users (name, email, password, phone)
                VALUES (?, ?, ?, ?)
            ''', (name, email, hashed_password, phone))
            conn.commit()
            user_id = cursor.lastrowid
            conn.close()
            return True, "User created successfully", user_id
        except sqlite3.IntegrityError:
            conn.close()
            return False, "Email already exists", None
        except Exception as e:
            conn.close()
            return False, str(e), None
    
    def get_user_by_email(self, email):
        """Get user by email"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE email = ?", (email,))
        user = cursor.fetchone()
        conn.close()
        return dict(user) if user else None
    
    def get_user_by_id(self, user_id):
        """Get user by ID"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
        user = cursor.fetchone()
        conn.close()
        return dict(user) if user else None
    
    def update_last_login(self, user_id):
        """Update user's last login time"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE users SET last_login = CURRENT_TIMESTAMP
            WHERE id = ?
        ''', (user_id,))
        conn.commit()
        conn.close()
    
    def get_all_users(self):
        """Get all users"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users ORDER BY created_at DESC")
        users = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return users
    
    def delete_user(self, user_id):
        """Delete user"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM users WHERE id = ?", (user_id,))
        conn.commit()
        conn.close()
        return True
    
    # Admin operations
    def verify_admin(self, email, password):
        """Verify admin credentials"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM admins WHERE email = ?", (email,))
        admin = cursor.fetchone()
        conn.close()
        
        if admin and self.verify_password(password, admin['password']):
            return dict(admin)
        return None
    
    def update_admin_last_login(self, admin_id):
        """Update admin's last login"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE admins SET last_login = CURRENT_TIMESTAMP
            WHERE id = ?
        ''', (admin_id,))
        conn.commit()
        conn.close()
    
    # Prediction operations
    def save_prediction(self, user_id, rank, category, year, predicted_colleges, 
                       quota='AI', pool='Gender-Neutral', institute_type=None, round_no=6):
        """Save prediction to database"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO predictions 
            (user_id, rank, category, year, quota, pool, institute_type, round_no, predicted_colleges)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (user_id, rank, category, year, quota, pool, institute_type, round_no, str(predicted_colleges)))
        
        conn.commit()
        prediction_id = cursor.lastrowid
        conn.close()
        
        # Update model metrics
        self.update_model_metrics(is_guest=(user_id is None))
        
        return prediction_id
    
    def get_user_predictions(self, user_id, limit=10):
        """Get user's prediction history"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT * FROM predictions 
            WHERE user_id = ? 
            ORDER BY prediction_date DESC 
            LIMIT ?
        ''', (user_id, limit))
        predictions = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return predictions
    
    def get_all_predictions(self, limit=100):
        """Get all predictions for admin"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT p.*, u.name, u.email 
            FROM predictions p
            LEFT JOIN users u ON p.user_id = u.id
            ORDER BY p.prediction_date DESC
            LIMIT ?
        ''', (limit,))
        predictions = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return predictions
    
    # College data operations
    def import_college_data(self, csv_path):
        """Import college data from CSV"""
        import pandas as pd
        
        df = pd.read_csv(csv_path)
        
        # Drop unnamed index column if exists
        if 'Unnamed: 0' in df.columns:
            df = df.drop('Unnamed: 0', axis=1)
        
        conn = self.get_connection()
        
        # Clear existing data
        conn.execute("DELETE FROM colleges")
        
        # Insert new data
        df.to_sql('colleges', conn, if_exists='append', index=False)
        conn.commit()
        conn.close()
        
        return len(df)
    
    def search_colleges(self, filters):
        """Search colleges with filters"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        query = "SELECT * FROM colleges WHERE 1=1"
        params = []
        
        if filters.get('institute_type'):
            query += " AND institute_type = ?"
            params.append(filters['institute_type'])
        
        if filters.get('category'):
            query += " AND category = ?"
            params.append(filters['category'])
        
        if filters.get('rank'):
            query += " AND opening_rank <= ? AND closing_rank >= ?"
            params.extend([filters['rank'], filters['rank']])
        
        query += " ORDER BY closing_rank ASC LIMIT 30"
        
        cursor.execute(query, params)
        colleges = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        return colleges
    
    # Statistics for admin dashboard
    def get_statistics(self):
        """Get system statistics"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        stats = {}
        
        # Total users
        cursor.execute("SELECT COUNT(*) as count FROM users")
        stats['total_users'] = cursor.fetchone()['count']
        
        # Total predictions
        cursor.execute("SELECT COUNT(*) as count FROM predictions")
        stats['total_predictions'] = cursor.fetchone()['count']
        
        # Total colleges
        cursor.execute("SELECT COUNT(*) as count FROM colleges")
        stats['total_colleges'] = cursor.fetchone()['count']
        
        # Recent users (last 7 days)
        cursor.execute('''
            SELECT COUNT(*) as count FROM users 
            WHERE created_at >= datetime('now', '-7 days')
        ''')
        stats['recent_users'] = cursor.fetchone()['count']
        
        conn.close()
        return stats
    
    # Logging
    def log_action(self, user_id, action, details=None, ip_address=None):
        """Log user action"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO system_logs (user_id, action, details, ip_address)
            VALUES (?, ?, ?, ?)
        ''', (user_id, action, details, ip_address))
        conn.commit()
        conn.close()
    
    # Enhanced statistics
    def get_enhanced_statistics(self):
        """Get enhanced statistics for admin dashboard"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        stats = self.get_statistics()
        
        # Searches today
        cursor.execute('''
            SELECT COUNT(*) as count FROM predictions 
            WHERE DATE(prediction_date) = DATE('now')
        ''')
        stats['searches_today'] = cursor.fetchone()['count']
        
        # Weekly activity (last 7 days)
        cursor.execute('''
            SELECT DATE(prediction_date) as date, COUNT(*) as count
            FROM predictions 
            WHERE prediction_date >= datetime('now', '-7 days')
            GROUP BY DATE(prediction_date)
            ORDER BY date DESC
        ''')
        stats['weekly_activity'] = [dict(row) for row in cursor.fetchall()]
        
        conn.close()
        return stats
    
    # User management with search count
    def get_users_with_stats(self):
        """Get all users with their search statistics"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT u.*, 
                   COUNT(p.id) as total_searches,
                   MAX(p.prediction_date) as last_search_date
            FROM users u
            LEFT JOIN predictions p ON u.id = p.user_id
            GROUP BY u.id
            ORDER BY u.created_at DESC
        ''')
        users = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return users
    
    # Dataset management
    def save_dataset_version(self, filename, version, uploaded_by, record_count, file_path):
        """Save dataset version information"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO dataset_versions 
            (filename, version, uploaded_by, record_count, file_path)
            VALUES (?, ?, ?, ?, ?)
        ''', (filename, version, uploaded_by, record_count, file_path))
        conn.commit()
        conn.close()
    
    def get_dataset_versions(self, limit=10):
        """Get recent dataset versions"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT dv.*, a.name as uploaded_by_name
            FROM dataset_versions dv
            LEFT JOIN admins a ON dv.uploaded_by = a.id
            ORDER BY dv.upload_date DESC
            LIMIT ?
        ''', (limit,))
        versions = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return versions
    
    # Model metrics
    def init_model_metrics(self):
        """Initialize model metrics if not exists"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) as count FROM model_metrics')
        if cursor.fetchone()['count'] == 0:
            cursor.execute('''
                INSERT INTO model_metrics 
                (total_predictions, guest_predictions, user_predictions)
                VALUES (0, 0, 0)
            ''')
            conn.commit()
        conn.close()
    
    def update_model_metrics(self, is_guest=False):
        """Update model prediction metrics"""
        self.init_model_metrics()
        conn = self.get_connection()
        cursor = conn.cursor()
        
        if is_guest:
            cursor.execute('''
                UPDATE model_metrics 
                SET total_predictions = total_predictions + 1,
                    guest_predictions = guest_predictions + 1,
                    updated_at = CURRENT_TIMESTAMP
            ''')
        else:
            cursor.execute('''
                UPDATE model_metrics 
                SET total_predictions = total_predictions + 1,
                    user_predictions = user_predictions + 1,
                    updated_at = CURRENT_TIMESTAMP
            ''')
        
        conn.commit()
        conn.close()
    
    def get_model_metrics(self):
        """Get model metrics"""
        self.init_model_metrics()
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM model_metrics LIMIT 1')
        metrics = dict(cursor.fetchone())
        conn.close()
        return metrics
    
    def update_model_retrain_info(self, dataset_version, accuracy_score):
        """Update model retrain information"""
        self.init_model_metrics()
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE model_metrics 
            SET last_retrain_date = CURRENT_TIMESTAMP,
                dataset_version = ?,
                accuracy_score = ?,
                updated_at = CURRENT_TIMESTAMP
        ''', (dataset_version, accuracy_score))
        conn.commit()
        conn.close()
    
    # Recent activity
    def get_recent_activity(self, limit=10):
        """Get recent system activity"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT p.*, u.name as user_name, u.email as user_email
            FROM predictions p
            LEFT JOIN users u ON p.user_id = u.id
            ORDER BY p.prediction_date DESC
            LIMIT ?
        ''', (limit,))
        activity = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return activity

# Initialize database on import
def init_db():
    """Initialize database"""
    db = DatabaseManager()
    return db

if __name__ == "__main__":
    init_db()