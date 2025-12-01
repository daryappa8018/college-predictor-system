from database.db_manager import DatabaseManager
import os

# Initialize database manager
db = DatabaseManager()

# Path to CSV file
csv_path = 'data/data.csv'

if os.path.exists(csv_path):
    print(f"Loading data from {csv_path}...")
    count = db.import_college_data(csv_path)
    print(f"Successfully imported {count} college records!")
    
    # Verify import
    conn = db.get_connection()
    cursor = conn.cursor()
    
    cursor.execute('SELECT COUNT(*) as count FROM colleges')
    total = cursor.fetchone()['count']
    print(f"\nTotal colleges in database: {total}")
    
    cursor.execute('SELECT DISTINCT category FROM colleges ORDER BY category')
    categories = [row['category'] for row in cursor.fetchall()]
    print(f"\nAvailable categories: {categories}")
    
    cursor.execute('SELECT DISTINCT institute_type FROM colleges ORDER BY institute_type')
    institutes = [row['institute_type'] for row in cursor.fetchall()]
    print(f"\nAvailable institute types: {institutes}")
    
    conn.close()
else:
    print(f"Error: CSV file not found at {csv_path}")
