from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3
import os

DB_PATH = 'users.db'

class User(UserMixin):
    """User class for Flask-Login"""
    def __init__(self, user_id, username):
        self.id = user_id
        self.username = username

def init_db():
    """Initialize SQLite database for users"""
    if not os.path.exists(DB_PATH):
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('''CREATE TABLE users
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      username TEXT UNIQUE NOT NULL,
                      password_hash TEXT NOT NULL,
                      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
        conn.commit()
        conn.close()
        print("✓ Database initialized successfully")

def register_user(username, password):
    """
    Register a new user
    
    Args:
        username (str): Username
        password (str): Password (min 6 chars)
    
    Returns:
        tuple: (success: bool, message: str)
    """
    if not username or not password:
        return False, "Username and password required"
    
    username = username.strip()
    if len(username) < 3:
        return False, "Username must be at least 3 characters"
    
    if len(password) < 6:
        return False, "Password must be at least 6 characters"
    
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        password_hash = generate_password_hash(password, method='pbkdf2:sha256')
        c.execute('INSERT INTO users (username, password_hash) VALUES (?, ?)',
                  (username, password_hash))
        conn.commit()
        conn.close()
        return True, "✓ User registered successfully"
    except sqlite3.IntegrityError:
        return False, "✗ Username already exists"
    except Exception as e:
        return False, f"✗ Registration error: {str(e)}"

def authenticate_user(username, password):
    """
    Authenticate user and return User object if valid
    
    Args:
        username (str): Username
        password (str): Password
    
    Returns:
        User: User object if authenticated, None otherwise
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('SELECT id, password_hash FROM users WHERE username = ?', (username,))
        result = c.fetchone()
        conn.close()
        
        if result and check_password_hash(result[1], password):
            return User(result[0], username)
        return None
    except Exception as e:
        print(f"Authentication error: {e}")
        return None

def get_user_by_id(user_id):
    """
    Get user by ID for Flask-Login
    
    Args:
        user_id (int): User ID
    
    Returns:
        User: User object if found, None otherwise
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('SELECT id, username FROM users WHERE id = ?', (user_id,))
        result = c.fetchone()
        conn.close()
        
        if result:
            return User(result[0], result[1])
        return None
    except Exception as e:
        print(f"Error getting user: {e}")
        return None

def check_user_exists(username):
    """Check if a username already exists"""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('SELECT id FROM users WHERE username = ?', (username,))
        result = c.fetchone()
        conn.close()
        return result is not None
    except Exception as e:
        print(f"Error checking user: {e}")
        return False
