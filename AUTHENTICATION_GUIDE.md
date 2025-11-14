# Authentication Setup Guide for Render Deployment

This guide walks you through adding authentication to your Mental Health Sentiment Tracking project and deploying it on Render.

## Table of Contents
1. [Step 1: Create Authentication Module](#step-1-create-authentication-module)
2. [Step 2: Update Flask App with Authentication](#step-2-update-flask-app-with-authentication)
3. [Step 3: Update Requirements](#step-3-update-requirements)
4. [Step 4: Create Render Account](#step-4-create-render-account)
5. [Step 5: Set Environment Variables](#step-5-set-environment-variables)
6. [Step 6: Deploy on Render](#step-6-deploy-on-render)
7. [Step 7: Test Authentication](#step-7-test-authentication)

---

## Step 1: Create Authentication Module

### What You Need to Know
- We'll use `Flask-Login` for session management
- `Werkzeug` for password hashing
- SQLite for storing user credentials (can upgrade to PostgreSQL on Render)

### Create `src/auth.py`

```python
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3
import os

DB_PATH = 'users.db'

class User(UserMixin):
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
                      password_hash TEXT NOT NULL)''')
        conn.commit()
        conn.close()
        print("Database initialized")

def register_user(username, password):
    """Register a new user"""
    if not username or not password:
        return False, "Username and password required"
    
    if len(password) < 6:
        return False, "Password must be at least 6 characters"
    
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        password_hash = generate_password_hash(password)
        c.execute('INSERT INTO users (username, password_hash) VALUES (?, ?)',
                  (username, password_hash))
        conn.commit()
        conn.close()
        return True, "User registered successfully"
    except sqlite3.IntegrityError:
        return False, "Username already exists"
    except Exception as e:
        return False, f"Registration error: {str(e)}"

def authenticate_user(username, password):
    """Authenticate user and return User object if valid"""
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
    """Get user by ID for Flask-Login"""
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
```

---

## Step 2: Update Flask App with Authentication

### Update `src/app.py`

Add these imports at the top:
```python
from flask_login import LoginManager, login_required, login_user, logout_user, current_user
from auth import init_db, register_user, authenticate_user, get_user_by_id
```

Add configuration after Flask app initialization:
```python
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

@login_manager.user_loader
def load_user(user_id):
    return get_user_by_id(int(user_id))

# Initialize database
init_db()
```

Add authentication routes:
```python
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        data = request.get_json()
        username = data.get('username', '').strip()
        password = data.get('password', '').strip()
        
        success, message = register_user(username, password)
        if success:
            return jsonify({'success': True, 'message': message}), 201
        else:
            return jsonify({'success': False, 'message': message}), 400
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        data = request.get_json()
        username = data.get('username', '').strip()
        password = data.get('password', '').strip()
        
        user = authenticate_user(username, password)
        if user:
            login_user(user)
            return jsonify({'success': True, 'message': 'Login successful'}), 200
        else:
            return jsonify({'success': False, 'message': 'Invalid credentials'}), 401
    
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect('/login')
```

Protect your analysis routes with `@login_required`:
```python
@app.route('/predict', methods=['POST'])
@login_required  # Add this decorator
def predict():
    # ... existing code ...
```

Similarly add `@login_required` to:
- `/plot_sentiment`
- `/wordcloud`
- `/export_pdf`
- `/export_csv`
- `/get_history`

---

## Step 3: Update Requirements

Add to `requirements.txt`:
```
Flask-Login==0.6.2
Werkzeug==2.3.0
```

Run locally to test:
```bash
pip install -r requirements.txt
```

---

## Step 4: Create Render Account

### Step-by-Step:

1. **Go to Render.com**
   - Visit https://render.com
   - Click "Sign up" in top right

2. **Sign Up Options**
   - Use GitHub account (recommended - easier for deployment)
   - Use Email address
   - Or use Google/GitLab

3. **Verify Email** (if using email signup)
   - Check inbox and click verification link

4. **Complete Profile**
   - Enter name and organization (optional)
   - Accept terms and click "Create account"

---

## Step 5: Set Environment Variables

### On Render Dashboard:

1. Click "New +"
2. Select "Web Service"
3. Connect your GitHub repository
4. Fill in service details:
   - **Name**: `mental-health-sentiment`
   - **Region**: Select closest to you (e.g., `Oregon`)
   - **Branch**: `main`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn src.app:app`

5. **Environment Variables** (Critical):
   - Click "Advanced" → "Add Environment Variable"
   - Add these variables:

| Key | Value |
|-----|-------|
| `SECRET_KEY` | Generate random string (use `python -c "import secrets; print(secrets.token_hex(32))"`) |
| `FLASK_ENV` | `production` |
| `PYTHONUNBUFFERED` | `1` |

### Generate SECRET_KEY Locally:
```powershell
python -c "import secrets; print(secrets.token_hex(32))"
```
Copy the output and paste in Render's SECRET_KEY environment variable.

---

## Step 6: Deploy on Render

### Complete Deployment Steps:

1. **In Render Dashboard:**
   - Make sure all environment variables are set (from Step 5)
   - Click "Create Web Service"
   - Wait for build to complete (5-10 minutes)

2. **Monitor Deployment:**
   - Check "Logs" tab to see build progress
   - Look for errors and fix if needed

3. **Access Your App:**
   - Once deployed, you'll get a URL like: `https://mental-health-sentiment.onrender.com`
   - Visit this URL in your browser

### Troubleshooting Deployment:

| Issue | Solution |
|-------|----------|
| "Module not found" | Check `requirements.txt` has all dependencies |
| "SECRET_KEY not set" | Add environment variable in Render dashboard |
| "Port already in use" | Render automatically assigns ports, no action needed |
| "Build fails" | Check Logs tab for specific error messages |

---

## Step 7: Test Authentication

### Test Locally First:

```bash
# Terminal 1: Start Flask app
python -c "from src.app import app; app.run(debug=True)"
```

### Create Test User:

```python
# In Python terminal
from src.auth import init_db, register_user

init_db()
success, msg = register_user('testuser', 'password123')
print(msg)  # Should print success message
```

### Test Login Flow:

1. Open browser: `http://localhost:5000/register`
2. Register with:
   - Username: `testuser`
   - Password: `password123`
3. Visit: `http://localhost:5000/login`
4. Login with same credentials
5. Try accessing: `http://localhost:5000/analyze`
   - Should be accessible after login
   - Should redirect to login if not logged in

---

## Quick Command Reference

### Local Testing
```bash
# Navigate to project folder
cd "c:\Users\HP\Desktop\New folder\Mental health sentiment tracking using social media data"

# Install dependencies
pip install -r requirements.txt

# Run app
python -c "from src.app import app; app.run(debug=True)"

# Test endpoints
curl -X POST http://localhost:5000/register -H "Content-Type: application/json" -d "{\"username\":\"test\",\"password\":\"pass123\"}"
```

### Render Setup Commands
```bash
# Generate SECRET_KEY
python -c "import secrets; print(secrets.token_hex(32))"

# Check gunicorn is installed
pip install gunicorn

# Test production server locally
gunicorn src.app:app --bind 0.0.0.0:8000
```

---

## Production Checklist

- [ ] Authentication module created (`src/auth.py`)
- [ ] Flask app updated with authentication
- [ ] Requirements.txt updated with new packages
- [ ] All routes protected with `@login_required`
- [ ] Register page template created
- [ ] Login page template created
- [ ] SECRET_KEY generated and set as environment variable
- [ ] Database initialized (first deployment)
- [ ] Deployment tested on Render
- [ ] Login/register flows tested
- [ ] Protected routes verified (require login)

---

## Next Steps After Deployment

1. **Backup User Database:**
   - Download `users.db` regularly
   - Store securely

2. **Monitor Performance:**
   - Check Render metrics dashboard
   - Monitor for errors in logs

3. **Scale If Needed:**
   - Upgrade Render plan if traffic increases
   - Consider PostgreSQL for production

4. **Security Enhancements:**
   - Add email verification
   - Implement password reset
   - Add 2FA (two-factor authentication)
   - Rate limiting on login attempts

---

## Troubleshooting Common Issues

### Issue: "Cannot connect to database"
**Solution**: Database is created on first run. If persistent issues, check file permissions on Render.

### Issue: "Login always fails"
**Solution**: 
- Verify user was registered (check database)
- Check SECRET_KEY is set in environment variables
- Clear browser cookies and try again

### Issue: "Website is very slow"
**Solution**:
- Upgrade Render plan
- Enable caching (already implemented in app)
- Optimize database queries

---

## Contact & Support

If you encounter issues:
1. Check Render logs: Dashboard → Your App → Logs tab
2. Check Flask logs locally for error details
3. Verify all environment variables are set
4. Make sure all files are pushed to GitHub before deploying

Happy deploying! 🚀
