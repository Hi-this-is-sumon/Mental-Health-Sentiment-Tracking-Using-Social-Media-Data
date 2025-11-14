#!/usr/bin/env python3
"""
Quick Setup Script for Authentication
Run this before deploying to Render
"""

import os
import sys
import subprocess
import secrets

def generate_secret_key():
    """Generate a secure SECRET_KEY"""
    return secrets.token_hex(32)

def create_env_file():
    """Create .env file with configuration"""
    secret_key = generate_secret_key()
    
    env_content = f"""# Flask Configuration
SECRET_KEY={secret_key}
FLASK_ENV=production
PYTHONUNBUFFERED=1

# Database
DATABASE_URL=sqlite:///users.db

# Security
SESSION_COOKIE_SECURE=True
SESSION_COOKIE_HTTPONLY=True
SESSION_COOKIE_SAMESITE=Lax
"""
    
    with open('.env', 'w') as f:
        f.write(env_content)
    
    print("✓ Created .env file")
    print(f"✓ Generated SECRET_KEY: {secret_key[:16]}...")
    return secret_key

def install_dependencies():
    """Install required packages"""
    print("\n📦 Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
        print("✓ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("✗ Failed to install dependencies")
        return False

def verify_files():
    """Verify all necessary files exist"""
    required_files = [
        'src/auth.py',
        'templates/login.html',
        'templates/register.html',
        'requirements.txt',
        'AUTHENTICATION_GUIDE.md'
    ]
    
    print("\n🔍 Verifying files...")
    all_exist = True
    for file in required_files:
        if os.path.exists(file):
            print(f"✓ {file}")
        else:
            print(f"✗ {file} - MISSING!")
            all_exist = False
    
    return all_exist

def print_next_steps():
    """Print next steps for deployment"""
    print("\n" + "="*60)
    print("✓ AUTHENTICATION SETUP COMPLETE!")
    print("="*60)
    print("\n📋 NEXT STEPS:\n")
    print("1. TEST LOCALLY:")
    print("   python -c \"from src.app import app; app.run(debug=True)\"")
    print("   → Visit http://localhost:5000/register")
    print("   → Create a test account and login\n")
    
    print("2. PUSH TO GITHUB:")
    print("   git add .")
    print("   git commit -m \"Add authentication system\"")
    print("   git push origin main\n")
    
    print("3. SET ENVIRONMENT VARIABLES ON RENDER:")
    print("   Dashboard → Your App → Environment")
    print("   Add these variables:")
    print("   • SECRET_KEY: [copy from .env]")
    print("   • FLASK_ENV: production")
    print("   • PYTHONUNBUFFERED: 1\n")
    
    print("4. DEPLOY:")
    print("   → Render will automatically redeploy on GitHub push")
    print("   → Check Logs tab for any issues\n")
    
    print("5. VERIFY DEPLOYMENT:")
    print("   → Visit https://your-app.onrender.com/register")
    print("   → Create account and test login flow\n")
    
    print("📚 For detailed guide, see: AUTHENTICATION_GUIDE.md")
    print("="*60 + "\n")

def main():
    """Main setup function"""
    print("🚀 Mental Health Sentiment Tracking - Authentication Setup\n")
    
    # Verify files exist
    if not verify_files():
        print("\n✗ Some files are missing!")
        return False
    
    # Create .env file
    secret_key = create_env_file()
    
    # Install dependencies
    if not install_dependencies():
        return False
    
    # Print next steps
    print_next_steps()
    
    return True

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
