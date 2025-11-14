# Render Deployment Checklist

## Pre-Deployment (Local Setup)

- [ ] **Install dependencies**
  ```bash
  pip install -r requirements.txt
  ```

- [ ] **Run authentication setup**
  ```bash
  python setup_auth.py
  ```

- [ ] **Test authentication locally**
  ```bash
  python -c "from src.app import app; app.run(debug=True, port=5000)"
  ```
  - Navigate to http://localhost:5000/register
  - Create test account
  - Navigate to http://localhost:5000/login
  - Test login with credentials
  - Verify /analyze page is protected
  - Test logout

- [ ] **Verify database creation**
  - Check that `users.db` file is created
  - Confirm no errors in terminal

- [ ] **Test all protected routes**
  - [ ] /analyze (should redirect to login if not authenticated)
  - [ ] /predict (should be protected)
  - [ ] /export_pdf (should be protected)
  - [ ] /export_csv (should be protected)

- [ ] **Update Flask app.py** with authentication
  - [ ] Import Flask-Login and auth module
  - [ ] Add SECRET_KEY configuration
  - [ ] Initialize LoginManager
  - [ ] Add authentication routes (/login, /register, /logout)
  - [ ] Decorate protected routes with @login_required

- [ ] **Commit all changes to GitHub**
  ```bash
  git add .
  git commit -m "Add authentication system for production deployment"
  git push origin main
  ```

## Render Setup (Dashboard)

### Step 1: Create Render Account
- [ ] Visit https://render.com
- [ ] Sign up with GitHub (recommended)
- [ ] Authorize GitHub access

### Step 2: Create Web Service
- [ ] Click "New +" → Select "Web Service"
- [ ] Select your GitHub repository
- [ ] Confirm GitHub connection

### Step 3: Configure Service
- [ ] **Service Name**: `mental-health-sentiment`
- [ ] **Environment**: Python 3
- [ ] **Region**: Select closest region (e.g., Oregon)
- [ ] **Branch**: `main`
- [ ] **Build Command**: `pip install -r requirements.txt`
- [ ] **Start Command**: `gunicorn src.app:app`

### Step 4: Set Environment Variables
- [ ] Click "Advanced" → "Add Environment Variable"
- [ ] Add each variable:

| Key | Value | Source |
|-----|-------|--------|
| `SECRET_KEY` | (copy from .env file) | Generate: `python -c "import secrets; print(secrets.token_hex(32))"` |
| `FLASK_ENV` | `production` | Fixed value |
| `PYTHONUNBUFFERED` | `1` | Fixed value |

### Step 5: Configure Plan
- [ ] **Tier**: Free (for testing) or Paid (for production)
- [ ] **Region**: Closest to your users
- [ ] Keep "Auto-Deploy" enabled (redeploys on GitHub push)

### Step 6: Create Web Service
- [ ] Review all settings
- [ ] Click "Create Web Service"
- [ ] **Wait 5-10 minutes** for initial deployment

## Post-Deployment Testing

- [ ] **Check deployment status**
  - View Logs tab for any errors
  - Should see "Build succeeded" and "Server listening on port 10000"

- [ ] **Test registration**
  - Navigate to `https://your-app.onrender.com/register`
  - Create new account
  - Verify success message
  - Verify redirect to login

- [ ] **Test login**
  - Navigate to `https://your-app.onrender.com/login`
  - Enter credentials from test account
  - Verify login success
  - Verify redirect to /analyze

- [ ] **Test protected routes**
  - [ ] /analyze (should be accessible after login)
  - [ ] /predict (try sending text for sentiment analysis)
  - [ ] /export_pdf (try exporting PDF)
  - [ ] /export_csv (try exporting CSV)

- [ ] **Test logout**
  - Click logout button
  - Verify redirect to login page
  - Verify can't access /analyze directly

- [ ] **Test with different browser/incognito**
  - Ensures cookies/sessions work correctly

- [ ] **Monitor performance**
  - Check Render dashboard for resource usage
  - Look for any error patterns in logs

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'flask_login'"
**Solution**: 
- Check requirements.txt has `flask-login`
- Push changes to GitHub
- Render will automatically redeploy

### Issue: "SECRET_KEY not configured"
**Solution**:
- Go to Render dashboard
- Environment tab
- Add SECRET_KEY variable (don't leave empty)

### Issue: "Application failed to start"
**Solution**:
- Check Logs tab for specific error
- Verify build command succeeded
- Ensure gunicorn is in requirements.txt
- Check for Python syntax errors in code

### Issue: "Database locked" or "Cannot create users.db"
**Solution**:
- This is normal on first run
- Database will be created automatically
- Check again in 1-2 minutes

### Issue: "Login always fails"
**Solution**:
- Verify user was registered (check logs for "User registered successfully")
- Clear browser cookies
- Try with different username/password
- Check SECRET_KEY is set correctly

### Issue: "Static files not loading (CSS/JS)"
**Solution**:
- Verify static files are in GitHub repository
- Check Logs for 404 errors
- Ensure `static/` folder is committed

## Security Checklist

- [ ] SECRET_KEY is unique and secure (32+ characters)
- [ ] FLASK_ENV is set to `production` (not debug)
- [ ] Database file (users.db) is not exposed
- [ ] Password hashing is using werkzeug
- [ ] All sensitive routes have @login_required
- [ ] HTTPS is enforced (Render does this automatically)
- [ ] Regular backups of users.db database
- [ ] Monitor logs for suspicious activity

## Performance Optimization

- [ ] Cache is configured (already in app.py)
- [ ] Rate limiting is enabled (already in app.py)
- [ ] Database queries are optimized
- [ ] Static files are gzipped (Render handles this)
- [ ] Consider CDN for static files (advanced)

## Maintenance Tasks

After deployment, do these regularly:

- [ ] **Weekly**: Check Render logs for errors
- [ ] **Weekly**: Monitor resource usage
- [ ] **Monthly**: Backup users.db database
- [ ] **Monthly**: Review active sessions
- [ ] **As needed**: Update dependencies in requirements.txt

## Scaling for Production

When ready for production, consider:

1. **Upgrade Render Plan**: Free tier has 15-minute inactivity limit
2. **Switch to PostgreSQL**: Render provides managed PostgreSQL
3. **Enable SSL**: Already done by Render
4. **Setup monitoring**: Use Render's metrics dashboard
5. **Add error tracking**: Sentry or similar service
6. **Setup backups**: Automated database backups
7. **Implement 2FA**: Two-factor authentication for users

## Contact & Support

- **Render Docs**: https://render.com/docs
- **Render Support**: https://render.com/support
- **Flask Documentation**: https://flask.palletsprojects.com
- **Flask-Login Docs**: https://flask-login.readthedocs.io
