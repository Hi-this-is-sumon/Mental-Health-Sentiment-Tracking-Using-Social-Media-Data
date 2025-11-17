# OAuth Setup (Server-side) — GitHub & Google

Your login page now uses **server-side OAuth (Authlib)** to support GitHub and Google sign-in without storing user data.

## Current Status

- ✅ Server-side OAuth routes created
- ✅ Login page displays (shows credentials status)
- ⏳ **YOU NEED TO:** Register OAuth apps and paste credentials into `.env`

---

## Quick Setup (3 Steps)

### Step 1: Register GitHub OAuth App
1. Go to: https://github.com/settings/developers → **OAuth Apps**
2. Click **"New OAuth App"**
3. Fill in:
   - **Application name**: `MindSentiment`
   - **Homepage URL**: `http://localhost:5000`
   - **Authorization callback URL**: `http://localhost:5000/auth/github/callback`
4. Click **"Register application"**
5. Copy your **Client ID** and **Client Secret**

### Step 2: Register Google OAuth Client
1. Go to: https://console.cloud.google.com/apis/credentials
2. Click **"Create Credentials"** → **"OAuth 2.0 Client ID"** → **"Web application"**
3. Under **"Authorized redirect URIs"**, add:
   - `http://localhost:5000/auth/google/callback`
4. Click **"Create"**
5. Copy your **Client ID** and **Client Secret**

### Step 3: Add Credentials to `.env`
1. Open `.env` file in your project root (already created)
2. Fill in:
   ```
   GITHUB_CLIENT_ID=your_github_client_id
   GITHUB_CLIENT_SECRET=your_github_client_secret
   GOOGLE_CLIENT_ID=your_google_client_id
   GOOGLE_CLIENT_SECRET=your_google_client_secret
   SECRET_KEY=dev-secret
   ```
3. **Save the file**
4. **Restart Flask server** (Ctrl+C and run again)

---

## How It Works

1. User clicks "Continue with GitHub" or "Continue with Google" on `/login` page
2. App redirects to GitHub/Google login page
3. User authenticates with their account
4. GitHub/Google redirects back to `/auth/github/callback` or `/auth/google/callback`
5. Server exchanges the code for user profile info
6. Flask creates a **session** (no database needed)
7. User is redirected to `/analyze` page
8. Session persists until logout

---

## Testing Locally

1. **Add credentials to `.env`** (see Step 3 above)
2. **Start Flask server**:
   ```powershell
   cd "C:\Users\HP\Desktop\New folder\Mental health sentiment tracking using social media data"
   .venv\Scripts\python.exe src/app.py
   ```
3. **Visit** http://localhost:5000/login
4. Click **GitHub** or **Google** button
5. You should see the provider's login page, then get redirected to `/analyze`

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Buttons are grayed out on `/login` | Credentials not in `.env`. Check GitHub/Google client IDs are filled in |
| "invalid_client" error from Google | Wrong Google Client ID/Secret OR redirect URI mismatch |
| "Page not found - GitHub" error | Wrong GitHub Client ID/Secret OR redirect URI mismatch |
| Server says "credentials not found" | `.env` file is empty or not in the project root |

---

## Security Notes

- `.env` is in `.gitignore` (secrets won't be committed)
- Do **NOT** share your client secret
- For production (Render), use environment variables in the platform settings
- Sessions are stored in Flask memory (expires when server restarts)

---

## Files Modified

- `src/app.py` → Added OAuth routes and credential checking
- `templates/login.html` → Shows credential status
- `requirements.txt` → Added `python-dotenv`
- `.env` → Created (empty, you fill it in)
- `.env.example` → Reference file