# Firebase Setup Guide - GitHub & Google OAuth Only

## ✅ Your Login Page is Ready!

The login page now has **GitHub and Google buttons only** - no email/password.

Your updated login page is at: `templates/login.html`

---

## 🔴 WHAT YOU NEED TO DO (One-Time Setup)

### Step 1: Create a Firebase Project

1. Go to https://console.firebase.google.com
2. Click **"+ Add project"**
3. Name it: `MindSentiment`
4. Click **"Create project"** (wait ~1 minute)
5. Click **"Continue"** when done

---

### Step 2: Get Your Firebase Config

1. Click the **⚙️ (Settings)** icon (top-left, next to "Project Overview")
2. Click **"Project Settings"**
3. Scroll down to **"Your apps"** section
4. Look for a **Web icon `</>`** - click it
5. A window will show your `firebaseConfig` object - **COPY THE ENTIRE CONFIG**

It looks like this:
```javascript
{
  apiKey: "AIzaSyD...",
  authDomain: "mindsent-xxxxx.firebaseapp.com",
  projectId: "mindsent-xxxxx",
  storageBucket: "mindsent-xxxxx.appspot.com",
  messagingSenderId: "123456789",
  appId: "1:123456789:web:abcd1234ef"
}
```

---

### Step 3: Paste Config Into Your Login Page

1. Open `templates/login.html` in your text editor
2. Find this line (around line 170-175):
   ```javascript
   const firebaseConfig = {
       apiKey: "YOUR_API_KEY",
       authDomain: "YOUR_AUTH_DOMAIN",
       projectId: "YOUR_PROJECT_ID",
       ...
   };
   ```
3. **Replace the entire object** with what you copied from Firebase
4. **Save the file**

---

### Step 4: Enable GitHub OAuth in Firebase

1. In Firebase Console, go to **"Authentication"** (left menu)
2. Click **"Sign-in method"** tab
3. Click **"GitHub"**
4. Toggle it **ON**
5. A form appears asking for:
   - **GitHub Client ID**
   - **GitHub Client Secret**

#### How to get GitHub OAuth credentials:

1. Go to https://github.com/settings/developers
2. Click **"OAuth Apps"** → **"New OAuth App"**
3. Fill in:
   - **Application name**: `MindSentiment`
   - **Homepage URL**: `http://localhost:5000` (for local testing)
   - **Authorization callback URL**: Copy from Firebase (it shows in the Firebase form)
4. Click **"Register application"**
5. Copy **Client ID** and **Client Secret** → paste into Firebase
6. Click **"Save"** in Firebase

---

### Step 5: Enable Google OAuth in Firebase

1. In Firebase Console, stay in **"Authentication"** → **"Sign-in method"**
2. Click **"Google"**
3. Toggle it **ON**
4. Enter:
   - **Project name**: `MindSentiment`
   - **Project support email**: (any Gmail you own)
5. Click **"Save"**

**That's it!** Google doesn't need extra setup for local testing.

---

## 🚀 Test It

1. Make sure Flask is running:
   ```bash
   cd "C:\Users\HP\Desktop\New folder\Mental health sentiment tracking using social media data"
   .venv\Scripts\python.exe src/app.py
   ```

2. Visit: http://localhost:5000/login

3. Click **GitHub** or **Google** button

4. If configured correctly:
   - A popup opens
   - You authenticate with GitHub/Google
   - You get redirected to `/analyze`
   - You see "Welcome, [Your Name]" at the top

---

## ❌ Troubleshooting

| Problem | Solution |
|---------|----------|
| "Firebase config not set" message | You didn't paste your Firebase config into login.html |
| Buttons don't open popup | Firebase config is still placeholder (`YOUR_API_KEY` etc.) |
| "GitHub login failed" | GitHub OAuth not enabled in Firebase Console OR wrong credentials |
| "Google login failed" | Google OAuth not enabled in Firebase Console |
| Can't find OAuth apps section on GitHub | Go to https://github.com/settings/developers (must be logged in) |

---

## 📝 Notes

- **No server database**: User data is stored ONLY in Firebase, not on your server ✅
- **Session stored locally**: Browser keeps user logged in using localStorage ✅
- **GitHub & Google only**: No email/password login ✅
- **For Render deployment**: Add Firebase config as environment variable (later)

---

## 🎯 Quick Summary

| Step | Action |
|------|--------|
| 1️⃣ | Create Firebase project (console.firebase.google.com) |
| 2️⃣ | Copy Firebase config |
| 3️⃣ | Paste into `templates/login.html` |
| 4️⃣ | Enable GitHub & Google in Firebase Console |
| 5️⃣ | Set up GitHub OAuth app (for GitHub auth in Firebase) |
| 6️⃣ | Test at http://localhost:5000/login |

**Done!** Your login page will work with real GitHub/Google authentication.
