# AutoDL Setup Guide: Using a Forked Repository

This guide outlines the standard workflow for setting up your project on a remote server like AutoDL, ensuring your local code changes are correctly synced. Manually uploading files is discouraged as it is error-prone.

The core idea is to use your own fork of the `openpi` repository as a central point for your changes.

---

### Step 1: On Your Local Machine (Prepare Your Code)

First, you need to commit your local changes and push them to your personal fork on GitHub.

#### A. Fork the `openpi` Repository

1.  Navigate to the original `openpi` repository on GitHub.
2.  Click the **"Fork"** button in the top-right corner to create a personal copy under your account.

#### B. Point Your Local Repository to Your Fork

Your local clone is likely configured to point to the original `openpi` repository. You need to update it to point to your new fork.

```bash
# Replace <YOUR_GITHUB_USERNAME> with your actual username
git remote set-url origin https://github.com/<YOUR_GITHUB_USERNAME>/openpi.git
```

#### C. Commit and Push Your Local Changes

Now, commit the changes you've made locally and push them to your fork.

```bash
# Stage all your changes
git add .

# Commit them with a descriptive message
git commit -m "My local changes for AutoDL training"

# Push the commit to your fork's main branch
git push origin main
```

---

### Step 2: On the AutoDL Server (Deploy Your Code)

With your changes now on your GitHub fork, you can easily pull them down to the remote server.

#### A. Clone Your Fork

Connect to your AutoDL instance and clone *your forked repository*.

```bash
# Replace <YOUR_GITHUB_USERNAME> with your actual username
git clone https://github.com/<YOUR_GITHUB_USERNAME>/openpi.git
cd openpi
```

If you had already cloned the original repository, you can navigate into its directory and run `git pull https://github.com/<YOUR_GITHUB_USERNAME>/openpi.git main` to pull in your changes.

---

### Result

The `openpi` directory on your AutoDL server now contains the exact version of the code from your local machine, including all your recent changes. You can now proceed with installing dependencies and running your training scripts.
