# 🚀 Quick Deployment Guide

Get your TI-Tool app deployed to Streamlit Community Cloud in ~15 minutes.

## 📋 What You Need

Before you start, gather these credentials:

1. **AWS Account** 
   - Access Key ID
   - Secret Access Key
   
2. **Azure OpenAI** (or OpenAI)
   - API Key
   - Endpoint URL
   - Deployment Name
   
3. **GitHub Account**
   - Repository access

---

## 🎯 5-Step Deployment

### Step 1: AWS S3 Setup (5 min)

1. Go to [AWS S3 Console](https://s3.console.aws.amazon.com/)
2. Click **Create bucket**
   - Name: `ti-tool-s3-storage`
   - Region: `us-east-1` (or your preferred region)
   - Click **Create bucket**

3. Create folders in your bucket:
   - `crawled_data/`
   - `processed_data/`
   - `summarised_content/`
   - `rag_storage/`

4. Go to [IAM Console](https://console.aws.amazon.com/iam/)
5. Create user:
   - Click **Users** → **Create user**
   - Username: `ti-tool-streamlit`
   - Attach policy: `AmazonS3FullAccess`
   - Click **Create user**

6. Create access key:
   - Click on user → **Security credentials** tab
   - Click **Create access key**
   - Application type: **Application running outside AWS**
   - **SAVE THESE CREDENTIALS** ⚠️

---

### Step 2: Push to GitHub (2 min)

```bash
# In your project directory
git init
git add .
git commit -m "Initial commit for Streamlit Cloud"

# Create new repo on GitHub, then:
git remote add origin https://github.com/YOUR_USERNAME/ti-tool-app.git
git branch -M main
git push -u origin main
```

**Verify**: `.streamlit/secrets.toml` should NOT be in your repo (check `.gitignore`)

---

### Step 3: Deploy on Streamlit Cloud (2 min)

1. Go to [share.streamlit.io](https://share.streamlit.io/)
2. Sign in with GitHub
3. Click **New app**
4. Select:
   - Repository: `YOUR_USERNAME/ti-tool-app`
   - Branch: `main`
   - Main file: `app.py`
5. Click **Deploy!**

Wait 2-3 minutes for initial deployment...

---

### Step 4: Add Secrets (3 min)

1. Once deployed, click your app name
2. Click **⋮** menu → **Settings**
3. Go to **Secrets** tab
4. Copy and paste this template, filling in YOUR values:

```toml
[LLM_PROVIDER]
PROVIDER = "azure"

[AZURE_OPENAI]
AZURE_OPENAI_API_KEY = "paste-your-azure-api-key"
AZURE_OPENAI_ENDPOINT = "https://your-resource.openai.azure.com/"
AZURE_OPENAI_DEPLOYMENT_NAME = "gpt-4o-mini"
AZURE_OPENAI_API_VERSION = "2024-02-15-preview"

[AWS]
AWS_ACCESS_KEY_ID = "paste-your-access-key-id"
AWS_SECRET_ACCESS_KEY = "paste-your-secret-access-key"
AWS_DEFAULT_REGION = "us-east-1"
S3_BUCKET_NAME = "ti-tool-s3-storage"

[SEARXNG]
SEARXNG_URL = "http://your-searxng-instance:8080"

[LINKEDIN]
LINKEDIN_EMAIL = "your-email@example.com"
LINKEDIN_PASSWORD = "your-password"
```

5. Click **Save**
6. App will restart automatically (~30 seconds)

---

### Step 5: Test Your App (3 min)

Open your app URL (something like `https://your-app.streamlit.app`)

Test these features:
- [ ] Navigate between pages (no errors)
- [ ] Web Crawler: Start a small crawl (5-10 pages)
- [ ] Check S3 bucket: Files should appear in `crawled_data/`
- [ ] URL Filtering: Filter the crawled data
- [ ] Database: View saved data

✅ **You're live!**

---

## 🔧 Troubleshooting

### App won't start
- Check logs (click **Manage app** → view logs)
- Verify all secrets are filled in
- Check for typos in secret keys

### S3 Access Denied
- Verify AWS credentials are correct
- Check IAM user has S3 permissions
- Verify bucket name matches in secrets

### Azure OpenAI Error
- Verify API key is correct
- Check endpoint URL format
- Ensure deployment name is correct

---

## 📚 Next Steps

- Read full guide: [STREAMLIT_CLOUD_DEPLOYMENT.md](STREAMLIT_CLOUD_DEPLOYMENT.md)
- Use checklist: [DEPLOYMENT_CHECKLIST.md](DEPLOYMENT_CHECKLIST.md)
- Check main docs: [README.md](README.md)

---

## 💡 Pro Tips

1. **Test locally first**: Run `streamlit run app.py` before deploying
2. **Use .env locally**: Copy `.env.example` to `.env` for local development
3. **Monitor costs**: Check AWS billing for S3 usage
4. **Free tier limits**: Streamlit Cloud free tier = 1 private app
5. **Update easily**: Just `git push` to update your deployed app

---

## 📊 Estimated Costs

- **Streamlit Cloud**: $0/month (free tier)
- **AWS S3**: ~$1-5/month (small usage)
- **Azure OpenAI**: ~$5-20/month (moderate usage)
- **Total**: ~$6-30/month

---

## ✅ Success Checklist

- [ ] S3 bucket created with folders
- [ ] IAM user created with access keys
- [ ] Code pushed to GitHub
- [ ] App deployed on Streamlit Cloud
- [ ] Secrets configured correctly
- [ ] Test crawl successful
- [ ] Files appearing in S3
- [ ] App URL shared with team

---

**Need help?** Check the detailed guide or Streamlit docs:
- [Streamlit Community Forum](https://discuss.streamlit.io/)
- [AWS S3 Documentation](https://docs.aws.amazon.com/s3/)

**Estimated total time**: 15 minutes ⏱️
