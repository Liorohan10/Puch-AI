# Keep-Alive Solutions for Render Deployment

This document explains how to prevent your MCP server from going idle on Render's free tier.

## 🔧 Solutions Implemented

### 1. Internal Self-Ping System ⭐ (Recommended)

The MCP server now includes a built-in keep-alive system that automatically pings itself every 10 minutes.

**Features:**
- Automatically detects Render environment
- Self-pings every 10 minutes
- Runs in background thread
- New health check endpoints: `/health` and `/server_status`

**No additional setup required** - this runs automatically when deployed on Render!

### 2. External Keep-Alive Script

Run the external Python script from any server/computer:

```bash
python keep_alive_external.py
```

**Environment Variables:**
- `TARGET_URL`: Your Render service URL (default: https://puch-ai-ssnl.onrender.com)
- `PING_INTERVAL_MINUTES`: Ping interval in minutes (default: 10)

### 3. GitHub Actions Workflow ⭐ (Recommended)

The repository includes a GitHub Actions workflow that pings your service every 10 minutes.

**Setup:**
1. Push the `.github/workflows/keep-alive.yml` file to your repository
2. Enable GitHub Actions in your repository settings
3. The workflow runs automatically every 10 minutes

**Manual trigger:** Go to Actions tab → "Keep Render Service Alive" → "Run workflow"

### 4. Windows Batch Script

For Windows users, run the batch script locally:

```cmd
keep_alive_windows.bat
```

This will ping your service every 10 minutes and log the results.

## 🚀 Deployment Instructions

### For Render:

1. **Update your repository** with the new code
2. **Redeploy on Render** - the internal keep-alive will start automatically
3. **Enable GitHub Actions** for additional redundancy

### Environment Variables on Render:

Add this environment variable to your Render service:
- `RENDER_SERVICE_URL`: `https://puch-ai-ssnl.onrender.com`

## 📊 Monitoring

### Health Check Endpoints

Your service now provides monitoring endpoints:

- **Health Check**: `https://puch-ai-ssnl.onrender.com/health`
- **Server Status**: `https://puch-ai-ssnl.onrender.com/server_status`

### Example Health Check Response:

```json
{
  "status": "healthy",
  "timestamp": "2025-08-10T12:30:45.123456",
  "server": "Puch AI MCP Server",
  "version": "1.0.0",
  "last_self_ping": "2025-08-10T12:30:00.000000",
  "keep_alive_active": true
}
```

## 🔧 Troubleshooting

### If Keep-Alive Isn't Working:

1. **Check Render logs** for keep-alive messages:
   - Look for: "🔄 Keep-alive started"
   - Look for: "✅ Keep-alive ping successful"

2. **Test health endpoints manually**:
   ```bash
   curl https://puch-ai-ssnl.onrender.com/health
   ```

3. **Verify GitHub Actions**:
   - Go to your repo's Actions tab
   - Check if the "Keep Render Service Alive" workflow is running

### Common Issues:

- **Cold starts**: First ping after idle may take 30+ seconds
- **Network timeouts**: Keep-alive uses 30-second timeout
- **Rate limiting**: Internal ping uses 10-minute intervals to avoid abuse

## 📈 Best Practices

1. **Use multiple methods**: Combine internal + GitHub Actions for redundancy
2. **Monitor regularly**: Check health endpoints occasionally
3. **Update intervals**: Adjust ping intervals based on your needs
4. **Log monitoring**: Check Render logs for keep-alive activity

## 🎯 Expected Behavior

- **Idle prevention**: Service stays warm 24/7
- **Auto-recovery**: Handles temporary network issues
- **Minimal overhead**: Lightweight HTTP pings
- **Logging**: Clear logs for monitoring

Your MCP server should now stay active indefinitely! 🚀
