import os

# Render sets the PORT environment variable automatically
port = os.environ.get("PORT", "10000")
bind = f"0.0.0.0:{port}"

# Increase timeout for slow AI imports on free tier
timeout = 120
workers = 1

# Output logs to the console so we can see what's happening
accesslog = "-"
errorlog = "-"
loglevel = "debug"
