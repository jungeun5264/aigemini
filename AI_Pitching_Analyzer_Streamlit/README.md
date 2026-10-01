# AI Pitching Analyzer - Streamlit

## Files
- `streamlit_app.py`: main web app
- `requirements.txt`: Python dependencies
- `.streamlit/config.toml`: upload/theme config

## Deploy on Streamlit Community Cloud
1. Create a GitHub repository and upload these files.
2. Go to https://share.streamlit.io and create an app from the repository.
3. Set the entrypoint to `streamlit_app.py`.
4. In **Advanced settings → Secrets**, add:

```toml
GEMINI_API_KEY = "YOUR_REAL_GEMINI_API_KEY"
MODEL = "gemini-3.1-flash-lite"
```

Do not commit your API key to GitHub.

If your API key supports a newer model and you want to change it, edit only the `MODEL` secret.
