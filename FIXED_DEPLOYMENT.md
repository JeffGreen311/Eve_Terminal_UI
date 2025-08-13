# EVE Terminal Deployment - Fixed Configuration

## The Problem (Now Fixed)

Your deployment was failing because:
1. The `.replit` file was pointing to `web_run_new:app` but the original file is `web_run.py`
2. There was no error handling for missing dependencies in the imports

## What Has Been Fixed

1. **Updated the `.replit` file** to correctly point to `web_run.py`:
   ```toml
   [deployment]
   run = ["sh", "-c", "pip install flask requests psycopg2-binary gunicorn python-dotenv && gunicorn --bind 0.0.0.0:8888 --timeout 30 --workers 1 web_run:app"]
   ```

2. **Added dependency handling** to the beginning of `web_run.py`:
   - Auto-installs critical packages if they're missing
   - Provides fallbacks for imports that fail
   - Adds proper error reporting

## How to Deploy

1. Upload these updated files to your Replit project:
   - The corrected `.replit` file
   - The updated `web_run.py` with dependency handling

2. Deploy the application in Replit. It should now correctly:
   - Install all required dependencies before starting
   - Use the correct entry point (`web_run.py`)
   - Handle missing dependencies gracefully

## Troubleshooting

If you still encounter issues:

1. Check the Replit logs to see what specific error is occurring
2. Verify that `web_run.py` exists in your Replit project (not renamed)
3. Make sure your `web_run.py` has a Flask app variable called `app`

## Final Notes

- The deployment command now installs dependencies explicitly before starting Gunicorn
- The timeout has been increased to 30 seconds to allow for dependency installation
- The `.replit` file has been cleaned up to remove duplicate port configurations
