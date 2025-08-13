# EVE Terminal Deployment Troubleshooting

If you're experiencing issues with Replit deployment, try these alternative configurations:

## Option 1: Use Direct Python Entry Point

Edit your `.replit` file to use the `run_direct.sh` script:

```toml
[deployment]
buildCommand = "bash build.sh"
run = ["bash", "run_direct.sh"]
publicPorts = [8888]
```

## Option 2: Simplify to Absolute Minimum

If you're still having issues, try this minimal `.replit` configuration:

```toml
run = "python3 app.py"

[deployment]
run = ["python3", "app.py"]
publicPorts = [8888]
```

## Option 3: Use a Procfile

Create a file named `Procfile` with:

```
web: gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --timeout 60
```

Then update `.replit` to:

```toml
[deployment]
run = []  # Empty to use Procfile
```

## Debugging

If deployment is still failing:

1. Check the deployment logs for specific error messages
2. Try running the app locally in the Replit environment first
3. Verify that app.py can be imported and run without errors
4. Make sure all dependencies are listed in requirements.txt

## Last Resort: Static Files

If nothing else works, you can deploy a static HTML file as a temporary solution:

1. Create a file named `index.html` with a message
2. Use a simple Python HTTP server in your `.replit` deployment:
   ```
   run = ["python3", "-m", "http.server", "8888"]
   ```
