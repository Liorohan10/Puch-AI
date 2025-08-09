
# Entry point for Vercel Python Serverless Function
import importlib.util
import os
import sys

module_path = os.path.join(os.path.dirname(__file__), '..', 'mcp-bearer-token', 'mcp_starter.py')
spec = importlib.util.spec_from_file_location("mcp_starter", module_path)
mcp_starter = importlib.util.module_from_spec(spec)
sys.modules["mcp_starter"] = mcp_starter
spec.loader.exec_module(mcp_starter)

# If you need to define a handler for Vercel, do it here
# For example, if using FastAPI or Flask, expose as 'app'
# If plain Python, ensure the function responds to HTTP events

# ...existing code...
