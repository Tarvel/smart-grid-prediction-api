"""
Compatibility entry point for the prediction service.

The implementation lives in new_prediction_api.py and now uses the latest
model artifacts from New folder.
"""

import os

import uvicorn

from new_prediction_api import *  # noqa: F401,F403


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    uvicorn.run(app, host="0.0.0.0", port=port)
