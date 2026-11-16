import os

import uvicorn

from spotify_popularity.api import app

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "9696")))
