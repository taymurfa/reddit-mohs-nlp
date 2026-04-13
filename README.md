# SubredditLDA

A cross-platform desktop application built with Electron and Python to collect Reddit data and run LDA topic modeling on subreddits.

## Setup for Development

1. **Install Node JS** (v18+) and **Python** (3.10+)
2. **Clone the repo**
3. **Install JS dependencies:**
   `npm install`
4. **Set up Python environment:**
   Create a virtual environment: `python -m venv venv`
   Activate the virtual environment:
     - Windows: `venv\Scripts\activate`
     - Mac/Linux: `source venv/bin/activate`
   Install Python dependencies: `pip install -r requirements.txt`
5. **Set up credentials:**
   Rename `.env.example` to `.env` and insert your Reddit API credentials.
6. **Run development version:**
   `npm run dev`

## Setup for End Users
Download and run the provided `.dmg` (Mac) or `.exe` installer (Windows) from the `dist/` folder. No need to install Python or Node.js.
