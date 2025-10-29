# omnimarkai

One Platform. Every Channel. Infinite Reach.

Full‑stack AI web application that unifies SEO, AEO, and SMM metrics into one intelligent dashboard with simulated AI workflows.

## Tech Stack
- Frontend: React (Vite) + TailwindCSS + Framer Motion + Recharts
- Backend: Flask (Python) + scikit‑learn + NLTK
- Database (optional logs): MongoDB Atlas or Firebase
- Hosting: Frontend (Vercel/Netlify), Backend (Render/Railway)

## Contact
- Support: 06bhardwaj@gmail.com

## Local Development

### Backend
```bash
# From OmniMarkAI/backend
python -m venv .venv
. .venv/Scripts/Activate.ps1
pip install -r requirements.txt
python app.py
```
The backend runs at http://127.0.0.1:5000

### Frontend
```bash
# From OmniMarkAI/frontend
npm install
npm run dev
```
The frontend runs at http://localhost:5173

## New Analytics API
- GET `/sources/popular?platform=instagram|youtube|all&limit=10` → simulated popular creators
- POST `/analytics/ingest` → send JSON array of records; saved to `backend/data/custom.csv`
  - Auto-derives `EngagementRate` if `Likes, Comments, Impressions` exist
  - Auto-derives `ROI` if `Revenue, Cost` exist
- POST `/analytics/ingest_csv` → upload CSV file as `file` (multipart)
- GET `/analytics/summary` → aggregate stats (rows, platform means, monthly series)
- GET `/analytics/model` → trains LinearRegression and returns coefficients + R2
- GET `/analytics/best_times?platform=all&limit=10` → best day/hour per platform and creator suggestions

Example ingest body:
```json
[
  {"Date":"2025-01-01","Platform":"Instagram","Impressions":12000,"Likes":900,"Comments":120,"Followers":45000,"Revenue":1200,"Cost":420},
  {"Date":"2025-01-02","Platform":"YouTube","Impressions":24000,"Likes":1100,"Comments":140,"Followers":75000,"Revenue":1800,"Cost":700}
]
```

## Deployment
- Backend: Render/Railway. Start: `gunicorn app:app`.
- Frontend: Netlify/Vercel. Build: `npm run build`, Output: `dist`.

## Notes
- `SECRET_KEY`, `ADMIN_USER`, `ADMIN_PASS` can be set as env vars for admin.
- The backend auto‑generates `data/marketing_data_1500.csv` if missing.
- Update `frontend/src/api/flaskAPI.js` `BASE_URL` for deployed backend.
