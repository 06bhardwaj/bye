from flask import Flask, jsonify, request, session
from flask_cors import CORS
import os
import pandas as pd
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
import random
from werkzeug.security import generate_password_hash, check_password_hash
from data_sources import get_popular_users

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
DATA_PATH = os.path.join(DATA_DIR, 'marketing_data_1500.csv')
VISITS_PATH = os.path.join(DATA_DIR, 'visits.csv')
USERS_PATH = os.path.join(DATA_DIR, 'users.csv')
CUSTOM_PATH = os.path.join(DATA_DIR, 'custom.csv')

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret')
CORS(app, supports_credentials=True)

ADMIN_USER = os.environ.get('ADMIN_USER', 'AshokMBharadwaj')
ADMIN_PASS = os.environ.get('ADMIN_PASS', '189669')


def ensure_dataset():
	"""Create a synthetic dataset if it doesn't exist."""
	os.makedirs(DATA_DIR, exist_ok=True)
	if os.path.exists(DATA_PATH):
		return
	random.seed(42)
	start_date = datetime.now() - timedelta(days=180)
	platforms = ['Instagram', 'YouTube', 'LinkedIn', 'Twitter', 'TikTok']
	rows = []
	for i in range(1500):
		date = (start_date + timedelta(days=i % 180)).strftime('%Y-%m-%d')
		platform = random.choice(platforms)
		impressions = random.randint(1000, 20000)
		engagement_rate = round(random.uniform(0.01, 0.12), 4)
		followers = random.randint(1000, 200000)
		# Simulated ROI as a function of impressions and engagement with noise
		roi = round(0.0008 * impressions + 120 * engagement_rate + random.uniform(-5, 5), 2)
		rows.append([date, platform, impressions, engagement_rate, followers, roi])
	pd.DataFrame(rows, columns=['Date', 'Platform', 'Impressions', 'EngagementRate', 'Followers', 'ROI']).to_csv(DATA_PATH, index=False)


def ensure_users_file():
	os.makedirs(DATA_DIR, exist_ok=True)
	if not os.path.exists(USERS_PATH):
		with open(USERS_PATH, 'w', encoding='utf-8') as f:
			f.write('username,passwordHash,role\n')
		# ensure admin user exists
		df = pd.read_csv(USERS_PATH)
		if not (df['username'] == ADMIN_USER).any():
			admin_row = pd.DataFrame([{
				'username': ADMIN_USER,
				'passwordHash': generate_password_hash(ADMIN_PASS),
				'role': 'admin'
			}])
			pd.concat([df, admin_row], ignore_index=True).to_csv(USERS_PATH, index=False)


def save_visit(path: str, user_agent: str, ts_iso: str, ip: str):
	os.makedirs(DATA_DIR, exist_ok=True)
	header_needed = not os.path.exists(VISITS_PATH)
	with open(VISITS_PATH, 'a', encoding='utf-8') as f:
		if header_needed:
			f.write('timestamp,path,userAgent,ip\n')
		f.write(f'{ts_iso},{path.replace(","," ")},{user_agent.replace(","," ")},{ip}\n')


def current_user():
	return session.get('user')


def is_admin():
	user = current_user()
	return bool(user and user.get('role') == 'admin')


@app.route('/')
def root():
	return jsonify({"message": "omnimarkai backend running"})


# -------- User auth --------
@app.route('/auth/register', methods=['POST'])
def auth_register():
	ensure_users_file()
	data = request.get_json(silent=True) or {}
	username = (data.get('username') or '').strip()
	password = (data.get('password') or '').strip()
	if not username or not password:
		return jsonify({"ok": False, "error": "Missing username or password"}), 400
	df = pd.read_csv(USERS_PATH)
	if (df['username'] == username).any():
		return jsonify({"ok": False, "error": "User already exists"}), 400
	new_row = pd.DataFrame([{
		'username': username,
		'passwordHash': generate_password_hash(password),
		'role': 'user'
	}])
	pd.concat([df, new_row], ignore_index=True).to_csv(USERS_PATH, index=False)
	return jsonify({"ok": True})


@app.route('/auth/login', methods=['POST'])
def auth_login():
	ensure_users_file()
	data = request.get_json(silent=True) or {}
	username = (data.get('username') or '').strip()
	password = (data.get('password') or '').strip()
	df = pd.read_csv(USERS_PATH)
	row = df.loc[df['username'] == username]
	if row.empty:
		return jsonify({"ok": False, "error": "Invalid credentials"}), 401
	stored_hash = row.iloc[0]['passwordHash']
	role = row.iloc[0]['role']
	if not check_password_hash(stored_hash, password):
		return jsonify({"ok": False, "error": "Invalid credentials"}), 401
	session['user'] = { 'username': username, 'role': role }
	return jsonify({"ok": True, "role": role})


@app.route('/auth/logout', methods=['POST'])
def auth_logout():
	session.clear()
	return jsonify({"ok": True})


@app.route('/auth/me')
def auth_me():
	return jsonify(current_user() or {})


# -------- Admin auth (legacy) --------
@app.route('/admin/login', methods=['POST'])
def admin_login():
	# keep for compatibility; uses same users store
	return auth_login()


@app.route('/admin/logout', methods=['POST'])
def admin_logout():
	return auth_logout()


@app.route('/admin/me')
def admin_me():
	return jsonify({"isAdmin": is_admin()})


# -------- Visit tracking --------
@app.route('/track/visit', methods=['POST'])
def track_visit():
	data = request.get_json(silent=True) or {}
	path = (data.get('path') or '/')
	ua = request.headers.get('User-Agent', '')
	ts = data.get('timestamp') or datetime.utcnow().isoformat()
	ip = request.headers.get('X-Forwarded-For', request.remote_addr or '')
	try:
		save_visit(path, ua, ts, ip)
		return jsonify({"ok": True})
	except Exception as e:
		return jsonify({"ok": False, "error": str(e)}), 500


@app.route('/analytics/visits')
def analytics_visits():
	if not is_admin():
		return jsonify({"error": "Unauthorized"}), 401
	if not os.path.exists(VISITS_PATH):
		return jsonify({"total": 0, "byPath": []})
	df = pd.read_csv(VISITS_PATH)
	by_path = (
		df.groupby('path').size().reset_index(name='count')
		.sort_values('count', ascending=False)
		.head(20)
		.to_dict(orient='records')
	)
	return jsonify({"total": int(df.shape[0]), "byPath": by_path, "recent": df.tail(50).to_dict(orient='records')})


@app.route('/analyzeSEO')
def analyze_seo():
	return jsonify({"SEOscore": random.randint(70, 95), "topKeyword": random.choice(["organic coffee", "arabica beans", "fair trade coffee", "single-origin"])})


@app.route('/predictROI')
def predict_roi():
	ensure_dataset()
	df = pd.read_csv(DATA_PATH)
	X = df[['Impressions', 'EngagementRate']]
	y = df['ROI']
	model = LinearRegression().fit(X, y)
	pred = round(float(model.predict([[5000, 0.05]])[0]), 2)
	return jsonify({"PredictedROI": pred})


@app.route('/aiInsights')
def ai_insights():
	insights = {
		"AIInsight": "Post at 8 PM for ~18% better engagement.",
		"NextAction": "Add keyword 'organic coffee beans' to SEO tags.",
		"TrendingKeyword": random.choice(["organic coffee", "cold brew", "pour-over", "espresso grind"]),
		"BestTime": "8 PM"
	}
	return jsonify(insights)


@app.route('/logs')
def logs():
	sample_logs = [
		{"Time": "10:30 AM", "Decision": "Scheduled post", "Reason": "Engagement down 12%", "Action": "Post at 8 PM"},
		{"Time": "11:00 AM", "Decision": "Updated SEO tags", "Reason": "Keyword ranking drop", "Action": "Added trending tag"},
	]
	return jsonify(sample_logs)


@app.route('/kpis')
def kpis():
	ensure_dataset()
	seo = random.randint(72, 94)
	aeo = random.randint(68, 92)
	smm = random.randint(70, 96)
	pred_roi = random.uniform(8.0, 24.0)
	return jsonify({
		"SEO": seo,
		"AEO": aeo,
		"SMM": smm,
		"PredictedROI": round(pred_roi, 2)
	})


@app.route('/charts')
def charts():
	ensure_dataset()
	df = pd.read_csv(DATA_PATH)
	# Aggregate by month
	df['Date'] = pd.to_datetime(df['Date'])
	monthly = df.groupby(pd.Grouper(key='Date', freq='M')).agg({
		'Impressions': 'sum',
		'ROI': 'mean',
		'EngagementRate': 'mean'
	}).reset_index()
	engagement_series = [
		{"name": d['Date'].strftime('%b'), "Engagement": round(d['EngagementRate'] * 100, 2)}
		for _, d in monthly.iterrows()
	]
	platform_perf = df.groupby('Platform').agg({'Impressions': 'sum'}).reset_index()
	platform_series = [
		{"platform": row['Platform'], "Impressions": int(row['Impressions'])}
		for _, row in platform_perf.iterrows()
	]
	return jsonify({"engagement": engagement_series[-12:], "platforms": platform_series})


@app.route('/chatMock', methods=['POST'])
def chat_mock():
	data = request.get_json(silent=True) or {}
	prompt = (data.get('prompt') or '').strip().lower()
	if not prompt:
		return jsonify({"answer": "Please provide a question.", "reasoning": "No prompt provided.", "suggestedAction": None})
	# Simple heuristic responses
	if 'when' in prompt or 'time' in prompt:
		answer = 'Best time to post is 8 PM for your audience.'
		reasoning = 'Historical engagement peaks around 8 PM; recent trend confirms.'
		suggested = 'Schedule your next Instagram post at 8 PM today.'
	elif 'keyword' in prompt or 'seo' in prompt:
		answer = 'Try adding “organic coffee beans” and “fair trade coffee”.'
		reasoning = 'These keywords trend up and align with your content.'
		suggested = 'Update homepage meta title and tags with the suggested keywords.'
	else:
		answer = 'Focus on a carousel post with a CTA and link in bio.'
		reasoning = 'Carousel posts have 12–20% higher engagement in your category.'
		suggested = 'Draft a 5-slide carousel highlighting product USPs.'
	return jsonify({"answer": answer, "reasoning": reasoning, "suggestedAction": suggested})


@app.route('/sources/popular')
def sources_popular():
	platform = request.args.get('platform')
	limit = int(request.args.get('limit', '10'))
	return jsonify(get_popular_users(platform, limit))


@app.route('/analytics/ingest', methods=['POST'])
def analytics_ingest():
	# Accept JSON array of records; normalize and persist
	payload = request.get_json(silent=True)
	if not isinstance(payload, list) or len(payload) == 0:
		return jsonify({"ok": False, "error": "Expected JSON array of records"}), 400
	os.makedirs(DATA_DIR, exist_ok=True)
	df = pd.DataFrame(payload)
	# Basic normalization: derive EngagementRate if likes/comments/impressions present
	if 'EngagementRate' not in df.columns:
		if set(['Likes', 'Comments', 'Impressions']).issubset(df.columns):
			eng = (df['Likes'].fillna(0) + df['Comments'].fillna(0)) / df['Impressions'].replace(0, 1)
			df['EngagementRate'] = eng.clip(lower=0, upper=1.0)
	# If ROI missing but Revenue/Cost exist, compute
	if 'ROI' not in df.columns:
		if set(['Revenue', 'Cost']).issubset(df.columns):
			roi = ((df['Revenue'].fillna(0) - df['Cost'].fillna(0)) / df['Cost'].replace(0, 1)) * 100
			df['ROI'] = roi
	df.to_csv(CUSTOM_PATH, index=False)
	return jsonify({"ok": True, "rows": int(df.shape[0])})


@app.route('/analytics/ingest_csv', methods=['POST'])
def analytics_ingest_csv():
	"""Accept a CSV file upload (multipart/form-data, field name 'file'), normalize columns, save to custom.csv."""
	if 'file' not in request.files:
		return jsonify({"ok": False, "error": "No file provided"}), 400
	file = request.files['file']
	os.makedirs(DATA_DIR, exist_ok=True)
	try:
		df = pd.read_csv(file)
		# Normalize known columns from provided dataset
		# Map EngagementRate(%) -> EngagementRate (0..1)
		if 'EngagementRate(%)' in df.columns and 'EngagementRate' not in df.columns:
			df['EngagementRate'] = pd.to_numeric(df['EngagementRate(%)'], errors='coerce').fillna(0) / 100.0
		# Map ROI(%) -> ROI
		if 'ROI(%)' in df.columns and 'ROI' not in df.columns:
			df['ROI'] = pd.to_numeric(df['ROI(%)'], errors='coerce').fillna(0)
		# PredictedNextMonthROI(%) keep as-is (auxiliary)
		# Standardize Date
		if 'Date' in df.columns:
			df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
		# Ensure Impressions numeric if present, otherwise try to synthesize
		if 'Impressions' in df.columns:
			df['Impressions'] = pd.to_numeric(df['Impressions'], errors='coerce').fillna(0).astype(int)
		else:
			# Synthesize Impressions if Clicks present
			if 'Clicks' in df.columns:
				df['Impressions'] = (pd.to_numeric(df['Clicks'], errors='coerce').fillna(0) * 10).astype(int)
		# Followers
		if 'Followers' in df.columns:
			df['Followers'] = pd.to_numeric(df['Followers'], errors='coerce').fillna(0).astype(int)
		# Save normalized CSV
		df.to_csv(CUSTOM_PATH, index=False)
		return jsonify({"ok": True, "rows": int(df.shape[0]), "columns": list(df.columns)})
	except Exception as e:
		return jsonify({"ok": False, "error": str(e)}), 500


@app.route('/analytics/summary')
def analytics_summary():
	# Read custom first, fallback to default dataset
	path = CUSTOM_PATH if os.path.exists(CUSTOM_PATH) else DATA_PATH
	if not os.path.exists(path):
		return jsonify({"error": "No dataset found"}), 404
	df = pd.read_csv(path)
	stats = {}
	stats['rows'] = int(df.shape[0])
	for col in [c for c in ['Impressions', 'Followers', 'EngagementRate', 'ROI'] if c in df.columns]:
		series = df[col].dropna()
		stats[col] = {
			'min': float(series.min()) if not series.empty else 0,
			'max': float(series.max()) if not series.empty else 0,
			'mean': float(series.mean()) if not series.empty else 0,
		}
	by_platform = []
	if 'Platform' in df.columns:
		grp = df.groupby('Platform').agg({ c: 'mean' for c in df.columns if c in ['Impressions','EngagementRate','ROI'] }).reset_index()
		by_platform = grp.to_dict(orient='records')
	monthly = []
	if 'Date' in df.columns:
		df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
		m = df.dropna(subset=['Date']).groupby(pd.Grouper(key='Date', freq='M')).agg({
			'Impressions': 'sum' if 'Impressions' in df.columns else 'size',
			'EngagementRate': 'mean' if 'EngagementRate' in df.columns else 'size',
			'ROI': 'mean' if 'ROI' in df.columns else 'size'
		}).reset_index()
		monthly = [{ 'name': d['Date'].strftime('%b'),
			'Impressions': int(d.get('Impressions', 0)) if pd.notna(d.get('Impressions', 0)) else 0,
			'Engagement': round(float(d.get('EngagementRate', 0))*100,2) if pd.notna(d.get('EngagementRate', 0)) else 0,
			'ROI': round(float(d.get('ROI', 0)),2) if pd.notna(d.get('ROI', 0)) else 0 } for _, d in m.iterrows()]
	return jsonify({
		"source": os.path.basename(path),
		"stats": stats,
		"byPlatform": by_platform,
		"monthly": monthly[-12:]
	})


@app.route('/analytics/model')
def analytics_model():
	# Train a simple regression on the available dataset
	path = CUSTOM_PATH if os.path.exists(CUSTOM_PATH) else DATA_PATH
	if not os.path.exists(path):
		return jsonify({"error": "No dataset found"}), 404
	df = pd.read_csv(path)
	# Choose features/target based on availability
	features = []
	for c in ['Impressions', 'EngagementRate', 'Followers']:
		if c in df.columns:
			features.append(c)
	if 'ROI' in df.columns:
		target = 'ROI'
	elif 'EngagementRate' in df.columns and 'Impressions' in df.columns:
		target = 'EngagementRate'
	else:
		return jsonify({"error": "Not enough columns for modeling"}), 400
	X = df[features].fillna(0)
	y = df[target].fillna(0)
	if X.shape[0] < 10:
		return jsonify({"error": "Not enough rows for modeling"}), 400
	model = LinearRegression().fit(X, y)
	coef = dict(zip(features, [float(x) for x in model.coef_]))
	score = float(model.score(X, y))
	# Predict for a nominal scenario
	example = { 'Impressions': 5000, 'EngagementRate': 0.05, 'Followers': 10000 }
	vec = [[example.get(f, 0) for f in features]]
	pred = float(model.predict(vec)[0])
	return jsonify({
		"target": target,
		"features": features,
		"coefficients": coef,
		"r2": round(score, 4),
		"exampleInput": { k: example[k] for k in features },
		"predicted": round(pred, 2)
	})


@app.route('/analytics/best_times')
def analytics_best_times():
	"""Compute best posting time (day + hour) per platform and for popular users.
	Heuristic: Use highest mean ROI if available else EngagementRate. Recommend IG: 20:00, YT: 18:00."""
	platform = request.args.get('platform')  # instagram|youtube|all
	limit = int(request.args.get('limit', '10'))
	# pick dataset
	path = CUSTOM_PATH if os.path.exists(CUSTOM_PATH) else DATA_PATH
	if not os.path.exists(path):
		return jsonify({"error": "No dataset found"}), 404
	df = pd.read_csv(path)
	if 'Date' not in df.columns:
		return jsonify({"error": "Dataset missing Date column"}), 400
	# Normalize columns
	metric = None
	if 'ROI' in df.columns:
		metric = 'ROI'
	elif 'EngagementRate' in df.columns:
		metric = 'EngagementRate'
	else:
		return jsonify({"error": "Dataset missing ROI or EngagementRate"}), 400
	# prepare day-of-week aggregation
	df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
	df['DOW'] = df['Date'].dt.day_name()
	# If EngagementRate is in 0..1 scale already; if in % (unlikely after normalization), cap
	if metric == 'EngagementRate' and df['EngagementRate'].max() > 1.5:
		df['EngagementRate'] = df['EngagementRate'] / 100.0
	# compute per platform day mean
	results_by_platform = {}
	for p in ['Instagram', 'YouTube']:
		if 'Platform' in df.columns:
			pdf = df[df['Platform'].str.lower() == p.lower()]
		else:
			pdf = df.copy()
		if pdf.empty:
			continue
		agg = pdf.groupby('DOW')[metric].mean().reset_index()
		if agg.empty:
			continue
		best_row = agg.sort_values(metric, ascending=False).iloc[0]
		best_day = str(best_row['DOW'])
		best_score = float(best_row[metric])
		min_score = float(agg[metric].min())
		max_score = float(agg[metric].max())
		conf = 0.0 if max_score == min_score else (best_score - min_score) / (max_score - min_score)
		best_hour = 20 if p.lower() == 'instagram' else 18
		results_by_platform[p.lower()] = {
			"platform": p,
			"bestDay": best_day,
			"bestHour": best_hour,
			"metric": metric,
			"score": round(best_score, 4),
			"confidence": round(conf, 3)
		}
	# Build suggestions for popular users
	popular = get_popular_users(platform, limit)
	suggestions = []
	for u in popular:
		plat_key = u['platform'].lower()
		bp = results_by_platform.get(plat_key)
		if not bp:
			continue
		suggestions.append({
			"username": u['username'],
			"platform": u['platform'],
			"followers": u['followers'],
			"avgEngagement": u['avgEngagement'],
			"bestDay": bp['bestDay'],
			"bestHour": bp['bestHour'],
			"metric": bp['metric'],
			"score": bp['score'],
			"confidence": bp['confidence'],
			"reason": f"{bp['bestDay']} shows the highest mean {bp['metric']} in recent data; {bp['bestHour']}:00 aligns with evening peaks"
		})
	return jsonify({
		"byPlatform": list(results_by_platform.values()),
		"suggestions": suggestions
	})


if __name__ == '__main__':
	ensure_dataset()
	ensure_users_file()
	app.run(debug=True)
