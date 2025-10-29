import random

POPULAR_INSTAGRAM = [
	{"username": "luxurybeans", "platform": "instagram", "followers": 1200000, "avgEngagement": 0.045, "category": "food"},
	{"username": "brew_daily", "platform": "instagram", "followers": 540000, "avgEngagement": 0.062, "category": "lifestyle"},
	{"username": "espresso_lab", "platform": "instagram", "followers": 320000, "avgEngagement": 0.071, "category": "coffee"},
	{"username": "plantbased_cups", "platform": "instagram", "followers": 210000, "avgEngagement": 0.055, "category": "health"},
]

POPULAR_YOUTUBE = [
	{"username": "DailyRoast", "platform": "youtube", "followers": 980000, "avgEngagement": 0.038, "category": "coffee"},
	{"username": "CafeVibes", "platform": "youtube", "followers": 430000, "avgEngagement": 0.044, "category": "lifestyle"},
	{"username": "BeanScience", "platform": "youtube", "followers": 260000, "avgEngagement": 0.052, "category": "education"},
	{"username": "BrewTech", "platform": "youtube", "followers": 150000, "avgEngagement": 0.047, "category": "tech"},
]


def get_popular_users(platform: str | None = None, limit: int = 10):
	candidates = []
	if platform in (None, '', 'all'):
		candidates = POPULAR_INSTAGRAM + POPULAR_YOUTUBE
	elif platform.lower() == 'instagram':
		candidates = POPULAR_INSTAGRAM
	elif platform.lower() == 'youtube':
		candidates = POPULAR_YOUTUBE
	else:
		candidates = []
	# add slight noise to engagement to simulate freshness
	result = []
	for u in candidates:
		noise = random.uniform(-0.005, 0.006)
		result.append({ **u, "avgEngagement": round(max(0.005, u["avgEngagement"] + noise), 4) })
	return sorted(result, key=lambda x: x["followers"], reverse=True)[:limit]
