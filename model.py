from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk

# Attempt lightweight setup; ignore if already present
try:
	nltk.download('punkt', quiet=True)
	nltk.download('stopwords', quiet=True)
except Exception:
	pass


def readability_score(text: str) -> float:
	"""Very simple readability proxy: fewer stopwords + shorter tokens = higher score."""
	if not text:
		return 0.0
	tokens = word_tokenize(text.lower())
	stops = set(stopwords.words('english'))
	meaningful = [t for t in tokens if t.isalpha() and t not in stops]
	if not tokens:
		return 0.0
	ratio = len(meaningful) / max(1, len(tokens))
	avg_len = sum(len(t) for t in meaningful) / max(1, len(meaningful))
	# Normalize into 0..100
	score = min(100.0, max(0.0, ratio * 70 + (8 - min(8, avg_len)) * 3))
	return round(score, 2)
