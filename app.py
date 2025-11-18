# app.py — Hybrid AI Sentiment Analyzer (Fixed Version)
from flask import Flask, render_template, request, redirect, url_for, session, make_response, jsonify
from pymongo import MongoClient
from datetime import datetime
import asyncio, json, csv, io, requests, os, re, nltk, emoji
from transformers import pipeline
from twikit import Client

# ---------------- Flask Setup ----------------
app = Flask(__name__)
app.secret_key = '8a7f3c4d29e940aabf23d8e7bcf91c67'

# ---------------- MongoDB (With Offline Fallback) ----------------
users = None
mongo_online = False

try:
    client = MongoClient("mongodb+srv://hasnatrasool:hasnat@cluster0.xvtjszp.mongodb.net/",
                         serverSelectionTimeoutMS=5000)
    client.admin.command("ping")
    db = client["sentiment_tool"]
    users = db["users"]
    mongo_online = True
    print("[DB] ✅ MongoDB Connected")
except Exception as e:
    print("[DB] ❌ MongoDB Offline, switching to DEMO MODE")
    mongo_online = False

# ---------------- Twikit Setup ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
COOKIES_FILE = os.path.join(BASE_DIR, "x.com.cookies")
twitter_client = Client(language='en-US')

async def load_cookies():
    if not os.path.exists(COOKIES_FILE):
        print("[ERROR] x.com.cookies file missing.")
        return

    try:
        with open(COOKIES_FILE, "r", encoding="utf-8") as f:
            cookies_json = json.load(f)
        cookies_dict = {c["name"]: c["value"] for c in cookies_json if "name" in c and "value" in c}
        twitter_client.set_cookies(cookies_dict)
        print("[SUCCESS] Cookies loaded.")
    except Exception as e:
        print("[ERROR] Could not load cookies:", e)

loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)
loop.run_until_complete(load_cookies())

# ---------------- NLP Setup ----------------
nltk.download('punkt', quiet=True)

# Load transformer
try:
    ai_sentiment = pipeline("sentiment-analysis",
                            model="cardiffnlp/twitter-roberta-base-sentiment-latest")
    print("[AI] ✅ Transformer Loaded")
except Exception as e:
    print("[AI] ❌ Transformer Failed:", e)
    ai_sentiment = None

# ---------------- Preprocessing ----------------
URL_RE = re.compile(r"https?://\S+")
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#(\w+)")
MULTI_WS = re.compile(r"\s+")

# ✅ STRONG lexicon only (NO weak emotions)
NEGATIVE_LEXICON = {
    "kill", "killed", "murder", "murdered", "rape", "raped", "raping",
    "threat", "threaten", "threatened", "suicide", "torture", "tortured",
    "genocide", "molest", "molested", "slavery", "shoot", "shot",
    "weapon", "stab", "stabbed", "assault", "attacked"
}

def preprocess_text(text):
    if not text:
        return ""
    text = re.sub(URL_RE, " ", text)
    text = re.sub(MENTION_RE, " ", text)
    text = re.sub(HASHTAG_RE, r"\1", text)
    text = emoji.demojize(text)
    return re.sub(MULTI_WS, " ", text.strip())

def lexicon_negative_score(text):
    text = text.lower()
    return [w for w in NEGATIVE_LEXICON if re.search(rf"\b{w}\b", text)]

# ---------------- Sentiment Analysis ----------------
def analyze_text_deep(text):
    text = preprocess_text(text)
    if not text:
        return {'label': 'NEUTRAL', 'score': 0.0, 'by_lexicon': [], 'sentences': []}

    sentences = nltk.sent_tokenize(text)
    sent_results = []

    for s in sentences:
        s_trunc = s[:512]
        if ai_sentiment:
            try:
                r = ai_sentiment(s_trunc)[0]
                label = r["label"].upper().replace("LABEL_", "")
                score = float(r["score"])
            except:
                label, score = "NEUTRAL", 0.0
        else:
            label, score = "NEUTRAL", 0.0
        sent_results.append({"text": s_trunc, "label": label, "score": score})

    # Lexicon score
    lex_found = lexicon_negative_score(text)

    # Weighted decision
    pos = sum(s["score"] for s in sent_results if "POS" in s["label"])
    neg = sum(s["score"] for s in sent_results if "NEG" in s["label"])

    avg_pos = pos / len(sent_results)
    avg_neg = neg / len(sent_results)

    # ✅ If lexicon found → strong negative only
    if lex_found:
        return {"label": "NEGATIVE", "score": max(avg_neg, 0.75),
                "by_lexicon": lex_found, "sentences": sent_results}

    # Normal case: choose stronger average
    if avg_pos > avg_neg + 0.25:
        label = "POSITIVE"
        score = avg_pos
    elif avg_neg > avg_pos + 0.25:
        label = "NEGATIVE"
        score = avg_neg
    else:
        label = "NEUTRAL"
        score = max(avg_pos, avg_neg)

    return {"label": label, "score": score, "by_lexicon": lex_found, "sentences": sent_results}

# ---------------- Routes ----------------

@app.route('/')
def home():
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if not mongo_online:
        return "MongoDB offline. Signup disabled temporarily."

    if request.method == "POST":
        data = {
            "email": request.form.get("email"),
            "username": request.form.get("username"),
            "name": request.form.get("name"),
            "password": request.form.get("password")
        }
        if not all(data.values()):
            return "All fields required."

        if users.find_one({"email": data["email"]}):
            return "Email already exists."

        users.insert_one({**data, "created_at": datetime.utcnow()})
        return redirect(url_for("login"))

    return render_template("signup.html")

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == "POST":
        email, password = request.form.get("email"), request.form.get("password")

        if mongo_online:
            user = users.find_one({"email": email, "password": password})
            if user:
                session["user"] = user["username"]
                return redirect(url_for("dashboard"))
            return "Invalid credentials"
        else:
            # ✅ offline fallback
            session["user"] = "DemoUser"
            return redirect(url_for("dashboard"))

    return render_template("login.html")

@app.route('/dashboard')
def dashboard():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("dashboard.html", username=session["user"])

# Manual Analyzer
@app.route('/analyze', methods=['GET', 'POST'])
def analyze():
    if "user" not in session:
        return redirect(url_for("login"))

    text = request.form.get("text", "")
    result = None
    details = None

    if request.method == "POST" and text.strip():
        details = analyze_text_deep(text)
        result = f"{details['label']} ({details['score']:.2f})"

    return render_template('analyze.html',
                           username=session["user"],
                           text=text, result=result, details=details)
@app.route('/download_result', methods=['POST'])
def download_result():
    text = request.form.get('text', '')
    result = request.form.get('result', '')

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['Text', 'Sentiment'])
    writer.writerow([text, result])

    response = make_response(output.getvalue())
    response.headers["Content-Disposition"] = "attachment; filename=sentiment_result.csv"
    response.headers["Content-type"] = "text/csv"
    return response


# Twitter Analyzer
@app.route('/twitter', methods=['GET', 'POST'])
def twitter_analysis():
    if "user" not in session:
        return redirect(url_for("login"))

    tweets_data, keyword, error = [], None, None

    if request.method == "POST":
        keyword = request.form.get("keyword", "").strip()
        if not keyword:
            error = "Enter a keyword."
        else:
            try:
                async def fetch():
                    results = await twitter_client.search_tweet(keyword, "Latest")
                    for t in results[:10]:
                        text = getattr(t, "full_text", t.text)
                        res = analyze_text_deep(text)
                        tweets_data.append({
                            "tweet": text,
                            "sentiment": res["label"],
                            "score": res["score"]
                        })
                loop.run_until_complete(fetch())

            except Exception:
                error = "Failed to fetch live tweets. Showing sample."
                samples = [
                    f"{keyword} is amazing today!",
                    f"I hate how {keyword} turned out.",
                    f"People are neutral about {keyword}.",
                    f"This {keyword} is disappointing."
                ]
                for s in samples:
                    r = analyze_text_deep(s)
                    tweets_data.append({"tweet": s, "sentiment": r["label"], "score": r["score"]})

    return render_template("twitter.html",
                           username=session["user"],
                           tweets=tweets_data, keyword=keyword, error_message=error)
@app.route('/download_twitter_csv', methods=['POST'])
def download_twitter_csv():
    tweets = request.form.getlist('tweets[]')
    sentiments = request.form.getlist('sentiments[]')

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['Tweet', 'Sentiment'])

    for t, s in zip(tweets, sentiments):
        writer.writerow([t, s])

    response = make_response(output.getvalue())
    response.headers["Content-Disposition"] = "attachment; filename=twitter_sentiment.csv"
    response.headers["Content-type"] = "text/csv"
    return response

# Logout
@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for("login"))

@app.route('/about')
def about():
    return render_template("aboutus.html")

@app.route('/privacy')
def privacy():
    return render_template("privacypolicy.html")

@app.route('/terms')
def terms():
    return render_template("termsofservices.html")

if __name__ == '__main__':
    app.run(debug=True)
