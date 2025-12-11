import os
import tweepy
import feedparser
import re
import openai
import json
import time
import random
from datetime import datetime, timedelta

# =========================================================
#              ENV + CONSTANTS + BOOT GUARDS
# =========================================================

# Load API keys from GitHub Secrets
TWITTER_API_KEY = os.getenv("TWITTER_API_KEY_MAIN")
TWITTER_SECRET = os.getenv("TWITTER_SECRET_MAIN")
TWITTER_ACCESS_TOKEN = os.getenv("TWITTER_ACCESS_TOKEN_MAIN")
TWITTER_ACCESS_SECRET = os.getenv("TWITTER_ACCESS_SECRET_MAIN")
TWITTER_BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN_MAIN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
XAI_API_KEY = os.getenv("XAI_API_KEY")

# --- Updated model definitions (October 2025) ---
OPENAI_MODEL = "gpt-5"                     # Replaces GPT-4
XAI_MODEL = "grok-4-fast-reasoning"        # Replaces Grok-2-1212

# =========================================================
#                        TWITTER
# =========================================================

# Authenticate Twitter API (Using API v2)
bearer_client = tweepy.Client(
    bearer_token=TWITTER_BEARER_TOKEN
)  # For reads (OAuth 2.0 app-only)

# Authenticate Twitter API (Using API v2)
twitter_client = tweepy.Client(
    consumer_key=TWITTER_API_KEY,
    consumer_secret=TWITTER_SECRET,
    access_token=TWITTER_ACCESS_TOKEN,
    access_token_secret=TWITTER_ACCESS_SECRET
)   # For writes (OAuth 1.0a user context)

# **Accounts to Follow** You can look up IDs with the X API or tools like tweeterid.com. https://twiteridfinder.com/
TARGET_ACCOUNTS = {
    "PaulG": "183749519",        # Paulg688@hotmail.com
    "pmarca": "5943622",   # Pmarca77@gmail.com
    "VitalikButerin": "295218901",
    "a16z": "64844802", # AZ509@outlook.com
    "naval": "745273" # Naval999@outlook.com
}

# How many recent tweets we READ per reply run
REPLY_FETCH_LIMIT = 5  # 5 minimum enforced by X

# =========================================================
#                     STORAGE + LIMITS
# =========================================================

# Log file to track posted and filtered news
LOG_FILE = "parcely_news_main.json"
REPLY_LOG_FILE = "replied_parcely_tweets_main.json"
TARGET_TWEETS_LOG = "parcely_target_engagement_tweets_main.json"

MENTIONS_REPLY_LOG = "parcely_mentions_reply_log_main.json"
MENTIONS_RATE_LIMIT_FILE = "parcely_last_mentions_check_main.txt" 

RETENTION_DAYS = 10  # Remove news older than 10 days

NEWS_MIN_SCORE = 9 # Define score threshold for tweets
REPLY_MIN_SCORE = 2     # Minimum score to reply to a target account tweet
QUOTE_MIN_SCORE = 5     # 6–10 → Quote with AI comment
REPOST_MIN_SCORE = 4    # 4–10 → Native repost
LIKE_MIN_SCORE = 3 

# Random tweets probabilities (Parcely)
RANDOM_NEWS          = 0.1
RANDOM_STATISTIC     = 0.15
RANDOM_INFRASTRUCTURE= 0.15
RANDOM_REPLY         = 0.2
RANDOM_ENGAGEMENT    = 0.2
RANDOM_NONE          = 0.2

# Random engagement probabilities
ENGAGEMENT_QUOTE_WEIGHT    = 0.5   # ← 85% chance to QUOTE (with AI comment)
ENGAGEMENT_REPOST_WEIGHT  = 0.5   # ← 15% chance to native REPOST
ENGAGEMENT_LIKE_WEIGHT    = 0

# Daily tweet limits
NEWS_TWEETS_LIMIT = 2  # Max news tweets per day
STAT_TWEETS_LIMIT = 1  # Max statistical tweets per day
INFRA_TWEETS_LIMIT= 1
REPLY_TWEETS_LIMIT = 0
MENTIONS_REPLY_DAILY_LIMIT = 6

# Daily limits for retweets/quotes/likes
DAILY_QUOTE_LIMIT        = 2
DAILY_REPOST_LIMIT       = 1
DAILY_LIKE_LIMIT         = 0   # Very safe (off by default)

# =========================================================
#                         RSS
# =========================================================

# Google News + Industry-Specific RSS Feeds
RSS_FEEDS = [
    # Google News – global logistics + e-commerce + smart buildings/cities
    "https://news.google.com/rss/search?q=last+mile+delivery+logistics&hl=en&gl=US&ceid=US:en",
    "https://news.google.com/rss/search?q=e-commerce+delivery&hl=en&gl=US&ceid=US:en",
    "https://news.google.com/rss/search?q=parcel+locker+network&hl=en&gl=US&ceid=US:en",
    "https://news.google.com/rss/search?q=smart+building+package+delivery&hl=en&gl=US&ceid=US:en",
    "https://news.google.com/rss/search?q=smart+city+urban+logistics&hl=en&gl=US&ceid=US:en",
    "https://news.google.com/rss/search?q=autonomous+delivery+robot&hl=en&gl=US&ceid=US:en",

    # Industry / niche logistics & parcel feeds (examples)
    "https://www.postandparcel.info/feed/",
    "https://www.parcelandpostaltechnologyinternational.com/feed",
    "https://www.supplychaindigital.com/rss",
    "https://www.theloadstar.com/feed/"
]


# =========================================================
#                        HELPERS
# =========================================================

# Define common words to ignore (stopwords)
STOPWORDS = set([
    "the", "and", "is", "in", "on", "at", "to", "of", "for", "with", "a", "an",
    "this", "that", "from", "by", "as", "it", "its", "was", "were", "are", "be", "new", "latest"
])

# Function to extract important words & numbers
def extract_key_terms(text):
    text = text.lower()
    words = re.findall(r'\b\w+\b', text)  # Extract words
    numbers = re.findall(r'\d+', text)  # Extract numbers
    keywords = [word for word in words if word not in STOPWORDS] + numbers
    return set(keywords)

# Check if a new article is similar to previously processed high-scoring articles."""
def is_similar_news(new_title, new_summary, processed_articles, threshold=0.6, limit=30):    
    new_keywords = extract_key_terms(new_title) | extract_key_terms(new_summary)

    # Debugging: Identify any invalid scores in JSON
    for article in processed_articles:
        if not isinstance(article.get("score", 0), (int, float)):
            print(f"⚠️ Warning: Invalid score detected in article - {article}")

    # ✅ Fix: Ensure scores are valid numbers before filtering
    recent_articles = [article for article in processed_articles 
                       if isinstance(article.get("score", 0), (int, float)) 
                       and article.get("score", 0) >= NEWS_MIN_SCORE][-limit:]

    for article in recent_articles:
        old_keywords = extract_key_terms(article.get("tweet", "")) | extract_key_terms(article.get("title", "")) | extract_key_terms(article.get("summary", ""))
        
        if old_keywords:
            similarity = len(new_keywords & old_keywords) / len(new_keywords | old_keywords)
            if similarity >= threshold:
                print(f"⚠️ Skipping similar news: {new_title} (Similarity: {similarity:.2f})")
                return True  # Found a similar article

    return False  # No duplicates found

# Load previously processed articles (both tweeted & filtered)
def load_processed_articles():
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, "r") as file:
                processed_data = json.load(file)
                print(f"Loaded {len(processed_data)} previously processed articles.")

                # ✅ Fix: Ensure all entries have 'link' key before returning
                valid_articles = [article for article in processed_data if isinstance(article, dict) and "link" in article]

                if len(valid_articles) < len(processed_data):
                    print(f"⚠️ Warning: {len(processed_data) - len(valid_articles)} malformed entries found and ignored.")

                return valid_articles

        except json.JSONDecodeError:
            print("⚠️ Corrupted JSON file. Resetting to an empty list.")
            return []
    return []

# Remove articles older than RETENTION_DAYS to prevent JSON file growth.
def cleanup_old_articles(processed_articles):
    cutoff_date = datetime.utcnow() - timedelta(days=RETENTION_DAYS)
    return [article for article in processed_articles if datetime.strptime(article["date"], "%Y-%m-%d") >= cutoff_date]

# Save processed articles (ensuring correct update & GitHub push)
def save_processed_articles(processed):
    print("💾 Writing to {LOG_FILE}...")
    try:
        # Always write the entire list, including both new and previously processed entries
        with open(LOG_FILE, "w") as file:
            json.dump(processed, file, indent=4)
        print("✅ Successfully wrote to {LOG_FILE}!")
    except Exception as e:
        print(f"❌ Error writing to JSON {LOG_FILE}: {e}")
        return  # Stop execution if writing fails

# HELPER FOR QUOTE REPOST / REPOST / LIKES
def load_target_tweets():
    if os.path.exists(TARGET_TWEETS_LOG):
        try:
            with open(TARGET_TWEETS_LOG, "r") as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_target_tweets(data):
    with open(TARGET_TWEETS_LOG, "w") as f:
        json.dump(data, f, indent=4)

def cleanup_target_tweets():
    data = load_target_tweets()
    cutoff = (datetime.utcnow() - timedelta(days=RETENTION_DAYS)).strftime("%Y-%m-%d")
    cleaned = {tid: entry for tid, entry in data.items() if entry.get("date", "0000-00-00") >= cutoff}
    save_target_tweets(cleaned)
    return cleaned

# HELPER FOR REPLY MENTION
def load_mentions_reply_log():
    if os.path.exists(MENTIONS_REPLY_LOG):
        try:
            with open(MENTIONS_REPLY_LOG, "r") as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_mentions_reply_log(data):
    try:
        with open(MENTIONS_REPLY_LOG, "w") as f:
            json.dump(data, f, indent=4)
        print(f"Saved {MENTIONS_REPLY_LOG}")
    except Exception as e:
        print(f"Failed to save {MENTIONS_REPLY_LOG}: {e}")

# Check if we can fetch mentions (Free tier: 1 request every 15 minutes)
def can_check_mentions():
    if not os.path.exists(MENTIONS_RATE_LIMIT_FILE):
        print("DEBUG: No rate limit file exists, allowing check.")
        return True
    try:
        last_check = float(open(MENTIONS_RATE_LIMIT_FILE).read().strip())
        time_since = time.time() - last_check
        print(f"DEBUG: Last check {time_since:.0f} seconds ago.")
        return time_since >= 960  # Increased to 16 min for safety
    except Exception as e:
        print(f"DEBUG: Error reading rate limit file: {e}. Allowing check.")
        return True

def update_mentions_timestamp():
    try:
        with open(MENTIONS_RATE_LIMIT_FILE, "w") as f:
            f.write(str(time.time()))
    except Exception as e:
        print(f"Failed to update mentions timestamp: {e}")
        

# Consolidated randomness function for post type
def select_tweet_type():
    return random.choices(
        ["news", "statistical", "infrastructure", "reply", "engagement", "none"],
        [RANDOM_NEWS, RANDOM_STATISTIC, RANDOM_INFRASTRUCTURE, RANDOM_REPLY, RANDOM_ENGAGEMENT, RANDOM_NONE]
    )[0]
    
# Count how many news tweets were posted today.
def count_news_tweets_today(processed_articles):
    today = datetime.utcnow().strftime("%Y-%m-%d")
    return sum(1 for article in processed_articles if article.get("date") == today and article.get("type") == "news")

# Count how many statistical tweets were posted today.
def count_stat_tweets_today(processed_articles):
    today = datetime.utcnow().strftime("%Y-%m-%d")
    return sum(1 for article in processed_articles if article.get("date") == today and article.get("type") == "statistical")

# Count how many infrastructural tweets were posted today.
def count_infra_tweets_today(processed_articles):
    today = datetime.utcnow().strftime("%Y-%m-%d")
    return sum(1 for article in processed_articles if article.get("date") == today and article.get("type") == "infrastructure")

# Count how many engagement actions of a given type happened today
def count_engagement_action(data, action):
    today = datetime.utcnow().strftime("%Y-%m-%d")
    return sum(1 for entry in data.values() if entry.get("date") == today and entry.get("action") == action)

# Count how many real @-mention replies we made today
def count_mentions_replies_today(log):
    today = datetime.utcnow().strftime("%Y-%m-%d")
    return sum(1 for entry in log.values() if entry.get("date") == today)



# =========================================================
#                   NEWS FETCH + SCORING
# =========================================================

# Get latest news (only from the last hour) with error handling
def get_latest_news():
    news_list = []
    now = datetime.utcnow()

    for feed_url in RSS_FEEDS:
        try:
            print(f"🔄 Fetching news from: {feed_url}")
            feed = feedparser.parse(feed_url)

            if not feed.entries:
                print(f"⚠️ Warning: No news found in {feed_url}. It may be down.")
                continue

            for entry in feed.entries:
                title = entry.title
                link = entry.link
                published_time = datetime(*entry.published_parsed[:6]) if "published_parsed" in entry else now
                source = entry.source.title if hasattr(entry, 'source') else "Unknown Source"
                summary = entry.summary if hasattr(entry, 'summary') and entry.summary else ""

                if now - published_time < timedelta(hours=1):
                    news_list.append((title, link, source, summary))
        except Exception as e:
            print(f"❌ Error fetching feed {feed_url}: {e}")
            continue
    return news_list

# =========================================================
#               AI: SCORING + SUMMARIZATION
# =========================================================

# Use GROK-2-1212 to check if news is relevant
def get_news_relevance_score(title, summary):
    client = openai.OpenAI(
        api_key=XAI_API_KEY,  # Use xAI API key
        base_url="https://api.x.ai/v1"  # xAI endpoint
    )

    prompt = f"""
    You are ranking news articles for a Twitter feed about **global last-mile delivery, e-commerce logistics, parcel lockers, and smart buildings/cities**.
    
    Assign a **relevance score (0–10)** based on:
    - Impact on last-mile delivery, parcel handling, or building/city logistics.
    - Presence of concrete data (numbers, locations, contracts, pilots, policy, investment).
    - Connection to themes like: failed deliveries, package theft, traffic congestion, packaging waste, EV/micro-mobility fleets, robotics and automation.
    
    **Scoring:**
    - **9–10:** Major global/regional deals, pilots, policy or infrastructure that directly affects urban logistics, parcel lockers, automated delivery, EV/micro fleets or smart buildings/cities.
    - **7–8:** Strong but smaller news in these areas; city-level projects, serious pilots, notable technology launches.
    - **5–6:** Indirectly relevant (e-commerce, retail, real-estate, smart cities) with at least one clear logistics or building-operations angle.
    - **1–4:** Weak, vague, or mostly unrelated to logistics or smart buildings/cities.
    - **0:** Entertainment, politics, or finance pieces without any link to delivery, logistics, real estate, or smart cities.
    
    Subtract 1 point if the article lacks concrete data (no numbers, locations, or named projects).
    
    Reply with a **single integer 0–10** only.
    
    Article:
    Title: {title}
    Summary: {summary}
    """

    response = client.chat.completions.create(
        model=XAI_MODEL,  # Changed to Grok-2-1212
        messages=[{"role": "user", "content": prompt}]
    )

    score_text = response.choices[0].message.content.strip()
    try:
        score = int(score_text)
        return score if 0 <= score <= 10 else 0
    except ValueError:
        return 0
    except openai.OpenAIError as e:
        print(f"❌ xAI API Error: {e}")
        return 0

# Summarize news and format tweet
def summarize_news(title, summary, source):
    client = openai.OpenAI(
        api_key=XAI_API_KEY,  # Use xAI API key
        base_url="https://api.x.ai/v1"  # xAI endpoint
    )

    prompt = f"""
    Rewrite this logistics-related news title into a concise, natural tweet.

    - **Always place a country flag emoji at the START** if a country, city, or company is explicitly mentioned.
    - **Format:** (Flag) NEWS: Main content
    - **DO NOT use** quotes, hashtags, sources, or websites.
    - **DO NOT use any emojis except country flags.**
    
    Title: {title}
    """

    if summary:
        prompt += f"\n\nAlso, integrate one key point from this summary: {summary}"

    response = client.chat.completions.create(
        model=XAI_MODEL,  # Changed to Grok-2-1212
        messages=[{"role": "user", "content": prompt}]
    )

    ai_summary = response.choices[0].message.content.strip()
    ai_summary = ai_summary.replace('"', '').replace("'", "")
    tweet = f"{ai_summary}"
    return tweet[:280]


# =========================================================
#      AI: STATISTICAL TWEET GENERATORS
# =========================================================

# Generate statistical post #
# Global list of statistical tweet categories in a preferred order

STATISTICAL_CATEGORIES = [
    "failed first-attempt deliveries in global e-commerce",
    "package theft and porch piracy in major cities",
    "last-mile delivery costs per parcel in large metros",
    "urban traffic growth caused by delivery vans in dense cities",
    "packaging waste from e-commerce deliveries worldwide",
    "growth of smart residential and office buildings",
    "adoption of smart access systems for buildings",
    "EV and micro-mobility delivery fleets in cities",
    "parcel locker and pickup-point networks globally",
    "online grocery and food delivery frequency in large cities",
    "CO2 emissions from last-mile logistics versus building-integrated networks",
]


def generate_statistical_tweet(selected_category):
    """Generate a statistical tweet dynamically using GPT-4."""
    tweet_formats = {
        1: "A single striking statistic or future projection",
        2: "A direct comparison between two statistical facts",
        3: """Generate a ranked list of the top 5 or, if space permits, top 10, ensuring the tweet is under 280 characters.
    
Format:
<Very short overview of the metric description>

1. City/Country
2. City/Country
3. City/Country
4. City/Country
5. City/Country
"""
}
    
    selected_format_key = random.choice(list(tweet_formats.keys()))
    selected_format = tweet_formats[selected_format_key]
    
    prompt = f"""
    Assume the current year is 2025. Generate a concise, direct, factual, and impactful statistical tweet about {selected_category} that uses current data or realistic projections for 2025 and beyond. Avoid using outdated statistics from before 2023.

    {selected_format}

    The tweet should:
    - Present only clear, factual data
    - Use everyday language over jargon
    - **NEVER use quotes, hashtags, or generic emojis.**
    - **Keep it strictly under 280 characters.**
    - **NEVER use generic phrases and unnecessary filler words.** Keep it sharp and data-driven.
    - **Always place country flags before a location name.**
    - **Use proper line breaks for readability.** If the tweet contains multiple paragraphs, insert a blank line between them.
    """

    client = openai.OpenAI(
        api_key=XAI_API_KEY,
        base_url="https://api.x.ai/v1"
    )
    response = client.chat.completions.create(
        model=XAI_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )

    tweet = response.choices[0].message.content.strip()
    return tweet[:280]  # safety: hard cap at 280 chars

# =========================================================
#      AI: INFRASTRUCTURE TWEET GENERATORS
# =========================================================

# Generate an infrastructure tweet using your provided prompt.
def generate_infrastructure_tweet():
    client = openai.OpenAI(
        api_key=XAI_API_KEY,  # Use xAI API key
        base_url="https://api.x.ai/v1"  # xAI endpoint
    )
    
    prompt = """
    Assume the current year is 2025. Write a concise social media post from an external perspective about a logistics focused company that highlights a single key quantitative infrastructure metric. Focus strictly on presenting data with minimal wording.

    The tweet should:
    - Present only clear, factual data (e.g., daily data volumes, production figures, energy consumption, or efficiency ratings)
    - Use everyday language over jargon
    - **NEVER use quotes, hashtags, or generic emojis.**
    - **Keep it strictly under 280 characters.**
    - **NEVER use generic phrases and unnecessary filler words.** Keep it sharp and data-driven.
    - **Always place country flags before a location name.**
    - **Use proper line breaks for readability.** If the tweet contains multiple paragraphs, insert a blank line between them.     
    
    """

    response = client.chat.completions.create(
        model=XAI_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    tweet = response.choices[0].message.content.strip()
    return tweet

# =========================================================
#                       REPLIES
# =========================================================

def load_reply_log():
    """ Load previously replied tweets to avoid duplicates. """
    if os.path.exists(REPLY_LOG_FILE):
        with open(REPLY_LOG_FILE, "r") as file:
            return json.load(file)
    return {}

def save_reply_log(log_data):
    """ Save replied tweets to prevent duplicate replies. """
    print("💾 Writing to replied_parcely_tweets.json...")
    try:
        with open(REPLY_LOG_FILE, "w") as file:
            json.dump(log_data, file, indent=4)
        print("✅ Successfully wrote to replied_parcely_tweets.json!")
    except Exception as e:
        print(f"❌ Error writing to replied_parcely_tweets.json: {e}")

def count_replies_today(reply_log):
    today = datetime.utcnow().strftime("%Y-%m-%d")
    return sum(1 for entry in reply_log.values() if entry["date"] == today)

def fetch_latest_tweets(user_id, max_results=REPLY_FETCH_LIMIT):
    try:
        tweets = bearer_client.get_users_tweets(
            id=user_id,
            max_results=max_results,
            tweet_fields=["text", "created_at"],
            exclude=["retweets", "replies"]
        )
        if not tweets.data:
            return []

        log = load_target_tweets()
        today = datetime.utcnow().strftime("%Y-%m-%d")
        saved = 0

        for tweet in tweets.data:
            tid = str(tweet.id)
            if tid in log:
                continue  # already have it

            # Score relevance immediately
            score = classify_mention_relevance(tweet.text)
            handle = next((k for k, v in TARGET_ACCOUNTS.items() if v == user_id), "unknown")

            log[tid] = {
                "tweet_id": tid,
                "text": tweet.text,
                "author_id": user_id,
                "author_handle": handle,
                "date": today,
                "relevance_score": score,   # ← now 0–10 integer
                "action": None
            }
            saved += 1

        if saved > 0:
            save_target_tweets(log)
            print(f"Saved {saved} new tweets with relevance scores")

        return tweets.data

    except Exception as e:
        print(f"Error fetching tweets: {e}")
        return []


def generate_grok_reply(tweet_text, username):
    """ Use Grok-2-1212 to generate a smart, relevant reply based on the tweet. """
    prompt = f"""
    You are responding to @{username} on Twitter.

    - Read the following tweet and generate a **concise, data-driven reply** that adds a relevant statistic or fact. 
    - Ensure the response is **engaging, contextually relevant, and under 280 characters.**
    - The reply should **enhance the conversation** by providing a valuable insight related to the tweet's topic.
    - **Maintain a professional yet conversational tone.**
    - **DO NOT mention or @ the username to keep it natural.**
    - **DO NOT use hashtags, emojis, or generic phrases.**
    - If no suitable statistic is available, provide a **thoughtful industry insight, preferably related to one of @{username}'s companies.**

    **Tweet:** "{tweet_text}"

    **Your Reply:**
    """

    client = openai.OpenAI(
        api_key=XAI_API_KEY,  # Using xAI API key
        base_url="https://api.x.ai/v1"  # xAI endpoint
    )

    response = client.chat.completions.create(
        model=XAI_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content.strip()

def reply_to_random_tweet():
    """ Randomly select a user, fetch their latest tweet, and reply once per tweet. """
    reply_log = load_reply_log()

    # Daily limit check (using already loaded log)
    if count_replies_today(reply_log) >= REPLY_TWEETS_LIMIT:
        print(f"🚫 Reached daily reply limit ({REPLY_TWEETS_LIMIT}). Exiting.")
        return

    if not TARGET_ACCOUNTS:
        print("⚠️ No TARGET_ACCOUNTS configured. Skipping reply run.")
        return

    # **Step 1: Randomly choose a user**
    user_to_fetch = random.choice(list(TARGET_ACCOUNTS.keys()))
    user_id = TARGET_ACCOUNTS[user_to_fetch]
    print(f"🔍 Fetching tweets from @{user_to_fetch}...")

    # **Step 2: Fetch their latest tweets**
    all_tweets = fetch_latest_tweets(user_id)  # uses REPLY_FETCH_LIMIT

    if not all_tweets:
        print(f"🔍 No tweets found for @{user_to_fetch}.")
        return

    # Build set of tweet IDs we've already replied to
    replied_ids = set(reply_log.keys())

    # **Step 3: Filter out tweets we've already replied to**
    new_tweets = [t for t in all_tweets if str(t.id) not in replied_ids]

    if not new_tweets:
        print(f"🔁 All recent tweets from @{user_to_fetch} already replied to. Skipping this run.")
        return

    # # SMART FILTER: with a score over defined and pick the most recent new tweet
    selected_tweet = new_tweets[0]
    tweet_id = selected_tweet.id                  # ← fixed
    tweet_text = selected_tweet.text              # ← fixed

    target_data = load_target_tweets()
    score = target_data.get(str(tweet_id), {}).get("relevance_score", 0)

    if score <= REPLY_MIN_SCORE:
        print(f"Skipping reply → low relevance score {score}/10: \"{selected_tweet.text[:80]}...\"")
        return
        
    username = user_to_fetch  # Using stored username

    # **Step 4: Generate a Grok-powered reply**
    reply_text = generate_grok_reply(tweet_text, username)
    if not reply_text:
        print(f"❌ Failed to generate reply for @{username}. Skipping.")
        return

    # **Step 5: Post the reply**
    try:
        twitter_client.create_tweet(
            text=reply_text,  # No @{username} prefix to keep it natural
            in_reply_to_tweet_id=tweet_id
        )
        print(f"✅ Replied to @{username}: {reply_text}")

        # **Step 6: Log replied tweet (now with full texts)**
        reply_log[str(tweet_id)] = {
            "date": datetime.utcnow().strftime("%Y-%m-%d"),
            "username": username,
            "tweet_id": tweet_id,
            "source_text": tweet_text,
            "reply_text": reply_text,
            "relevance_score": score
        }
        save_reply_log(reply_log)

    except tweepy.errors.TweepyException as e:
        print(f"❌ Error posting reply: {e}")

# =========================================================
#             TARGET ENGAGEMENT (Quote/RT/Like)
# =========================================================

def classify_mention_relevance(text):
    client = openai.OpenAI(api_key=XAI_API_KEY, base_url="https://api.x.ai/v1")
    
    prompt = f"""
    Score this tweet 0–10 for Parcely — an account obsessed with how last-mile logistics, parcel lockers,
    automated building networks, and smart urban delivery reshape cities between 2025–2050.

    10 = Concrete last-mile / parcel / smart-building logistics being built (large-scale pilots, networks, regulations)
    9  = Major deals, policies, or deployments in urban logistics, delivery robots, lockers, EV fleets
    8  = Strong tech-logistics content (e-commerce infra, delivery optimization, urban access systems)
    7  = Solid discussion of logistics bottlenecks (failed deliveries, congestion, costs, waste)
    6  = Data centers / infra / real estate that clearly touches delivery networks or building operations
    5  = General tech, real estate, or urban planning with a plausible logistics angle
    4  = Adjacent founder/VC/tech chatter that could be reframed as logistics-relevant
    3  = Light memes/hot takes from target accounts – still potentially reply-worthy
    2  = Off-topic but not trash
    0–1 = gm/gn, one-word, pure spam

    Tweet: \"{text}\"
    Answer with only the number 0–10.
    """

    try:
        resp = client.chat.completions.create(
            model=XAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=5,
            temperature=0.1
        )
        score_text = resp.choices[0].message.content.strip()
        score = int(score_text)
        return max(0, min(10, score))  # Clamp to 0–10
    except:
        return 0

def generate_quote_comment(text):
    client = openai.OpenAI(api_key=XAI_API_KEY, base_url="https://api.x.ai/v1")
    
    prompt = f"""
    Write a sharp, professional quote tweet (max 180 chars) from Parcely
    that adds a precise insight or data point to the original tweet,
    ideally about last-mile logistics, parcel lockers, building access, or e-commerce delivery.

    Rules:
    - Generate a **concise, data-driven insight** that adds a relevant statistic or fact.
    - No hashtags, no @-mentions, no generic emojis (country flags OK).
    - Sound forward-looking and authoritative.
    - Never generic — always add a concrete angle or number when possible.

    Original tweet: {text}

    Quote comment only:
    """

    try:
        resp = client.chat.completions.create(
            model=XAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=110,
            temperature=0.8
        )
        comment = resp.choices[0].message.content.strip()
        return comment[:180] if comment else None
    except:
        return None

def process_mention_engagement():
    data = cleanup_target_tweets()  # Auto-remove old tweets
    if not data:
        print("No target tweets in pool")
        return

    # Filter tweets that haven't been engaged with yet
    available = [(tid, entry) for tid, entry in data.items() if entry.get("action") is None]
    if not available:
        print("All target tweets already engaged with")
        return

    # Random-within-random: pick action type
    action = random.choices(
        ["quote", "repost", "like"],
        weights=[ENGAGEMENT_QUOTE_WEIGHT, ENGAGEMENT_REPOST_WEIGHT, ENGAGEMENT_LIKE_WEIGHT],
        k=1
    )[0]

    print(f"Engagement mode: {action.upper()} → curating from target accounts")

    processed = 0
    random.shuffle(available)

    for tid, entry in available:
        text = entry["text"]
        score = entry.get("relevance_score", 0)  # Use pre-scored value

        # QUOTE:
        if (action == "quote" and score >= QUOTE_MIN_SCORE and 
            count_engagement_action(data, "quote") < DAILY_QUOTE_LIMIT):
            comment = generate_quote_comment(text)
            if comment and 15 < len(comment) < 200:
                try:
                    twitter_client.create_tweet(text=comment, quote_tweet_id=int(tid))
                    
                    # ←←← NOW SAVES THE ACTUAL QUOTE TEXT!
                    data[tid]["action"] = "quote"
                    data[tid]["date"] = datetime.utcnow().strftime("%Y-%m-%d")
                    data[tid]["quote_text"] = comment.strip()   # ← THIS IS THE FIX!
                    
                    print(f"Quote-tweeted: {comment[:60]}...")
                    processed += 1
                    save_target_tweets(data)
                    return  # One quote per run is enough
                except Exception as e:
                    print(f"Quote failed: {e}")

        # REPOST: 7–10
        elif (action == "repost" and score >= REPOST_MIN_SCORE and 
              count_engagement_action(data, "repost") < DAILY_REPOST_LIMIT):
            try:
                twitter_client.retweet(tweet_id=int(tid))  # ← This always works
                data[tid]["action"] = "repost"
                data[tid]["date"] = datetime.utcnow().strftime("%Y-%m-%d")
                print("Reposted from target account")
                processed += 1
                if processed >= 2:
                    save_target_tweets(data)
                    return
            except Exception as e:
                print(f"Repost failed: {e}")

        # LIKE: 5–10 (or everything if like mode)
        elif (action == "like" and score >= LIKE_MIN_SCORE and 
              count_engagement_action(data, "like") < DAILY_LIKE_LIMIT):
            try:
                twitter_client.like(tweet_id=int(tid))
                data[tid]["action"] = "like"
                data[tid]["date"] = datetime.utcnow().strftime("%Y-%m-%d")
                processed += 1
            except Exception as e:
                print(f"Like failed: {e}")

        if processed >= 3:
            break

    save_target_tweets(data)
    print(f"Engagement complete: {processed} actions")

# =========================================================
#         REAL @-MENTION → ALWAYS REPLY (Separate & Guaranteed)
# =========================================================

MY_USER_ID = None

def get_my_user_id():
    global MY_USER_ID
    if MY_USER_ID:
        return MY_USER_ID
    try:
        MY_USER_ID = twitter_client.get_me().data.id
        print(f"My user ID: {MY_USER_ID}")
        return MY_USER_ID
    except:
        return None

def process_mention_replies():
    if not can_check_mentions():
        print("Mentions check skipped (15-min rate limit)")
        return

    user_id = get_my_user_id()
    if not user_id:
        return

    log = load_mentions_reply_log()
    since_id = log.get('metadata', {}).get('last_mention_id')

    update_mentions_timestamp()  # Commit before fetch

    try:
        resp = bearer_client.get_users_mentions(
            id=user_id,
            max_results=10,
            tweet_fields=["author_id", "text"],
            since_id=since_id
        )
        print(f"Fetched {len(resp.data or [])} mentions")
    except tweepy.errors.TooManyRequests as e:
        print(f"Rate limit hit (429): {e}. Waiting longer next time.")
        return
    except Exception as e:
        print(f"Failed to fetch mentions: {e}")
        return

    mentions = resp.data or []
    today = datetime.utcnow().strftime("%Y-%m-%d")

    # Update metadata with max ID from this fetch (even if no replies)
    if mentions:
        new_max = max(int(tweet.id) for tweet in mentions)
        current_max = since_id or 0
        updated_max = max(current_max, new_max)
        if 'metadata' not in log:
            log['metadata'] = {}
        log['metadata']['last_mention_id'] = updated_max
        save_mentions_reply_log(log)  # Save updated max early

    if count_mentions_replies_today(log) >= MENTIONS_REPLY_DAILY_LIMIT:  # ← now uses the correct one
        print(f"Daily mention reply limit reached ({MENTIONS_REPLY_DAILY_LIMIT})")
        return

    replied = 0
    for tweet in mentions:
        tid = str(tweet.id)
        if tid in log or tweet.author_id == user_id:
            continue

        # Blocks: crypto spam (50+ tags) + Grok/Claude/Gemini replies (2 tags) + any mass-tag nonsense
        if len(re.findall(r'@\w+', tweet.text)) > 1:
            print(f"Blocked mention with multiple @ tags ({tweet.text[:100]}...)")
            continue

        reply_text = openai.OpenAI(api_key=XAI_API_KEY, base_url="https://api.x.ai/v1").chat.completions.create(
            model=XAI_MODEL,
            messages=[{
                "role": "user",
                "content": f"""
        You are @Parcely — a last-mile delivery and building-integrated logistics account.
        
        Someone just @-mentioned you with this:
        
        "{tweet.text}"
        
        Write a concise, natural, professional reply (max 240 chars).
        - Tie your answer to logistics, deliveries, parcel lockers, or smart buildings when possible.
        - No hashtags, no @-mentions (X adds them automatically)
        - No generic emojis (country flags OK)
        - Use everyday language over jargon.
        Reply directly with only the final reply text, nothing else:
        """
            }],
            temperature=0.7,
            max_tokens=300
        ).choices[0].message.content.strip()

        if not reply_text or len(reply_text) > 280:
            continue

        try:
            twitter_client.create_tweet(text=reply_text, in_reply_to_tweet_id=tweet.id)
            log[tid] = {"date": today, "replied": True, "text": reply_text}
            save_mentions_reply_log(log)
            print(f"Replied to @-mention: {reply_text[:60]}...")
            if count_mentions_replies_today(log) >= MENTIONS_REPLY_DAILY_LIMIT:
                break
        except Exception as e:
            print(f"Mention reply failed: {e}")

    if replied:
        print(f"Completed {replied} mention replies")


# =========================================================
#                      POSTING
# =========================================================

# Post to X (Twitter) using API v2 with a delay
def post_tweet(tweet):
    print(f"🚀 Attempting to tweet: {tweet}")  # Debugging line
    try:
        response = twitter_client.create_tweet(text=tweet)
        print(f"✅ Tweet posted successfully: {response.data}")

        # Introduce a 3-minute delay **after** posting each tweet
        print("⏳ Waiting 2 minutes before posting the next tweet...")
        time.sleep(120)  # 120 seconds 

        return True
    except tweepy.errors.Forbidden as e:
        if "Status is a duplicate" in str(e):
            print("⚠️ Duplicate Tweet detected. Skipping.")
        else:
            print(f"❌ Twitter API error: {e}")
        return False
    except tweepy.errors.TweepyException as e:
        print(f"❌ Other Tweepy error: {e}")
        return False
        
# =========================================================
#                        MAIN
# =========================================================

############## Main execution starts here ##############
if __name__ == "__main__":
    print("Agent started — checking real @-mentions first...")
    process_mention_replies()
    
    print("🔍 Loading previously processed articles...")
    processed_articles = load_processed_articles()
    filtered_links = {article["link"] for article in processed_articles if "link" in article} if processed_articles else set()
    print(f"📂 {len(processed_articles)} articles already processed.")

    today = datetime.utcnow().strftime("%Y-%m-%d")
    today_news_count = count_news_tweets_today(processed_articles)
    today_stat_count = count_stat_tweets_today(processed_articles)
    today_infra_count = count_infra_tweets_today(processed_articles)
    today_reply_count = count_replies_today(load_reply_log())

    # Consolidated random selection for type of tweet
    tweet_type = select_tweet_type()
    print(f"🔀 Selected tweet type: {tweet_type}")

    # Early exit if daily tweet limit for the selected type is reached
    if tweet_type == "news" and today_news_count >= NEWS_TWEETS_LIMIT:
        print(f"🚫 Reached daily news tweet limit ({NEWS_TWEETS_LIMIT}). Exiting to save resources.")
        exit(0)
    elif tweet_type == "statistical" and today_stat_count >= STAT_TWEETS_LIMIT:
        print(f"🚫 Reached daily statistical tweet limit ({STAT_TWEETS_LIMIT}). Exiting to save resources.")
        exit(0)
    elif tweet_type == "infrastructure" and today_infra_count >= INFRA_TWEETS_LIMIT:
        print(f"🚫 Reached daily infrastructure tweet limit ({INFRA_TWEETS_LIMIT}). Exiting to save resources.")
        exit(0)
    elif tweet_type == "reply" and today_reply_count >= REPLY_TWEETS_LIMIT:
        print(f"🚫 Reached daily reply tweet limit ({REPLY_TWEETS_LIMIT}). Exiting to save resources.")
        exit(0)

    # Tweet types
    if tweet_type == "news":
        latest_news = get_latest_news()
        print(f"📰 Found {len(latest_news)} new articles.")

        scored_news = []
        seen_links = set()  # ✅ Prevent processing duplicate links in the same workflow run

        for title, link, source, summary in latest_news:     
            if today_news_count >= NEWS_TWEETS_LIMIT:
                print(f"🚫 Stopping news tweets early: {today_news_count} tweets reached.")
                break  # 💡 STOP posting news if limit is reached      

            if link in seen_links or link in filtered_links:
                print(f"⏩ Skipping duplicate article from multiple RSS feeds: {title}")
                continue
            seen_links.add(link) 

            if link in filtered_links:
                print(f"⏩ Skipping already processed article: {title}")
                continue

            # ✅ Check for similarity first
            similar = is_similar_news(title, summary, processed_articles, threshold=0.5, limit=30)

            # ✅ If similar, store and skip further processing
            if similar:
                article_entry = {
                    "link": link,
                    "date": datetime.utcnow().strftime("%Y-%m-%d"),
                    "title": title,
                    "summary": summary,
                    "similarity_excluded": "Yes",
                    "score": 0,  # ✅ No GPT-4 scoring for similar articles
                    "status": "skipped",
                    "tweet": None
                }
                processed_articles.append(article_entry)
                continue  # 🚨 Skip scoring and tweet generation

            # ✅ If NOT similar, continue processing
            score = get_news_relevance_score(title, summary)

            # ✅ Always store the article
            article_entry = {
                "link": link,
                "date": datetime.utcnow().strftime("%Y-%m-%d"),
                "title": title,
                "summary": summary,
                "similarity_excluded": "No",
                "score": score,  # ✅ Storing score ONLY if not similar
                "status": "processed",
                "tweet": None
            }

            # ✅ Generate tweet only if score meets threshold
            if score >= NEWS_MIN_SCORE:
                article_entry["tweet"] = summarize_news(title, summary, source)

            processed_articles.append(article_entry)
            scored_news.append((score, title, link, source, summary))

        # ✅ Sort articles by highest relevance score
        scored_news.sort(reverse=True, key=lambda x: x[0])

        # ✅ Pick top 3 highest-ranked news articles
        top_articles = scored_news[:3]

        new_entries = []

        for score, title, link, source, summary in top_articles:
            if score >= NEWS_MIN_SCORE:
                tweet = summarize_news(title, summary, source)

                if post_tweet(tweet):
                    today_news_count += 1  # ✅ Update count after posting
                    new_entry = {
                        "link": link,
                        "date": datetime.utcnow().strftime("%Y-%m-%d"),
                        "title": title,
                        "summary": summary,
                        "similarity_excluded": "No",  # ✅ Now included for consistency
                        "score": score,
                        "status": "posted",
                        "tweet": tweet,
                        "type": "news"  # <-- Include this line to mark the tweet type
                    }
                    processed_articles.append(new_entry)
                    new_entries.append(new_entry)
            else:
                print("🚫 No high-scoring news found to post.")
    
    
    elif tweet_type == "statistical":
        if today_stat_count >= STAT_TWEETS_LIMIT:
            print(f"🚫 Reached daily statistical tweet limit ({STAT_TWEETS_LIMIT}). Skipping statistical tweets.")
        else:
            # Select a category from your predefined global list.
            selected_category = random.choice(STATISTICAL_CATEGORIES)
            # Generate a tweet specific to that category.
            tweet = generate_statistical_tweet(selected_category)
            if post_tweet(tweet):
                today_stat_count += 1
                processed_articles.append({
                    "link": None,  
                    "date": today,
                    "status": "posted",
                    "tweet": tweet,
                    "type": "statistical",
                    "category": selected_category
                })

    elif tweet_type == "infrastructure":
        if today_infra_count >= INFRA_TWEETS_LIMIT:
            print(f"🚫 Reached daily infrastructure tweet limit ({INFRA_TWEETS_LIMIT}). Skipping infrastructure tweets.")
        else:
            tweet = generate_infrastructure_tweet()
            if post_tweet(tweet):
                processed_articles.append({
                    "link": None,
                    "date": today,
                    "status": "posted",
                    "tweet": tweet,
                    "type": "infrastructure"
                })

    elif tweet_type == "reply":
        reply_to_random_tweet()

    elif tweet_type == "engagement":
        print("Engagement cycle — curating logistics/last-mile tweets silently")
        process_mention_engagement()

    else:
        print("🤖 No tweet posted in this run to simulate human-like activity.")

    # ✅ Save all processed articles to JSON
    processed_articles = cleanup_old_articles(processed_articles)
    save_processed_articles(processed_articles)
    print(f"✅ {LOG_FILE} updated successfully!")

