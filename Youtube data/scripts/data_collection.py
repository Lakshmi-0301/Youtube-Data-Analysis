import sqlite3
import pandas as pd
from googleapiclient.discovery import build
import time
from tqdm import tqdm
import isodate
import re
import datetime
import json
import sys

# Insert your API key here
API_KEY = "AIzaSyBuoTpWvf46ExRSpbnHpS1JqFFqMiaoBgQ"

# Initialize YouTube API client
youtube = build("youtube", "v3", developerKey=API_KEY)
DB_NAME = "youtube_data.db"

def init_db():
    """Create a database and table if they don't exist."""
    sqlite3.register_adapter(datetime.date, lambda d: d.isoformat())  # ✅ Register ISO format for dates
    conn = sqlite3.connect(DB_NAME, detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS videos (
            video_id TEXT PRIMARY KEY,
            title TEXT,
            views INTEGER,
            likes INTEGER,
            comments INTEGER,
            publish_date TEXT,
            duration INTEGER,
            category TEXT,
            tags TEXT,
            channel_name TEXT,
            subscribers INTEGER
        )
    """)
    cursor.execute("Delete from videos")
    conn.commit()
    conn.close()

def get_channel_details(channel_id):
    """Fetch channel name and subscriber count."""
    request = youtube.channels().list(part="snippet,statistics", id=channel_id)
    response = request.execute()
    
    if not response.get("items"):
        return None, None

    channel_info = response["items"][0]
    channel_name = channel_info["snippet"]["title"]
    subscribers = channel_info["statistics"].get("subscriberCount", "Unknown")

    return channel_name, subscribers

def clean_text(text):
    """Lowercase, remove special characters, and strip whitespace."""
    return re.sub(r'[^\w\s]', '', text).strip().lower() if text else "Unknown"

def clean_numeric(value):
    """Convert to integer, return 0 if conversion fails."""
    try:
        return int(value)
    except:
        return 0

def clean_duration(duration):
    """Convert YouTube ISO 8601 duration format to seconds."""
    try:
        seconds = int(isodate.parse_duration(duration).total_seconds())
        return max(seconds, 0)  # Ensure duration is not negative
    except:
        return 0

def get_video_details(video_ids):
    """Fetch and clean video data without displaying progress."""
    all_video_data = []

    for video_id in video_ids:  # ✅ No tqdm here
        request = youtube.videos().list(
            part="snippet,statistics,contentDetails",
            id=video_id
        )
        response = request.execute()

        if not response.get("items"):
            continue

        video_info = response["items"][0]
        snippet = video_info["snippet"]
        stats = video_info.get("statistics", {})
        content_details = video_info.get("contentDetails", {})

        title = clean_text(snippet.get("title", "Unknown"))
        views = clean_numeric(stats.get("viewCount", 0))
        likes = clean_numeric(stats.get("likeCount", 0))
        comments = clean_numeric(stats.get("commentCount", 0))
        publish_date = snippet.get("publishedAt", "Unknown")
        duration = clean_duration(content_details.get("duration", "PT0S"))
        category = snippet.get("categoryId", "Unknown")
        tags = ",".join([clean_text(tag) for tag in snippet.get("tags", [])])
        channel_id = snippet["channelId"]

        try:
            publish_date = datetime.datetime.strptime(publish_date, "%Y-%m-%dT%H:%M:%SZ").date()
        except:
            publish_date = "Unknown"

        video_data = {
            "video_id": video_id,
            "title": title,
            "views": views,
            "likes": likes,
            "comments": comments,
            "publish_date": publish_date,
            "duration": duration,
            "category": category,
            "tags": tags,
            "channel_id": channel_id
        }
        all_video_data.append(video_data)

        time.sleep(0.5)  # Prevent API rate limit issues

    return all_video_data


def search_videos(query, max_results=50):
    """Search for videos based on a keyword."""
    request = youtube.search().list(
        q=query,
        part="snippet",
        type="video",
        maxResults=int(max_results)
    )
    response = request.execute()
    video_ids = [item["id"]["videoId"] for item in response.get("items", [])]
    return video_ids

def store_data_in_db(video_data):
    """Store video data in the SQLite database."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    for video in video_data:
        channel_name, subscribers = get_channel_details(video["channel_id"])
        video["channel_name"] = clean_text(channel_name)
        video["subscribers"] = clean_numeric(subscribers)

        cursor.execute("""
            INSERT OR REPLACE INTO videos (video_id, title, views, likes, comments, publish_date, duration, category, tags, channel_name, subscribers)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            video["video_id"], video["title"], video["views"], video["likes"], video["comments"],
            video["publish_date"], video["duration"], video["category"], video["tags"],
            video["channel_name"], video["subscribers"]
        ))

    conn.commit()
    conn.close()

def fetch_youtube_data(query, count):
    """Fetch and store YouTube video data for a given query and count."""
    init_db()
    print(f"🔎 Searching for videos related to '{query}'...")

    video_ids = search_videos(query, max_results=count)

    if not video_ids:
        print(f"❌ No videos found for '{query}'. Try a different search query.")
        return json.dumps({"status": "fail", "message": "No videos found."})

    print("📥 Fetching video details...")
    video_data = get_video_details(video_ids)

    print("💾 Storing data in the database...")
    store_data_in_db(video_data)

    return json.dumps({"status": "success", "message": "Data collection complete."})

if __name__ == "__main__":
    # Read input from standard input (sent by `app.py`)
    input_data = sys.stdin.read()
    try:
        params = json.loads(input_data)
        search_query = params["query"]
        video_count = params["count"]
        result = fetch_youtube_data(search_query, video_count)
        print(result)  # Send JSON output back to `app.py`
    except Exception as e:
        print(json.dumps({"status": "error", "message": str(e)}), file=sys.stderr)
