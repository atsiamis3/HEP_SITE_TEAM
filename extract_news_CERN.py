import feedparser
import json

RSS_URL = "https://home.cern/api/news/news/feed.rss"
feed = feedparser.parse(RSS_URL)

posts = [
    {
        "title": e.title,
        "link": e.link,
        "published": e.published
    }
    for e in feed.entries[:5]
]

with open("cern_news.json", "w", encoding="utf-8") as f:
    json.dump(posts, f, ensure_ascii=False, indent=2)
