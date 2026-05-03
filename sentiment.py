import yfinance as yf
import feedparser
import requests
from datetime import datetime, timedelta, timezone
from collections import defaultdict
import time
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import streamlit as st
from bs4 import BeautifulSoup
import json

# ── Config ────────────────────────────────────────────────────────────────────

MODEL_NAME      = "project-aps/finbert-finetune"
NEWS_LOOKBACK   = 7      # days
MIN_HEADLINES   = 3      # minimum for sentiment to be meaningful
RECENCY_WEIGHTS = [1.0, 0.85, 0.70, 0.55, 0.40, 0.25, 0.10]  # day 0 → day 6

def _get_article_url(url):
    try:
        resp = requests.get(url)
        data = BeautifulSoup(resp.text, 'html.parser').select_one('c-wiz[data-p]').get('data-p')
        obj = json.loads(data.replace('%.@.', '["garturlreq",'))

        payload = {
            'f.req': json.dumps([[['Fbv4je', json.dumps(obj[:-6] + obj[-2:]), 'null', 'generic']]])
        }

        headers = {
        'content-type': 'application/x-www-form-urlencoded;charset=UTF-8',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36',
        }

        url = "https://news.google.com/_/DotsSplashUi/data/batchexecute"
        response = requests.post(url, headers=headers, data=payload)
        array_string = json.loads(response.text.replace(")]}'", ""))[0][2]
        article_url = json.loads(array_string)[1]

        return article_url
    except Exception:
        return url

def _deduplicate(articles: list[dict]) -> list[dict]:
    unique = []
    for article in articles:
        is_duplicate = False
        for seen in unique:
            a_words = set(article['title'].lower().split())
            b_words = set(seen['title'].lower().split())
            overlap = len(a_words & b_words) / len(a_words | b_words)
            if overlap > 0.8:
                is_duplicate = True
                break
        if not is_duplicate:
            unique.append(article)
    return unique

def _isrelevant(headline: str, company_name: str, ticker: str) -> bool:
    headline_lower = headline.lower()

    name_words = [w.lower() for w in company_name.split() if len(w) > 3]
    ticker_clean = ticker.replace(".NS", "").lower()

    return(
        any(word in headline_lower for word in name_words) or
        ticker_clean in headline_lower
    )



# ── Load FinBERT ──────────────────────────────────────────────────────────────

@st.cache_resource
def load_finbert():
    """
    Load FinBERT pipeline once and cache.
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    label_map = {0: "neutral", 1: "negative", 2: "positive"}
    model.config.id2label = label_map
    model.config.label2id = {v: k for k, v in label_map.items()}

    pipe = pipeline("text-classification",
                    model=model,
                    tokenizer=tokenizer,
                    device=0,
                    top_k=None)
    
    return pipe

# ── News Fetching ─────────────────────────────────────────────────────────────

def fetch_yfinance_news(ticker: str) -> list[dict]:
    """
    Fetch news headlines from yfinance for a given ticker.
    """
    stock = yf.Ticker(ticker)
    
    ls = []
    for new in stock.news:
        dic = {}
        dic['provider'] = {
            'providerName' : new['content']['provider']['displayName'],
            'providerUrl' : new['content']['provider']['url']
        }
        dic['title'] = new['content']['title']
        dic['summary'] = new['content']['summary']
        
        pub_date = new['content']['pubDate']
        dt = datetime.fromisoformat(pub_date.replace("Z", "+00:00"))
        dic['publishTime'] = dt
        dic['url'] = new['content']['canonicalUrl']['url']
        
        ls.append(dic)
    
    cutoff = datetime.now(tz=timezone.utc) - timedelta(days=NEWS_LOOKBACK)
    ls = [article for article in ls if article['publishTime'] >= cutoff]
    
    if len(ls) < MIN_HEADLINES:
        return []
    
    return ls



def fetch_google_news(company_name: str) -> list[dict]:
    """
    Fetch news from Google News RSS as fallback.
    """
    query = company_name.replace(' ', '+') + '+NSE+stock'
    url = f"https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"
    feed = dict(feedparser.parse(url))

    ls = []
    for entry in feed['entries'][:20]:
        dic = {}
        link = _get_article_url(entry['link'])
        dic['provider'] = {
            'providerName' : entry['source']['title'],
            'providerUrl' : entry['source']['url']
        }
        dic['title'] = entry['title']
        dic['summary'] = entry['title']

        time_ = entry['published_parsed']
        dt = datetime.fromtimestamp(time.mktime(time_), tz=timezone.utc)
        dic['publishTime'] = dt
        dic['url'] = link
        ls.append(dic)
    
    cutoff = datetime.now(tz=timezone.utc) - timedelta(days=NEWS_LOOKBACK)
    ls = [article for article in ls if article['publishTime'] >= cutoff]

    if len(ls) < MIN_HEADLINES:
        return []
    
    return ls


def fetch_news(ticker: str, company_name: str) -> list[dict]:
    """
    Combine yfinance + Google News, deduplicate, sort by date.
    """
    yfin = fetch_yfinance_news(ticker)
    rss = fetch_google_news(company_name)
    combined = yfin + rss
    combined = [a for a in combined if _isrelevant(a['title'], company_name, ticker)]

    ordered_list = _deduplicate(combined)
    
    ordered_list.sort(key=lambda x: x['publishTime'], reverse=True)

    if len(ordered_list) < MIN_HEADLINES:
        return []
    
    return ordered_list

# ── Sentiment Scoring ─────────────────────────────────────────────────────────

def score_headline(finbert, headline: str) -> float:
    """
    Score a single headline using FinBERT.
    Returns a float in [-1, 1].
    """
    result = finbert(headline[:400])
    scores = result[0] if isinstance(result, list) else result
    # print(scores, type(scores))
    
    pos = next(s['score'] for s in scores if s['label'] == 'positive')
    neg = next(s['score'] for s in scores if s['label'] == 'negative')

    return pos - neg

def aggregate_sentiment(finbert, news: list[dict]) -> dict:
    """
    Score all headlines and aggregate into a single sentiment score.
    """
    headline_scores = []
    daily_scores = defaultdict(list)

    for item in news:
        title = item["title"]
        summary = item.get("summary", "")
        dt: datetime = item["publishTime"]
        date = dt.date()

        text = title + (". " + item["summary"] if summary != title else "")
        score = score_headline(finbert, text)

        headline_scores.append((title, score, date))
        daily_scores[date].append(score)

    daily_avg = []
    for date, scores in daily_scores.items():
        avg_score = sum(scores) / len(scores)
        daily_avg.append((date, avg_score))

    daily_avg.sort(key=lambda x: x[0], reverse=True)

    weighted_sum = 0.0
    weight_total = 0.0

    for i, (date, score) in enumerate(daily_avg):
        if i >= len(RECENCY_WEIGHTS):
            break

        w = RECENCY_WEIGHTS[i]
        weighted_sum += score * w
        weight_total += w

    final_score = weighted_sum / weight_total if weight_total > 0 else 0.0

    if final_score > 0.15:
        label = "BULLISH"
    elif final_score < -0.15:
        label = "BEARISH"
    else:
        label = "NEUTRAL"

    return {
        "score": final_score,
        "label": label,
        "headline_scores": headline_scores,
        "num_headlines": len(news),
    }


# ── Main Entry Point ──────────────────────────────────────────────────────────

def get_sentiment(ticker: str, company_name: str) -> dict | None:
    """
    Full sentiment pipeline for one stock.
    """
    model = load_finbert()
    articles = fetch_news(ticker, company_name)
    
    if len(articles) == 0:
        return None
    
    sentiment = aggregate_sentiment(model, articles)
    return sentiment

if __name__ == "__main__":
    import random
    with open("tickers.json", "r") as f:
        ticks = json.load(f)
    
    all_news = {}
    count = 0
    keys = list(ticks.keys())
    random.shuffle(keys)
    tickers = {key: ticks[key] for key in keys}

    for company_name, ticker in tickers.items():
        all_news[company_name] = get_sentiment(ticker, company_name)
        count += 10
        if count == 1:
            break

    for company, sentiment in all_news.items():
        print(f"\nCOMPANY: {company}")
        if sentiment is None:
            print("  No news found — price only fallback")
        else:
            print(f"  Score:        {sentiment['score']:.4f}")
            print(f"  Label:        {sentiment['label']}")
            print(f"  Num Headlines:{sentiment['num_headlines']}")
            print("\n  Headlines:")
            for title, score, date in sentiment['headline_scores']:
                sentiment_emoji = "🟢" if score > 0.15 else "🔴" if score < -0.15 else "⚪"
                print(f"  {sentiment_emoji} [{date}] {score:+.3f} | {title[:80]}")
        print("+" * 100)