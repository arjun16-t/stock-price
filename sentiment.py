import yfinance as yf
import feedparser
import requests
from datetime import datetime, timedelta, timezone
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
                    return_all_scores=True)
    
    return pipe

# ── News Fetching ─────────────────────────────────────────────────────────────

def fetch_yfinance_news(ticker: str) -> list[dict]:
    """
    Fetch news headlines from yfinance for a given ticker.

    TODO:
    1. yf.Ticker(ticker).news → list of news dicts
    2. Each dict has keys: 'title', 'publisher', 'providerPublishTime' (unix timestamp)
    3. Filter to only last NEWS_LOOKBACK days
       Hint: datetime.now() - timedelta(days=NEWS_LOOKBACK)
       Convert providerPublishTime: datetime.fromtimestamp(article['providerPublishTime'])
    4. Return list of dicts with keys: {'title', 'publisher', 'date'}
       where date is a datetime object

    Return empty list if anything fails — never crash on news fetch.
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

    TODO:
    1. Build RSS URL:
       query = company_name.replace(' ', '+') + '+NSE+stock'
       url = f"https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"
    
    2. Use feedparser.parse(url) to parse RSS feed
       Each entry has: entry.title, entry.published, entry.source.title
    
    3. Parse entry.published string to datetime:
       Hint: use email.utils.parsedate_to_datetime() — handles RSS date format cleanly
    
    4. Filter to last NEWS_LOOKBACK days
    5. Return same format as fetch_yfinance_news: {'title', 'publisher', 'date'}

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

    TODO:
    1. Call fetch_yfinance_news(ticker)
    2. Call fetch_google_news(company_name)
    3. Combine both lists
    4. Deduplicate: two headlines are duplicates if their titles share
       more than 80% word overlap
       Hint: use set intersection on lowercased words
       overlap = len(set(a.split()) & set(b.split())) / len(set(a.split()) | set(b.split()))
    5. Sort combined list by date descending (most recent first)
    6. Return deduplicated sorted list

    If total headlines < MIN_HEADLINES → return empty list
    (caller will handle fallback to price-only)
    """
    yfin = fetch_yfinance_news(ticker)
    rss = fetch_google_news(company_name)
    ls = yfin + rss

    ordered_list = _deduplicate(ls)
    
    ordered_list.sort(key=lambda x: x['publishTime'], reverse=True)

    if len(ordered_list) < MIN_HEADLINES:
        return []
    
    return ordered_list

# ── Sentiment Scoring ─────────────────────────────────────────────────────────

def score_headline(finbert, headline: str) -> float:
    """
    Score a single headline using FinBERT.
    Returns a float in [-1, 1].


    TODO:
    1. Run finbert(headline) → list of dicts like:
       [{'label': 'positive', 'score': 0.82},
        {'label': 'negative', 'score': 0.10},
        {'label': 'neutral',  'score': 0.08}]
    2. Extract positive_prob and negative_prob
    3. Return positive_prob - negative_prob
       → +1.0 = fully positive, -1.0 = fully negative, 0 = neutral

    Truncate headline to 400 chars before passing to FinBERT —
    transformer models have max token limit.
    """
    pass


def aggregate_sentiment(finbert, news: list[dict]) -> dict:
    """
    Score all headlines and aggregate into a single sentiment score.

    TODO:
    1. Group headlines by date (just the date part, not time)
    2. For each day, average the scores of all headlines that day
       → gives one score per day
    3. Apply RECENCY_WEIGHTS:
       - Sort days descending (most recent = index 0)
       - Multiply each day's score by RECENCY_WEIGHTS[i]
       - Weighted average: sum(score_i * weight_i) / sum(weights used)
       Note: only use as many weights as you have days of data
    4. Return a dict:
       {
           'score': float,          # final weighted score [-1, 1]
           'label': str,            # 'BULLISH'/'BEARISH'/'NEUTRAL'
           'headline_scores': list, # [(title, score, date), ...]
           'num_headlines': int
       }

    Label thresholds:
    score > 0.15  → 'BULLISH'
    score < -0.15 → 'BEARISH'
    else          → 'NEUTRAL'
    """
    pass


# ── Main Entry Point ──────────────────────────────────────────────────────────

def get_sentiment(ticker: str, company_name: str) -> dict | None:
    """
    Full sentiment pipeline for one stock.

    TODO:
    1. load_finbert()
    2. fetch_news(ticker, company_name)
    3. If news is empty → return None (signals price-only fallback)
    4. aggregate_sentiment(finbert, news)
    5. Return result dict
    """
    pass