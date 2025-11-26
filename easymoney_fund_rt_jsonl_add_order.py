#!/usr/bin/env python3
# -*- coding: utf-8 -*-
 
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import time
import json
import os
 
# —— setting config —— #
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;q=0.9,"
        "*/*;q=0.8"
    ),
    "Connection": "close",  # Avoiding reusing connections can sometimes reduce EOF errors.
}

BASE_LIST_URL   = "https://roll.eastmoney.com/fund{}.html"
PAGE_RANGE      = range(1, 2)  # 126 page
ORIGINAL_FILE   = "fund_news.jsonl"
TEMP_NEW_FILE   = "new_fund_news_tmp.jsonl"
 
def load_existing_urls(path: str) -> set:
    urls = set()
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    urls.add(json.loads(line)["url"])
                except:
                    continue
    return urls
 
def get_original_max_datetime(path: str) -> datetime:
    max_dt = datetime.min
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    dt = datetime.fromisoformat(json.loads(line)["datetime"])
                    if dt > max_dt: max_dt = dt
                except:
                    continue
    return max_dt
 
def fetch_list_page(no: int):
    suffix = "" if no == 1 else f"_{no}"
    url    = BASE_LIST_URL.format(suffix)
    r      = requests.get(url, headers=HEADERS, timeout=10)
    r.raise_for_status()
    return BeautifulSoup(r.text, "html.parser")
 
def parse_list_items(soup):
    items = []
    for blk in soup.select("li, .newsList"):
        if not blk.find("a", string="基金"): 
            continue
        a = blk.find("a", href=True, title=True)
        if not a: 
            continue
        span = blk.find("span")
        items.append({
            "category": "基金",
            "title":    a["title"].strip(),
            "url":      a["href"],
            "time_str": span.get_text(strip=True) if span else ""
        })
    return items
 
def parse_datetime_obj(s: str):
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(s, fmt)
        except:
            pass
    return None
 

def fetch_detail_text(url: str, max_retry: int = 1, sleep_sec: float = 1.5) -> str:
    """
    Fetch news details pages.
    - Retry several times internally when encountering SSL/network errors;
    - Return an empty string instead of throwing an exception and crashing the entire script if it ultimately fails.
    """
    last_err = None
 
    for attempt in range(1, max_retry + 1):
        try:
            print(f"[detail] Fetch {url}, attempt {attempt}/{max_retry}")
            r = requests.get(url, headers=HEADERS, timeout=10)
            r.raise_for_status()
 
            # Some websites use the wrong default encoding;  best to change it to UTF-8.
            if not r.encoding or r.encoding.lower() in ("iso-8859-1", "latin-1"):
                r.encoding = "utf-8"
 
            soup = BeautifulSoup(r.text, "html.parser")
            bdy  = soup.find(id="ContentBody")
            if not bdy:
                return ""
 
            lines = [ln.strip() for ln in bdy.get_text("\n").splitlines() if ln.strip()]
            return "\n".join(lines)
 
        except requests.exceptions.SSLError as e:
            # for UNEXPECTED_EOF_WHILE_READING
            print(f"[WARN] SSL error when fetching {url}: {e!r}, retry {attempt}/{max_retry}")
            last_err = e
            time.sleep(sleep_sec)
 
        except requests.exceptions.RequestException as e:
            # Other network errors (connection failure, timeout, HTTPError, etc.)
            print(f"[WARN] Request error when fetching {url}: {e!r}, stop retry for this article")
            last_err = e
            break
 
    print(f"[ERROR] Give up fetching {url}, last error: {last_err!r}")
    # Returning an empty string allows the upper-level logic to decide whether to record empty content or skip the process.
    return ""

 
def main():
    # 1.loading URL & new time thredholds 
    existing_urls   = load_existing_urls(ORIGINAL_FILE)
    original_max_dt = get_original_max_datetime(ORIGINAL_FILE)
    print(f"have {len(existing_urls)} of URL; threholds = {original_max_dt.isoformat()}")
 
    new_records = []
 
    # 2. Iterate through pages and filter for new adding  
    for pg in PAGE_RANGE:
        soup = fetch_list_page(pg)
        for it in parse_list_items(soup):
            dt = parse_datetime_obj(it["time_str"])
            if not dt or dt <= original_max_dt:
                continue
            if it["url"] in existing_urls:
                continue
 
            # —— new added item, and print —— #
            dt_iso = dt.isoformat()
            print(f"[Add] {dt_iso} | {it['title']} | {it['url']}")

            # Fetch the main text
            content = fetch_detail_text(it["url"])
            rec = {
                "category": it["category"],
                "title":    it["title"],
                "url":      it["url"],
                "datetime": dt_iso,
                "content":  content
            }
            new_records.append(rec)
            existing_urls.add(it["url"])
            time.sleep(0.5)
 
    if not new_records:
        print("No new news, exit.")
        return
 
    # 3. Write intermediate JSONL (keeping the order of new_records unchanged).
    with open(TEMP_NEW_FILE, "w", encoding="utf-8") as tf:
        for rec in new_records:
            tf.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Intermediate files have been generated: {TEMP_NEW_FILE}containing {len(new_records)} records.")
 
    # 4. Insert the contents of the intermediate file line by line into the header of the original file.
    with open(TEMP_NEW_FILE, "r", encoding="utf-8") as tf:
        new_lines = tf.readlines()
    old_lines = []
    if os.path.exists(ORIGINAL_FILE):
        with open(ORIGINAL_FILE, "r", encoding="utf-8") as of:
            old_lines = of.readlines()
 
    with open(ORIGINAL_FILE, "w", encoding="utf-8") as of:
        of.writelines(new_lines)
        of.writelines(old_lines)
 
    os.remove(TEMP_NEW_FILE)
    print(f"Successfully inserted {len(new_lines)} new news items into the header of {ORIGINAL_FILE}")
 
if __name__ == "__main__":
    main()