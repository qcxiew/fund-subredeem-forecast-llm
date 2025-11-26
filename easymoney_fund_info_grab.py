#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import json
import csv
import time
import os

# -- Configuration setting -- #
INPUT_SERIES_CSV = "fund_apply_redeem_series.csv"
DETAIL_URL_TMPL   = "https://fundf10.eastmoney.com/jbgk_{code}.html"
OUTPUT_JSON  = "fund_details.json"
OUTPUT_JSONL = "fund_details.jsonl"
OUTPUT_CSV   = "fund_details.csv"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    )
}
 
# Map table items on the web page to field names (like short_name, fund_type).，
# note：'基金代码' → 'fund_code_ex'；'发行日期' → 'issue_date_ex'

TABLE_FIELDS = {
    '基金全称':     'full_name',
    '基金简称':     'short_name',
    '基金代码':     'fund_code_ex',        
    '基金类型':     'fund_type',
    '发行日期':     'issue_date_ex',       # keep chinese time format，and then convert to standard format
    '成立日期/规模': 'establish_date_scale',
    '资产规模':     'asset_scale',
    '份额规模':     'share_scale',
    '基金管理人':   'manager_company',
    '基金托管人':   'custodian',
    '基金经理人':   'fund_managers',
    '成立来分红':   'dividend_since_est',
    '管理费率':     'management_fee_rate',
    '托管费率':     'custody_fee_rate',
    '销售服务费率':'sales_service_fee_rate',
    '最高认购费率':'max_subscription_fee',
    '最高申购费率':'max_purchase_fee',
    '最高赎回费率':'max_redemption_fee',
    '业绩比较基准':'performance_benchmark',
    '跟踪标的':    'tracking_target'
}
 
SECTION_LABELS = {
    '投资目标':       'investment_objective',
    '投资范围':       'investment_scope',
    '投资策略':       'investment_strategy',
    '分红政策':       'dividend_policy',
    '风险收益特征':   'risk_reward_profile'
}
 
# —— Function implementation —— #
def load_fund_codes(csv_path: str) -> list[str]:
    df = pd.read_csv(csv_path, dtype=str)
    return df['fund_code'].dropna().unique().tolist()
 
def fetch_and_parse(code: str) -> dict:
    url  = DETAIL_URL_TMPL.format(code=code)
    resp = requests.get(url, headers=HEADERS, timeout=10)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    # 1. only querying for ID
    rec = {"fund_code": code}
 
    # 2. Scrape table fields
    table = soup.find("table", class_="info w790")

    if table:
        for tr in table.find_all("tr"):
            ths = tr.find_all("th")
            tds = tr.find_all("td")
            for th, td in zip(ths, tds):
                key = TABLE_FIELDS.get(th.text.strip())
                if not key:
                    continue
                rec[key] = td.get_text(strip=True)
 
    # 3. each section's Text o 
    for box in soup.find_all("div", class_="boxitem w790"):
        title = box.find("label", class_="left").text.strip()

        key   = SECTION_LABELS.get(title)
        if key:
            rec[key] = box.find("p").get_text("\n", strip=True)
 
    # 4. Release date standardization：from "2014年04月16日" → "20140416"
    if 'issue_date_ex' in rec:
        raw = rec['issue_date_ex']
        try:
            dt = datetime.strptime(raw, "%Y年%m月%d日")
            rec['issue_date'] = dt.strftime("%Y%m%d")
        except ValueError:
            # If parsing fails, leave it as is.
            rec['issue_date'] = raw
    return rec
 
def main():
    codes = load_fund_codes(INPUT_SERIES_CSV)
    print(f"There are {len(codes)} funds in total. Start fetching details...")
 
    results = []
    for idx, code in enumerate(codes, 1):
        print(f"[{idx}/{len(codes)}] fetching {code} …", end=" ")
        try:
            record = fetch_and_parse(code)
            print("✅")
        except Exception as e:
            print(f"❌ fail：{e}")
            continue
        results.append(record)
        time.sleep(0.5)
 
    # —— write JSON —— #
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"save JSON: {OUTPUT_JSON}")
 
    # —— write JSONL —— #
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for rec in results:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"save JSONL: {OUTPUT_JSONL}")
 
    # —— write CSV —— #
    all_keys = sorted({k for rec in results for k in rec.keys()})
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        for rec in results:
            # transfer multi-line text into a single line.
            row = {
                k: (v.replace("\n", "\\n") if isinstance(v, str) else v)
                for k, v in rec.items()
            }
            writer.writerow(row)

    print(f"save CSV: {OUTPUT_CSV}")


if __name__ == "__main__":

    main()

 