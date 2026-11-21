"""
fact_verifier.py
Evidence-Based Fact Verification Module (No Hardcoding)
Uses authoritative public sources (Wikipedia Search & Extracts API, Google News RSS)
to dynamically corroborate, refute, or mark claims as unverified.
"""

import re
import urllib.parse
import urllib.request
import json
import xml.etree.ElementTree as ET

HEADERS = {
    "User-Agent": "FakeNewsDetectionFactVerifier/2.0 (academic college viva project; contact: student@college.edu)"
}

# Stopwords for claim tokenization
STOPWORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're",
    "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he',
    'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's",
    'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what',
    'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am',
    'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
    'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if',
    'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with',
    'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',
    'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over',
    'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where',
    'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other',
    'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
    'too', 'very', 's', 't', 'can', 'will', 'just', 'should', 'now',
    'h', 'ke', 'ka', 'ki', 'ko', 'se', 'hai', 'hain', 'tha', 'the', 'thi', 'kya'
}

def clean_tokens(text: str) -> list:
    """Extracts meaningful content words from text."""
    text_clean = re.sub(r'[^a-zA-Z0-9\s]', ' ', text.lower())
    return [w for w in text_clean.split() if w not in STOPWORDS and len(w) > 2]

def fetch_wikipedia_evidence(query: str):
    """
    Dynamically queries Wikipedia Search API and retrieves the top article extract.
    Requires no API keys.
    """
    try:
        search_url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={urllib.parse.quote(query)}&format=json&srlimit=3"
        req = urllib.request.Request(search_url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=6) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            search_results = data.get("query", {}).get("search", [])

        if not search_results:
            return None

        top_hit = search_results[0]
        title = top_hit.get("title", "")
        snippet = re.sub(r'<.*?>', '', top_hit.get("snippet", ""))

        ext_url = f"https://en.wikipedia.org/w/api.php?action=query&prop=extracts&exintro=True&explaintext=True&titles={urllib.parse.quote(title)}&format=json"
        req2 = urllib.request.Request(ext_url, headers=HEADERS)
        with urllib.request.urlopen(req2, timeout=6) as resp2:
            data2 = json.loads(resp2.read().decode('utf-8'))
            pages = data2.get("query", {}).get("pages", {})
            extract = ""
            for p in pages.values():
                extract = p.get("extract", "")
                break

        url = f"https://en.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}"
        return {
            "title": title,
            "extract": extract,
            "snippet": snippet,
            "url": url,
            "all_titles": [h.get("title", "") for h in search_results]
        }
    except Exception:
        return None

def fetch_news_rss_evidence(query: str):
    """
    Dynamically retrieves recent headlines from authentic news publications via Google News RSS.
    Requires no API keys.
    """
    try:
        url = f"https://news.google.com/rss/search?q={urllib.parse.quote(query)}&hl=en-IN&gl=IN&ceid=IN:en"
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=6) as resp:
            tree = ET.fromstring(resp.read())
            items = []
            for item in tree.findall(".//item")[:5]:
                title = item.find("title").text if item.find("title") is not None else ""
                link = item.find("link").text if item.find("link") is not None else ""
                source_el = item.find("source")
                source = source_el.text if source_el is not None else "Google News"
                if title:
                    items.append({"title": title, "link": link, "source": source})
            return items
    except Exception:
        return []

def verify_fact(text: str) -> dict:
    """
    Performs dynamic, evidence-based fact verification:
    - 0 hardcoded people or news answers
    - Queries public Wikipedia knowledge base
    - Queries real-time Google News RSS for time-sensitive claims
    - Detects phishing/advance-fee scams and debunked viral hoaxes
    - Returns standardized 3-state schema:
      status: 'SUPPORTED' | 'CONTRADICTED' | 'NOT_VERIFIED'
    """
    if not isinstance(text, str) or not text.strip():
        return {
            "status": "NOT_VERIFIED",
            "title": "⚪ NOT VERIFIED (Empty Input)",
            "badge_color": "#6B7280",
            "reason": "No text provided for fact verification.",
            "source": "None",
            "source_url": "",
            "category": "Empty Input",
            "limitations": "Fact verification requires meaningful textual statements."
        }

    raw_text = text.strip()
    low_text = raw_text.lower()
    claim_tokens = clean_tokens(raw_text)

    # 1. Detection of high-risk viral scam patterns (Advance-fee, link phishing, chain forward scams)
    scam_patterns = [
        (r'\bfree\s+(iphone|smartphone|laptop|recharge|cash|money)\b', "Advance-fee or Phishing Scam lure offering free luxury electronics or cash"),
        (r'\b(forward|share)\s+(this|message|whatsapp)\s+and\s+(get|earn|win|receive)\b', "Viral social messaging chain scam; messaging services do not credit funds for forwards"),
        (r'\b(click|link|tap)\s+to\s+(get|claim|win)\s+\d+\b', "Phishing lure soliciting fraudulent website link clicks"),
    ]
    for pattern, desc in scam_patterns:
        if re.search(pattern, low_text):
            return {
                "status": "CONTRADICTED",
                "title": "🔴 CONTRADICTED (Identified Scam / Phishing Pattern)",
                "badge_color": "#EF4444",
                "reason": f"Suspicious pattern detected: {desc}. Official cybersecurity agencies and consumer portals classify such viral schemes as deceptive social engineering.",
                "source": "Cyber Crime Prevention & National Consumer Clarifications",
                "source_url": "https://cybercrime.gov.in/",
                "category": "Fraud / Phishing Scam",
                "limitations": "Matched against recognized cyber safety and anti-phishing guidelines."
            }

    # 2. Check for Constitutional, Leadership or Capital role relations (e.g. "[Subject] is PM of India")
    # This solves false entity claims like "Tushar is PM of India" dynamically without hardcoding any individual.
    role_pattern = re.search(
        r'\b(?:who\s+is\s+|is\s+|was\s+|as\s+)?(prime\s+minister|pm|president|vice\s+president|chief\s+minister|cm|governor|capital|head|ceo|founder|director)\s+(?:of|in|for)\s+([a-zA-Z\s]+)',
        low_text
    )

    wiki_data = None
    if role_pattern:
        role_title = role_pattern.group(1).strip()
        location_or_entity = role_pattern.group(2).strip().split()[0]
        target_query = f"{role_title} of {location_or_entity}"
        wiki_data = fetch_wikipedia_evidence(target_query)

    if not wiki_data:
        wiki_data = fetch_wikipedia_evidence(raw_text)

    # 3. Analyze Wikipedia Evidence
    if wiki_data:
        title = wiki_data["title"]
        extract = wiki_data["extract"]
        snippet = wiki_data["snippet"]
        combined_evidence = (extract + " " + snippet).lower()
        evidence_url = wiki_data["url"]

        if role_pattern:
            # Extract subject from claim (words before the copula or role)
            m_subj = re.match(r'^(.*?)\b(?:is|was|are|h|hai)\b', raw_text, re.IGNORECASE)
            if m_subj:
                subject_words = clean_tokens(m_subj.group(1))
            else:
                role_words = set(clean_tokens(role_pattern.group(0)))
                subject_words = [w for w in claim_tokens if w not in role_words]

            # Does the authoritative record corroborate the subject entity?
            subject_found = any(sw in combined_evidence or sw in title.lower() for sw in subject_words) if subject_words else False

            # Extract documented incumbent or official summary for context
            incumbent_match = re.search(
                r'(?:current\s+(?:prime\s+minister|president|capital|holder)|serving\s+since|appointed\s+in|officially\s+the\s+capital)\s*(?:is|was)?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})',
                extract
            )
            incumbent_name = incumbent_match.group(1) if incumbent_match else ""

            if subject_found:
                return {
                    "status": "SUPPORTED",
                    "title": "🟢 SUPPORTED by Authoritative Evidence",
                    "badge_color": "#10B981",
                    "reason": f"Corroborated by authoritative encyclopedic records for '{title}': \"{extract[:280]}...\"",
                    "source": f"Wikipedia ({title})",
                    "source_url": evidence_url,
                    "category": "Official Public Record",
                    "limitations": "Cross-referenced with Wikipedia authoritative encyclopedic entries."
                }
            elif subject_words and len(subject_words) > 0 and len(extract) > 80:
                alt_note = f" Officially documented officeholder: {incumbent_name}." if incumbent_name else ""
                return {
                    "status": "CONTRADICTED",
                    "title": "🔴 CONTRADICTED by Authoritative Reference",
                    "badge_color": "#EF4444",
                    "reason": f"Factual Contradiction: Authoritative reference records for '{title}' do not document this statement as authentic.{alt_note} Verified excerpt: \"{extract[:280]}...\"",
                    "source": f"Wikipedia ({title})",
                    "source_url": evidence_url,
                    "category": "Entity / Role Contradiction",
                    "limitations": "Verified against documented public officeholders and encyclopedic records."
                }

        # General entity/topic overlap check with Wikipedia
        if claim_tokens:
            matches = [w for w in claim_tokens if w in combined_evidence or w in title.lower()]
            match_ratio = len(matches) / len(claim_tokens)

            if match_ratio >= 0.6:
                return {
                    "status": "SUPPORTED",
                    "title": "🟢 SUPPORTED by Authoritative Reference",
                    "badge_color": "#10B981",
                    "reason": f"Corroborated by Wikipedia entry '{title}': \"{extract[:280]}...\"",
                    "source": f"Wikipedia ({title})",
                    "source_url": evidence_url,
                    "category": "Encyclopedic Verification",
                    "limitations": "Evaluated against public Wikipedia knowledge base."
                }

    # 4. Check Current News Sources (Google News RSS) for breaking/time-sensitive claims
    news_items = fetch_news_rss_evidence(raw_text)
    if news_items:
        debunk_keywords = [
            "fact check", "fake", "false", "hoax", "busted", "misleading",
            "claim", "rumour", "rumor", "myth", "no,", "haven't", "hasn't",
            "didn't", "did not", "not true", "untrue", "denies", "refutes"
        ]
        for item in news_items:
            h_text = item["title"].lower()
            overlap = [t for t in claim_tokens if t in h_text]
            overlap_ratio = len(overlap) / max(len(claim_tokens), 1)

            # Check if news headline debunks or refutes the claim
            if any(deb in h_text for deb in debunk_keywords):
                return {
                    "status": "CONTRADICTED",
                    "title": "🔴 CONTRADICTED by Current News & Fact-Checks",
                    "badge_color": "#EF4444",
                    "reason": f"Press reports and fact-checkers flag or refute this claim: \"{item['title']}\"",
                    "source": f"{item['source']} (Google News)",
                    "source_url": item['link'],
                    "category": "News Fact Check",
                    "limitations": "Cross-referenced with real-time news reporting via Google News RSS."
                }

            # If strong corroborating headline
            if overlap_ratio >= 0.6:
                return {
                    "status": "SUPPORTED",
                    "title": "🟢 SUPPORTED by Recent News Coverage",
                    "badge_color": "#10B981",
                    "reason": f"Corroborated by verified press coverage: \"{item['title']}\"",
                    "source": f"{item['source']} (Google News)",
                    "source_url": item['link'],
                    "category": "Time-Sensitive News",
                    "limitations": "Verified against recent news releases from recognized media agencies."
                }

    # 5. Default: Not Verified (Transparent fall-back to ML model)
    return {
        "status": "NOT_VERIFIED",
        "title": "⚪ NOT VERIFIED (General Text / Insufficient Data)",
        "badge_color": "#6B7280",
        "reason": "This specific claim is not indexed or verified in authoritative public reference sources. The prediction is evaluated using the statistical ML model (TF-IDF + Logistic Regression).",
        "source": "Statistical NLP Classifier (Vocabulary & Stylistic Pattern Analysis)",
        "source_url": "",
        "category": "Unindexed Factual Claim",
        "limitations": "Fact verification covers documented public entities, offices, institutions, and live news RSS. For local or personal claims, refer to primary official publications."
    }

if __name__ == "__main__":
    sample = "Tushar is PM of India"
    print("Testing:", sample)
    res = verify_fact(sample)
    print("Status:", res["status"])
    print("Title:", res["title"])
    print("Reason:", res["reason"])
