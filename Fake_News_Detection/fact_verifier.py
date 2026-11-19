import requests
import re

def _clean(text):
    return re.sub(r'[^a-z0-9 ]', ' ', text.lower())

def _get_wikipedia(query):
    try:
        url = "https://en.wikipedia.org/w/api.php"
        headers = {"User-Agent": "CollegeFactChecker/1.0"}
        r = requests.get(url, params={"action": "query", "list": "search", "srsearch": query, "format": "json", "srlimit": 3}, headers=headers, timeout=8)
        results = r.json().get("query", {}).get("search", [])
        if not results:
            return None, None
        title = results[0]["title"]
        r2 = requests.get(url, params={"action": "query", "prop": "extracts", "exintro": True, "explaintext": True, "titles": title, "format": "json"}, headers=headers, timeout=8)
        pages = r2.json().get("query", {}).get("pages", {})
        summary = ""
        for p in pages.values():
            summary = p.get("extract", "")
            break
        return title, summary
    except:
        return None, None

def _get_duckduckgo(query):
    try:
        r = requests.get("https://api.duckduckgo.com/", params={"q": query, "format": "json", "no_html": 1, "skip_disambig": 1}, timeout=8, headers={"User-Agent": "CollegeFactChecker/1.0"})
        d = r.json()
        abstract = d.get("AbstractText", "")
        source = d.get("AbstractSource", "")
        if abstract:
            return abstract, source
        return None, None
    except:
        return None, None

def verify_fact(text):
    if not text or len(text.strip()) < 3:
        return {"verdict": "Not Enough Info", "explanation": "Thoda detail me likho.", "source": "-"}

    user_q = text.strip()
    q_lower = user_q.lower()
    title, wiki_summary = _get_wikipedia(user_q)
    ddg_text, ddg_source = _get_duckduckgo(user_q)

    evidence = ""
    source_link = "-"
    if wiki_summary:
        evidence = wiki_summary
        source_link = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
    if ddg_text:
        evidence = (evidence + " " + ddg_text).strip()
        if source_link == "-":
            source_link = f"DuckDuckGo / {ddg_source}"

    if not evidence:
        return {"verdict": "⚠️ UNVERIFIED", "explanation": "Internet par is bare me koi pakki jankari nahi mili. Isko verified nahi mana ja sakta.", "source": "-"}

    evidence_lower = evidence.lower()
    stopwords = {"is", "the", "a", "an", "of", "in", "are", "was", "were", "ka", "ki", "kya", "hai", "hoga", "country", "desh", "sabse", "smallest", "largest", "biggest", "tallest", "longest", "highest"}
    user_words = [w for w in _clean(user_q).split() if w not in stopwords and len(w) > 2]
    superlatives = ["smallest", "largest", "biggest", "tallest", "longest", "highest", "richest", "oldest", "first"]
    is_superlative_q = any(s in q_lower for s in superlatives)

    if is_superlative_q:
        first_sentence = evidence.split(".")[0]
        found = False
        for w in user_words:
            if w in _clean(first_sentence):
                found = True
                break
        if not found and len(user_words) >= 1:
            return {"verdict": "❌ FALSE", "explanation": f"Galat hai! Asli sach ye hai: {first_sentence}.", "source": source_link}
        else:
            return {"verdict": "✅ LIKELY TRUE", "explanation": f"Wikipedia ke hisab se sahi lag raha hai: {first_sentence}.", "source": source_link}

    match_count = sum(1 for w in user_words if w in evidence_lower)
    match_ratio = match_count / max(len(user_words), 1)

    if match_ratio >= 0.6:
        verdict = "✅ LIKELY TRUE"
        expl = f"Wikipedia / Fact source ke hisab se ye sahi lag raha hai: {evidence[:500]}"
    elif match_ratio >= 0.3:
        verdict = "🔍 PARTIALLY TRUE / NEEDS CHECK"
        expl = f"Isme kuch sach hai, kuch galat. Asli jankari: {evidence[:500]}"
    else:
        verdict = "❌ LIKELY FALSE"
        expl = f"Is daave ka koi saboot nahi mila. Asli jankari ye hai: {evidence[:500]}"

    return {"verdict": verdict, "explanation": expl, "source": source_link}