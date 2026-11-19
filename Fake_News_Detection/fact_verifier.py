import re
import urllib.request
import json

# Curated reference facts from trusted public records (Govt of India, Official Portals, Encyclopedic Records)
VERIFIED_ENTITIES = [
    {
        "role_keywords": ["prime minister", "pm", "pradhan mantri"],
        "target_keywords": ["india", "bharat", "indian"],
        "true_entities": ["narendra modi", "modi", "narendra damodardas modi"],
        "entity_role": "Prime Minister of India",
        "description": "Narendra Modi is the Prime Minister of India (serving since May 2014).",
        "source": "Government of India Official Directory (pmindia.gov.in)"
    },
    {
        "role_keywords": ["president", "rashtrapati"],
        "target_keywords": ["india", "bharat", "indian"],
        "true_entities": ["droupadi murmu", "murmu", "draupadi murmu"],
        "entity_role": "President of India",
        "description": "Droupadi Murmu is the President of India (head of state since July 2022).",
        "source": "President of India Official Portal (presidentofindia.gov.in)"
    },
    {
        "role_keywords": ["vice president", "up rashtrapati"],
        "target_keywords": ["india", "bharat"],
        "true_entities": ["jagdeep dhankhar", "dhankhar"],
        "entity_role": "Vice President of India",
        "description": "Jagdeep Dhankhar is the Vice President of India and Chairman of the Rajya Sabha.",
        "source": "Vice President of India Portal (vicepresidentofindia.nic.in)"
    },
    {
        "role_keywords": ["capital", "rajdhani"],
        "target_keywords": ["india", "bharat"],
        "true_entities": ["new delhi", "delhi", "dilli"],
        "entity_role": "Capital of India",
        "description": "New Delhi is the official national capital of India.",
        "source": "Survey of India & National Portal of India (india.gov.in)"
    },
    {
        "role_keywords": ["home minister", "grah mantri"],
        "target_keywords": ["india", "bharat"],
        "true_entities": ["amit shah", "shah"],
        "entity_role": "Home Minister of India",
        "description": "Amit Shah is the Union Minister of Home Affairs of India.",
        "source": "Ministry of Home Affairs (mha.gov.in)"
    },
    {
        "role_keywords": ["defence minister", "defense minister", "raksha mantri"],
        "target_keywords": ["india", "bharat"],
        "true_entities": ["rajnath singh", "singh"],
        "entity_role": "Defence Minister of India",
        "description": "Rajnath Singh is the Union Minister of Defence of India.",
        "source": "Ministry of Defence (mod.gov.in)"
    },
    {
        "role_keywords": ["finance minister", "vitt mantri"],
        "target_keywords": ["india", "bharat"],
        "true_entities": ["nirmala sitharaman", "sitharaman"],
        "entity_role": "Finance Minister of India",
        "description": "Nirmala Sitharaman is the Union Minister of Finance of India.",
        "source": "Ministry of Finance (finmin.nic.in)"
    },
    {
        "role_keywords": ["external affairs minister", "foreign minister"],
        "target_keywords": ["india", "bharat"],
        "true_entities": ["s jaishankar", "jaishankar", "subrahmanyam jaishankar"],
        "entity_role": "External Affairs Minister of India",
        "description": "Dr. S. Jaishankar is the Minister of External Affairs of India.",
        "source": "Ministry of External Affairs (mea.gov.in)"
    },
    {
        "role_keywords": ["chief minister", "cm"],
        "target_keywords": ["uttar pradesh", "up"],
        "true_entities": ["yogi adityanath", "adityanath", "yogi"],
        "entity_role": "Chief Minister of Uttar Pradesh",
        "description": "Yogi Adityanath is the Chief Minister of Uttar Pradesh.",
        "source": "Government of Uttar Pradesh Official Portal (up.gov.in)"
    },
    {
        "role_keywords": ["chief minister", "cm"],
        "target_keywords": ["west bengal", "bengal"],
        "true_entities": ["mamata banerjee", "banerjee"],
        "entity_role": "Chief Minister of West Bengal",
        "description": "Mamata Banerjee is the Chief Minister of West Bengal.",
        "source": "Government of West Bengal Portal (wb.gov.in)"
    },
    {
        "role_keywords": ["chief minister", "cm"],
        "target_keywords": ["tamil nadu"],
        "true_entities": ["m k stalin", "stalin", "mk stalin"],
        "entity_role": "Chief Minister of Tamil Nadu",
        "description": "M.K. Stalin is the Chief Minister of Tamil Nadu.",
        "source": "Government of Tamil Nadu Portal (tn.gov.in)"
    },
    {
        "role_keywords": ["president", "potus"],
        "target_keywords": ["us", "usa", "america", "united states"],
        "true_entities": ["donald trump", "trump", "joe biden", "biden"],
        "entity_role": "President of the United States",
        "description": "Donald Trump / Joe Biden holds the executive presidency of the United States.",
        "source": "The White House Official Portal (whitehouse.gov)"
    },
    {
        "role_keywords": ["capital"],
        "target_keywords": ["us", "usa", "america", "united states"],
        "true_entities": ["washington", "washington dc", "washington d c"],
        "entity_role": "Capital of the United States",
        "description": "Washington, D.C. is the capital of the United States.",
        "source": "US Geological Survey / Official US Portal (usa.gov)"
    },
    {
        "role_keywords": ["capital"],
        "target_keywords": ["united kingdom", "uk", "britain", "england"],
        "true_entities": ["london"],
        "entity_role": "Capital of the United Kingdom",
        "description": "London is the capital city of the United Kingdom.",
        "source": "UK Government Official Portal (gov.uk)"
    },
    {
        "role_keywords": ["capital"],
        "target_keywords": ["france"],
        "true_entities": ["paris"],
        "entity_role": "Capital of France",
        "description": "Paris is the capital of France.",
        "source": "Government of the French Republic (gouvernement.fr)"
    },
    {
        "role_keywords": ["capital"],
        "target_keywords": ["japan"],
        "true_entities": ["tokyo"],
        "entity_role": "Capital of Japan",
        "description": "Tokyo is the capital city of Japan.",
        "source": "Government of Japan Portal (japan.go.jp)"
    },
    {
        "role_keywords": ["moon mission", "chandrayaan", "lunar mission"],
        "target_keywords": ["isro", "india"],
        "true_entities": ["isro", "india", "chandrayaan"],
        "entity_role": "Chandrayaan-3 Lunar Mission",
        "description": "ISRO successfully conducted soft landing near the Moon's South Pole in August 2023.",
        "source": "Indian Space Research Organisation (isro.gov.in)"
    },
    {
        "role_keywords": ["central bank", "monetary policy", "currency"],
        "target_keywords": ["india"],
        "true_entities": ["rbi", "reserve bank of india"],
        "entity_role": "Reserve Bank of India",
        "description": "The Reserve Bank of India (RBI) is the central banking institution regulating monetary policy.",
        "source": "Reserve Bank of India (rbi.org.in)"
    }
]

# Debunked viral hoaxes & common misinformation patterns
DEBUNKED_HOAXES = [
    {
        "triggers": [["free", "iphone"], ["free", "smartphone"], ["free", "laptop"]],
        "category": "Fake Government Welfare Scheme",
        "source": "PIB Fact Check (Press Information Bureau, Govt of India)",
        "reason": "The Government of India has never launched any scheme offering free iPhones or smartphones to citizens. Official fact-checkers classify this as an advance-fee phishing scam."
    },
    {
        "triggers": [["school", "closed", "forever"], ["college", "closed", "permanently"], ["exams", "cancelled", "forever"]],
        "category": "Education Sector Misinformation",
        "source": "Ministry of Education / PIB Fact Check",
        "reason": "No educational board, ministry, or Supreme Court bench has ordered the permanent closure of schools or cancellation of all exams. This is a fabricated social media rumor."
    },
    {
        "triggers": [["share", "10 lakh"], ["forward", "recharge"], ["forward", "paytm"], ["forward", "cashback"]],
        "category": "Social Media Chain Scam",
        "source": "Cyber Crime Prevention Cell & Telecom Authorities",
        "reason": "Messaging apps cannot credit funds or recharge telecom balances based on message forwards. Such messages are social-engineering phishing lures designed to steal personal data."
    },
    {
        "triggers": [["alien", "mumbai"], ["aliens", "delhi"], ["ufo", "red fort"], ["alien", "india"]],
        "category": "Extraterrestrial Sensation Hoax",
        "source": "Scientific Community & Space Research Authorities",
        "reason": "There is no authenticated scientific or governmental report of extraterrestrial beings or UFO landings in Mumbai or anywhere in India. This claim originates from clickbait hoaxes."
    },
    {
        "triggers": [["modi", "resigned"], ["pm", "resigned"]],
        "category": "Political Misinformation",
        "source": "Prime Minister's Office (pmo.gov.in) & Official Gazette",
        "reason": "Prime Minister Narendra Modi has not resigned. Official records confirm the Prime Minister is in active constitutional office."
    },
    {
        "triggers": [["2000", "gps"], ["nano", "chip", "note"]],
        "category": "Currency Technology Myth",
        "source": "Reserve Bank of India Clarification",
        "reason": "The Reserve Bank of India has repeatedly clarified that Indian currency notes do not contain nano GPS tracking chips or satellite transponders."
    },
    {
        "triggers": [["unesco", "best", "pm"], ["unesco", "best", "anthem"]],
        "category": "Viral UNESCO Hoax",
        "source": "UNESCO Official Spokesperson Statement",
        "reason": "UNESCO has officially confirmed it does not rank national anthems or name 'Best Prime Minister of the World'. This is a persistent internet hoax."
    },
    {
        "triggers": [["garlic", "cancer"], ["lemon", "oxygen"], ["onion", "corona"]],
        "category": "Unverified Medical Miracle Cure",
        "source": "World Health Organization (WHO) & ICMR Guidelines",
        "reason": "Medical authorities and WHO confirm that household spices cannot cure cancer or COVID-19 in hours. Relying on unverified claims delays vital medical care."
    }
]

def verify_fact(text: str) -> dict:
    """
    Performs fact verification against trusted reference records.
    Returns:
    {
        "status": "SUPPORTED" | "CONTRADICTED" | "NOT_VERIFIED",
        "title": str,
        "badge_color": str,
        "reason": str,
        "source": str,
        "category": str,
        "limitations": str
    }
    """
    if not isinstance(text, str) or not text.strip():
        return {
            "status": "NOT_VERIFIED",
            "title": "Not Verified (Empty Input)",
            "badge_color": "#6B7280",
            "reason": "No text provided for fact verification.",
            "source": "None",
            "category": "Empty",
            "limitations": "Fact verification requires meaningful textual claims."
        }

    clean_low = text.lower().strip()

    # 1. Check known debunked viral hoaxes
    for hoax in DEBUNKED_HOAXES:
        for trig in hoax["triggers"]:
            if all(k in clean_low for k in trig):
                return {
                    "status": "CONTRADICTED",
                    "title": "🔴 CONTRADICTED by Fact-Checking Records",
                    "badge_color": "#EF4444",
                    "reason": hoax["reason"],
                    "source": hoax["source"],
                    "category": hoax["category"],
                    "limitations": "Matched against indexed debunked misinformation and official fact-checks."
                }

    # 2. Check structured Entity-Role knowledge (Leadership, Capitals, etc.)
    for fact in VERIFIED_ENTITIES:
        has_role = any(re.search(r'\b' + re.escape(kw) + r'\b', clean_low) for kw in fact["role_keywords"])
        has_target = any(re.search(r'\b' + re.escape(t) + r'\b', clean_low) for t in fact["target_keywords"])

        if has_role and has_target:
            has_true_entity = any(val in clean_low for val in fact["true_entities"])

            if has_true_entity:
                return {
                    "status": "SUPPORTED",
                    "title": "🟢 SUPPORTED by Trusted Reference Data",
                    "badge_color": "#10B981",
                    "reason": fact["description"],
                    "source": fact["source"],
                    "category": "Factual Verification",
                    "limitations": "Cross-referenced with verified constitutional & governmental reference data."
                }
            else:
                # Contradiction: Role & Target match, but the entity is wrong!
                # (e.g. "Tushar is PM of India" or "Mumbai is capital of India")
                return {
                    "status": "CONTRADICTED",
                    "title": "🔴 CONTRADICTED by Trusted Reference Data",
                    "badge_color": "#EF4444",
                    "reason": f"Factual Contradiction: The claim associates the role of '{fact['entity_role']}' with an incorrect subject. Verified truth: {fact['description']}",
                    "source": fact["source"],
                    "category": "Entity / Role Contradiction",
                    "limitations": "Directly contradicts authoritative leadership and governmental reference databases."
                }

    # 3. Default: Not Verified
    return {
        "status": "NOT_VERIFIED",
        "title": "⚪ NOT VERIFIED (General Text / Insufficient Data)",
        "badge_color": "#6B7280",
        "reason": "This specific claim is not indexed in the local verified reference database. The classification below is determined by the statistical NLP model (TF-IDF + Logistic Regression).",
        "source": "Statistical NLP Classifier (Vocabulary & Stylistic Pattern Analysis)",
        "category": "Unindexed Factual Claim",
        "limitations": "Fact verification covers indexed constitutional roles, national capitals, space milestones, and known viral scams. For novel local events or opinion articles, manual verification against authorized news agencies is advised."
    }

if __name__ == "__main__":
    sample = "Tushar is PM of India"
    print("Testing:", sample)
    res = verify_fact(sample)
    print("Status:", res["status"])
    print("Reason:", res["reason"])
