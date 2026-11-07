import re, unicodedata

_ONES = ["సున్నా","ఒకటి","రెండు","మూడు","నాలుగు","ఐదు","ఆరు","ఏడు","ఎనిమిది","తొమ్మిది","పది",
         "పదకొండు","పన్నెండు","పదమూడు","పద్నాలుగు","పదిహేను","పదహారు","పదిహేడు","పద్దెనిమిది","పంతొమ్మిది"]
_TENS = ["","","ఇరవై","ముప్పై","నలభై","యాభై","అరవై","డెబ్బై","ఎనభై","తొంభై"]

def _below100(n):
    if n < 20: return _ONES[n]
    return _TENS[n // 10] + (" " + _ONES[n % 10] if n % 10 else "")

def num_to_words(n):
    if n < 100: return _below100(n)
    if n < 1000:
        h, r = divmod(n, 100)
        if h == 1: return "వంద" if r == 0 else "నూట " + _below100(r)
        return _ONES[h] + (" వందలు" if r == 0 else " వందల " + _below100(r))
    if n < 10000:
        t, r = divmod(n, 1000)
        head = "వెయ్యి" if t == 1 else _ONES[t] + (" వేలు" if r == 0 else " వేల")
        return head + ((" " + num_to_words(r)) if r else "")
    return " ".join(_ONES[int(d)] for d in str(n))

def _ordinal(word):
    if word.endswith("ు") or word.endswith("ి"): return word[:-1] + "వ"
    return word + "వ"

def normalize(text):
    t = unicodedata.normalize("NFC", text).replace("\u200c", "").replace("\u200d", "")
    t = t.translate({0x0C66 + i: ord(str(i)) for i in range(10)})
    def _time(m):
        h, mi = int(m.group(1)), int(m.group(2))
        return " " + num_to_words(h) + ("" if mi == 0 else " " + num_to_words(mi)) + " "
    t = re.sub(r"(\d{1,2}):(\d{2})", _time, t)
    t = re.sub(r"(\d+)వ", lambda m: " " + _ordinal(num_to_words(int(m.group(1)))) + " ", t)
    t = re.sub(r"\d+", lambda m: " " + num_to_words(int(m.group(0))) + " ", t)
    t = re.sub(r"[^\u0C00-\u0C7F\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()

def to_tokens(text):
    toks = []
    for i, w in enumerate(normalize(text).split()):
        if i: toks.append("wb")
        toks.extend(list(w))
    return toks
