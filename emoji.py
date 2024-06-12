import re
import unicodedata

EMOJIS = [[':)', '😀'],[';)', '😉'],[':(', '😞'],[';((', '😢'],[':p', '😛']]
_emoji_re = '[\U00010000-\U0010ffff]+'
emoji_re = re.compile(_emoji_re, flags=re.UNICODE)

def emoji_normalize(text):
    for e1, e2 in EMOJIS:
        text = text.replace(e1, e2)
    return text

def is_emoji(text):
    emoji = "".join(re.findall(_emoji_re, text))
    return emoji == text

def emoji_isolate(text):
    EMJ = "__EMOJI__"
    emoji_list = re.findall(_emoji_re, text)
    text = emoji_re.sub(f" {EMJ} ", text)
    new_str, ctr = [], 0
    for tok in text.split():
        if tok == EMJ:
            new_str.append(emoji_list[ctr])
            ctr += 1
        else:
            new_str.append(tok)
    return " ".join(new_str).strip()

def ascii_normalize(text):
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode("utf-8")
