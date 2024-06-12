from emoji import emoji_normalize, emoji_isolate, ascii_normalize

def preprocess(text):
    text = ascii_normalize(text) or text
    text = emoji_normalize(text) or text
    text = emoji_isolate(text) or text
    return text