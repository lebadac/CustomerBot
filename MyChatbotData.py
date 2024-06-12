
import pandas as pd
import re

class MyChatbotData:
    def __init__(self, json_obj, text_fld, answers):
        dfs = []
        for intent, data in json_obj.items():
            patterns = data[text_fld].copy()
            for i, p in enumerate(patterns):
                p = p.lower()
                p = self.remove_punctuation(p)
                patterns[i] = p
            df = pd.DataFrame(list(zip([intent]*len(patterns), patterns, [answers[intent]]*len(patterns))), \
                              columns=['intent', 'phrase', 'answer'])
            dfs.append(df)
        self.df = pd.concat(dfs)

    def get_answer(self, intent):
        answers = self.df[self.df['intent'] == intent]['answer']
        unique_answers = [answer for answer in answers]
        return unique_answers[0] if unique_answers else None




    def remove_punctuation(self, text):
        punct_re_escape = re.compile('[%s]' % re.escape('!"#$%&()*+,./:;<=>?@[\\]^_`{|}~'))
        return punct_re_escape.sub('', text)

    def get_phrases(self, intent):
        return list(self.df[self.df['intent'] == intent]['phrase'])

    def get_intents(self):
        return list(pd.unique(self.df['intent']))

    def show_batch(self, size=5):
        return self.df.head(size)

    def __len__(self):
        return len(self.df)