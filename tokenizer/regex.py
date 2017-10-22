import re
import nltk


class Tokenizer:

    def __init__(self):
        return

    def start_tokenize(self, string):

        list_regex_exception_token = []
        # [a-zA-Z||\s]*\Exception{1}

        set_string_match = set()
        list_java_token = []
        list_unk_token = []

        for regex_pattern in list_regex_exception_token:

            m = re.search(regex_pattern, string)

            if m is None:
                continue

            string_match = m.group(0)
            set_string_match.add(string_match)
            string = string.replace(string_match, "").strip()
            list_string_token.append({string_match: "JAVA"})

        sentence_token = nltk.word_tokenize(string)

        for sen_token in sentence_token:

            list_string_token.append({sen_token: sen_token})

        print list_string_token
