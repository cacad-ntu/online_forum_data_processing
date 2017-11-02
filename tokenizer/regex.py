import re
import nltk
import copy
import json


class Tokenizer:

    def __init__(self):
        self.list_java_token = []
        self.list_unk_token = []

        return

    def start_tokenize_from_folder(self, data_dir):

        result_after_token = []

        with open(data_dir + "data.json") as test:
            data = json.load(test)

        for sentence in data['list_string']:
            list_of_token = self.start_tokenize(sentence)
            list_of_token = filter(lambda data: data["token"] == "JAVA" or data["token"] == "UNK", list_of_token)
            result_after_token.append(list_of_token)

        with open(data_dir + "list_string_regex_token.json", "w") as out_file:
            json.dump(result_after_token, out_file, indent=4)

    def start_tokenize(self, string, just_return_list=False):

        list_return = []

        self.list_java_token = [
                            # test._this_method()
                            r"[a-zA-Z0-9\_]{2,}\.[a-zA-Z0-9\_]{2,}[\.a-zA-Z0-9\_]*",
                            # @supresswarning
                            r"[a-zA-Z0-9_]*@[a-zA-Z0-9_]+",
                            # List<Integer> | ArrayList<Test>
                            r"(List<[a-zA-Z0-9_]+>|ArrayList<[a-zA-Z0-9_]+>)",
                            # IlegationException() | TestException()
                            r"[a-zA-Z0-9_]*Exception\(?[a-zA-Z0-9_]*\)?",
                            # java.teyst.test
                            r"[a-zA-Z0-9_]+\.[\.a-zA-Z0-9_]*[a-zA-Z0-9_]+",
                            # JavaObject, javaObject, testParser
                            r"[a-zA-Z]+[a-z]+[A-Z][A-Za-z0-9\_]+",
                            # array byte[]s. test[]
                            r"[a-zA-Z0-9\_\.]+\[.*\][a-zA-Z0-9\.\_]*"
        ]
        self.list_unk_token = [
            # php code such as edward -> edward, $test
            r"[A-Za-z0-9\_\$\-\>]*\$[A-Za-z0-9\_\$\-\>]+|[A-Za-z0-9\_\$\-\>]+\$[A-Za-z0-9\_\$\-\>]*",
            # url
            r"\https?\:\/\/[a-zA-Z0-9\S]*\.[\com|\org\|net][\/a-zA-Z0-9\S]*"
        ]

        buffer_string = copy.deepcopy(string)
        check_set_token_java = set()
        check_set_token_unk = set()

        for regex_unk in self.list_unk_token:

            m = re.findall(regex_unk, string)

            for str_find in m:

                check_set_token_unk.add(str_find)
                string.replace(str_find, "")

        sentence_token = nltk.word_tokenize(buffer_string)
        list_pure_token = []

        for regex_java in self.list_java_token:

            m = re.findall(regex_java, string)

            for str_find in m:
                string = string.replace(str_find, "")
                check_set_token_java.add(str_find)

        cnt = 0

        while cnt < len(sentence_token):

            sentence_check = ""
            has_irregular_token = False
            skip = cnt

            for j in range(cnt, len(sentence_token)):
                sentence_check += sentence_token[j]

                if sentence_check in check_set_token_java:
                    list_return.append(sentence_check)
                    list_pure_token.append({'origin': sentence_check, 'token': 'JAVA'})
                    has_irregular_token = True
                    skip = j+1
                    break

                if sentence_check in check_set_token_unk:
                    list_return.append(sentence_check)
                    list_pure_token.append({'origin': sentence_check, 'token': 'UNK'})
                    has_irregular_token = True
                    skip = j+1
                    break

            if not has_irregular_token:
                list_return.append(sentence_token[cnt])
                list_pure_token.append({'origin': sentence_token[cnt], 'token': sentence_token[cnt]})
            else:
                cnt = skip

            cnt += 1

        if just_return_list:
            return list_return

        return list_pure_token

    def get_single_regex(self):
        combine_list = self.list_java_token + self.list_unk_token
        return '|'.join(combine_list)
