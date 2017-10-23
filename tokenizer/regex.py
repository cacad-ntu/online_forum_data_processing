import re
import nltk
import copy


class Tokenizer:

    def __init__(self):
        return

    def start_tokenize(self, string):

        list_java_token = [
                            # test._this_method()
                            r"[a-zA-Z0-9_\.]+\([a-zA-Z0-9_]*\)",
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
        list_unk_token = [
            # php code such as edward -> edward, $test
            r"[A-Za-z0-9\_\$\-\>]*\$[A-Za-z0-9\_\$\-\>]+|[A-Za-z0-9\_\$\-\>]+\$[A-Za-z0-9\_\$\-\>]*",
            # url
            r"\https?\:\/\/[a-zA-Z0-9\S]*\.[\com|\org\|net][\/a-zA-Z0-9\S]*"
        ]

        buffer_string = copy.deepcopy(string)
        check_set_token_java = set()
        check_set_token_unk = set()

        for regex_unk in list_unk_token:

            m = re.findall(regex_unk, string)

            for str_find in m:

                check_set_token_unk.add(str_find)
                string.replace(str_find, "")

        sentence_token = nltk.word_tokenize(buffer_string)
        list_pure_token = []

        for regex_java in list_java_token:

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
                    list_pure_token.append({sentence_check: 'JAVA'})
                    has_irregular_token = True
                    skip = j+1
                    break

                if sentence_check in check_set_token_unk:
                    list_pure_token.append({sentence_check: 'UNK'})
                    has_irregular_token = True
                    skip = j+1
                    break

            if not has_irregular_token:
                list_pure_token.append({sentence_token[cnt]: sentence_token[cnt]})
            else:
                cnt = skip

            cnt += 1

        print list_pure_token
