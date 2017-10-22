import json
import operator
import nltk

def stem_data(data_dir):
    stemmer = nltk.stem.porter.PorterStemmer()

    with open(data_dir + 'data.json') as data_file:
        data = json.load(data_file)

    stemmed_word_count = {}
    word_count = {}

    with open(data_dir + "stop_words.json") as stop_files:
        stop_words = json.load(stop_files)

    for item in data["list_string"]:
        item_tokens = nltk.word_tokenize(item)
        for element in item_tokens:
            
            if element in stop_words:
                continue

            if element in word_count:
                word_count[element] += 1
            else:
                word_count[element] = 1

            parsed = stemmer.stem(element)

            if parsed in stemmed_word_count:
                stemmed_word_count[parsed][0] += 1
                if not element in stemmed_word_count[parsed][1]:
                    stemmed_word_count[parsed][1].append(element)
            else:
                stemmed_word_count[parsed] = [1, [element]]

    stemmed_word_count = sorted(stemmed_word_count.items(), key=lambda count: count[1][0], reverse=True)
    word_count = sorted(word_count.items(), key=operator.itemgetter(1), reverse=True)

    with open(data_dir + 'result_stemmed.json', 'w') as result:
        json.dump(stemmed_word_count, result, indent=4)

    with open(data_dir + 'result_word_count.json', 'w') as result:
        json.dump(word_count, result, indent=4)
