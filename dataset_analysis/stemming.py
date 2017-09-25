import json
import nltk

def stem_data(data_dir):
    stemmer = nltk.stem.porter.PorterStemmer()

    with open(data_dir + 'data.json') as data_file:
        data = json.load(data_file)

    stemmed_word_count = {}
    word_count = {}

    for item in data["list_string"]:
        for element in item.split():
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

    with open(data_dir + 'result_stemmed.json', 'w') as result:
        json.dump(stemmed_word_count, result)

    with open(data_dir + 'result_word_count.json', 'w') as result:
        json.dump(word_count, result)
