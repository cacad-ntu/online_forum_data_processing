from tokenizer.regex import Tokenizer

if __name__ == "__main__":
    sentence = "ArrayList<String> wow dogs Test es.g. 1.5. wow dogz"

    tokenizer = Tokenizer()

    print tokenizer.start_tokenize(sentence)
