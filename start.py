from dataset_collection.collector_data import DataCollector
from dataset_analysis.pos_tagging import pos_tag
from dataset_analysis.stemming import stem_data
from tokenizer.regex import Tokenizer
from further_analysis.normal_pos_tagging import pos_tag as extension_pos_tag
from further_analysis.count_tokenizer import count_token_regex, count_token_crf
from further_analysis.crf_pos_tag import start_crf_pos_tag


if __name__ == "__main__":

    # Collect data
    data_collector = DataCollector()
    data_collector.start_data_collection()

    # tag the data
    pos_tag("./data/")
    # stem the data
    stem_data("./data/")

    # tokenizer using regex
    tokenizer = Tokenizer()
    tokenizer.start_tokenize("throw new UploadException")

    # pos tag using our tokenizer
    extension_pos_tag('./data/')

    # pos tag using crf
    start_crf_pos_tag('./data/train_data.json')

    # get top irregular token using crf and regex
    # CRF tokenizer will give accuracy, prediction, recall and f1
    count_token_crf('./data/')
    count_token_regex('./data/')
