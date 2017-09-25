from dataset_collection.collector_data import DataCollector
from dataset_analysis.pos_tagging import pos_tag
from dataset_analysis.stemming import stem_data

if __name__ == "__main__":
    data_collector = DataCollector()
    data_collector.start_data_collection()

    pos_tag("data/")
    stem_data("data/")
