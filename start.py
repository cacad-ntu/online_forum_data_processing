from dataset_collection.collector_data import DataCollector
from dataset_analysis.pos_tagging import pos_tag

if __name__ == "__main__":
    data_collector = DataCollector()
    data_collector.start_data_collection()

    pos_tag("data/")
