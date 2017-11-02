""" module to split data for manual labeling """
import json

def split_POS_tag(data_dir, split_sum, split_count):
    """ split data from post tag """
    with open(data_dir + "pos_tag.json") as pos_tag:
        pos_data = json.load(pos_tag)

    for i in range(0, split_sum, split_count):
        pos_data_split = pos_data[i:i+split_count]
        split_dir = data_dir + "pos_tag_"+ str(i) + "_" + str(i+split_count) + ".json"
        with open(split_dir, "w") as out_file:
            json.dump(pos_data_split, out_file, indent=4)

def split_data(data_dir, split_sum, split_count):
    """ split data from raw json """
    with open(data_dir + "data.json") as raw_file:
        raw_data = json.load(raw_file)["list_string"]

    raw_data_dict = {}
    raw_data_dict["pos"] = ""
    raw_data_dict["negation_label"] = 0
    raw_data_dict["error_label"] = 0
    raw_data_dict["semantic_label"] = 0

    for i in range(0, split_sum, split_count):
        raw_data_split = []
        raw_data_range = raw_data[i:i+split_count]

        for pos in raw_data_range:
            raw_data_dict["pos"] = pos
            raw_data_split.append(raw_data_dict.copy())

        split_dir = data_dir + "data_" + str(i) + "_" + str(i+split_count) + ".json"
        with open(split_dir, "w") as out_file:
            json.dump(raw_data_split, out_file, indent=4)

if __name__ == "__main__":
    split_POS_tag("../data/", 100, 20)
    split_data("../data/", 250, 50)
