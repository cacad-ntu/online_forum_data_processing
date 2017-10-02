import json

def splitPOSTag(data_dir, splitSum, splitCount):
    with open(data_dir + "pos_tag.json") as pos_tag:
        pos_data = json.load(pos_tag)

    for i in range(0, splitSum, splitCount):
        pos_data_split = pos_data[i:i+splitCount]
        split_dir = data_dir + "pos_tag_"+ str(i) + "_" + str(i+splitCount) + ".json"
        with open(split_dir, "w") as out_file:
            json.dump(pos_data_split, out_file, indent=4)

if __name__ == "__main__":
    splitPOSTag("../data/", 100, 20)
