import lxml.etree
import json
from bs4 import BeautifulSoup

# question PostTypeId="1" must have AcceptedAnswerId
    # question PostTypeId="2"
    # list_question -> question ID, test question, Tags= &lt;java&gt;
    # list_answer -> find(parent_id=question_id)

# format JSON:
# {"list_string": []}


class DataCollector:

    def __init__(self, file_path="data/Posts.xml"):
        root = lxml.etree.parse(file_path)

        self.list_question = root.xpath("row[contains(@Tags, '<java>') and @AcceptedAnswerId]")
        self.list_answer = []

        for question in self.list_question:
            list_answer_from_parent = root.xpath("row[@ParentId=\"" + question.attrib.get("Id") + "\"]")
            for answer in list_answer_from_parent:
                self.list_answer.append(answer)

    def start_data_collection(self):

        dict_save = {"list_string": []}

        for question in self.list_question:
            dict_save["list_string"].append(BeautifulSoup(question.attrib.get("Body")).text)

        for answer in self.list_answer:
            dict_save["list_string"].append(BeautifulSoup(answer.attrib.get("Body")).text)

        with open('data/data.json', 'w') as outfile:
            json.dump(dict_save, outfile)