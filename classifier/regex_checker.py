import re


def regex_checking(sentence):
    k = re.findall(
        r"([Nn][Oo][Tt][Hh][Ii][Nn][Gg]|[Nn][Oo][Tt]?|[Nn][Ee][Ii][Tt][Hh][Ee][Rr]|[Nn][Oo][Bb][Oo][Dd][Yy]|[Nn][Oo][Nn][Ee]|[Nn][Ee][Vv][Ee][Rr]|[a-zA-Z\S]*[Nn]\'?[Tt])",
        sentence)

    if len(k) > 0:
        return True
    else:
        return False
