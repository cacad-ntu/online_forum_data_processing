from tokenizer.regex import Tokenizer

if __name__ == "__main__":
    sentence = "From the other side, if you want to do it using C#, which will run on both Windows and Linux--with some limitations (EDIT: which may be out of date. I have no way to test it.). Just create a SerialPort object, set its baudrate, port and any other odd settings, call open on it, and write out your byte[]s. After all the setup, the SerialPort object acts very similar to any networked stream, so it should be easy enough to figure out.\nAnd as ibrandy states, you need to know all these settings, like baud rate, before you even start attempting to communicate to any serial device."

    tokenizer = Tokenizer()

    print tokenizer.start_tokenize(sentence)
