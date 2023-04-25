import nltk
nltk.download('punkt')


class SentenceSplit:
    """
    Split a text into sentences.
    """
    def __init__(self, language):
        self.language = language
        self.sent_detector = nltk.data.load('tokenizers/punkt/%s.pickle' % language)

    def split(self, text):
        return self.sent_detector.tokenize(text.strip())

    def __call__(self, text):
        return self.split(text)

def main():
    nltk.download()
    spliter = SentenceSplit("korean")
    print(spliter.split("안녕하세요 만나서 반가워요!"))


if __name__ == "__main__":
    main()