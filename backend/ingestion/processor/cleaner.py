import re


class TextCleaner:
    def clean(self, text: str) -> str:
        text = re.sub(r"\r\n", "\n", text)
        text = re.sub(r"\x00", "", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def clean_html(self, text: str) -> str:
        text = re.sub(r"<[^>]+>", "", text)
        return self.clean(text)
