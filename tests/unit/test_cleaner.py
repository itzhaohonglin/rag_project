from backend.ingestion.processor.cleaner import TextCleaner


class TestTextCleaner:
    def test_clean_extra_spaces(self):
        cleaner = TextCleaner()
        result = cleaner.clean("hello    world")
        assert result == "hello world"

    def test_clean_excessive_newlines(self):
        cleaner = TextCleaner()
        result = cleaner.clean("a\n\n\n\n\nb")
        assert result == "a\n\nb"

    def test_strip_whitespace(self):
        cleaner = TextCleaner()
        result = cleaner.clean("  hello world  ")
        assert result == "hello world"

    def test_clean_html(self):
        cleaner = TextCleaner()
        result = cleaner.clean_html("<p>hello <b>world</b></p>")
        assert result == "hello world"
