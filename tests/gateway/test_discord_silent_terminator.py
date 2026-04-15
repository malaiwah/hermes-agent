"""Tests for the [SILENT] bot-to-bot chain terminator in Discord gateway."""

import re
import unittest
from unittest.mock import MagicMock


def _make_author(*, bot: bool = True, user_id: int = 12345):
    author = MagicMock()
    author.bot = bot
    author.id = user_id
    return author


def _make_message(content: str, *, bot: bool = True, mentions=None):
    msg = MagicMock()
    msg.author = _make_author(bot=bot)
    msg.content = content
    msg.mentions = mentions or []
    return msg


def _is_silent(content: str) -> bool:
    """Replicate the [SILENT] check logic from discord.py on_message."""
    cleaned = re.sub(r"<@!?\d+>", "", content).strip().strip("`").strip("*").strip("_").strip()
    return cleaned == "[SILENT]"


class TestSilentTerminator(unittest.TestCase):
    """[SILENT] is a chain terminator — gateway drops the message without routing."""

    def test_bare_silent_matches(self):
        self.assertTrue(_is_silent("[SILENT]"))

    def test_silent_with_whitespace_matches(self):
        self.assertTrue(_is_silent("  [SILENT]  "))

    def test_silent_with_backticks_matches(self):
        """LLMs often wrap tokens they've seen in backticks."""
        self.assertTrue(_is_silent("`[SILENT]`"))

    def test_silent_with_bold_matches(self):
        self.assertTrue(_is_silent("**[SILENT]**"))

    def test_silent_with_italic_matches(self):
        self.assertTrue(_is_silent("_[SILENT]_"))

    def test_silent_with_mention_prefix_matches(self):
        """With allow_bots=mentions a bot must @mention recipient; strip it before check."""
        self.assertTrue(_is_silent("<@1492868467312033972> [SILENT]"))

    def test_silent_with_excl_mention_prefix_matches(self):
        """<@!id> form (nickname mention) should also be stripped."""
        self.assertTrue(_is_silent("<@!1492868467312033972> [SILENT]"))

    def test_silent_with_mention_and_backticks_matches(self):
        self.assertTrue(_is_silent("<@1492868467312033972> `[SILENT]`"))

    def test_silent_embedded_in_sentence_does_not_match(self):
        """[SILENT] that is part of a sentence must NOT trigger the gate."""
        self.assertFalse(_is_silent("I'm sending [SILENT] now"))

    def test_silent_with_extra_text_does_not_match(self):
        self.assertFalse(_is_silent("[SILENT] Thanks for the session!"))

    def test_non_silent_message_does_not_match(self):
        self.assertFalse(_is_silent("Task complete. Reporting results."))

    def test_empty_message_does_not_match(self):
        self.assertFalse(_is_silent(""))

    def test_mention_only_does_not_match(self):
        self.assertFalse(_is_silent("<@1492868467312033972>"))


class TestInBotThreadBypass(unittest.TestCase):
    """in_bot_thread bypass must not apply to bot senders."""

    def _check_in_bot_thread(self, is_thread: bool, thread_id: str,
                              participated_threads: set, is_bot: bool) -> bool:
        """Replicate the in_bot_thread logic from discord.py."""
        return (
            is_thread
            and thread_id in participated_threads
            and not is_bot
        )

    def test_human_in_participated_thread_gets_bypass(self):
        result = self._check_in_bot_thread(
            is_thread=True,
            thread_id="111",
            participated_threads={"111"},
            is_bot=False,
        )
        self.assertTrue(result)

    def test_bot_in_participated_thread_does_not_get_bypass(self):
        """Bots must always use an explicit @mention, even in participated threads."""
        result = self._check_in_bot_thread(
            is_thread=True,
            thread_id="111",
            participated_threads={"111"},
            is_bot=True,
        )
        self.assertFalse(result)

    def test_human_in_non_participated_thread_no_bypass(self):
        result = self._check_in_bot_thread(
            is_thread=True,
            thread_id="999",
            participated_threads={"111"},
            is_bot=False,
        )
        self.assertFalse(result)

    def test_not_a_thread_no_bypass(self):
        result = self._check_in_bot_thread(
            is_thread=False,
            thread_id="111",
            participated_threads={"111"},
            is_bot=False,
        )
        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
