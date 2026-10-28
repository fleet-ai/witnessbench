"""Reading a completion of the reflection call: the contents of its tags and the plan it asks for.

The model answers with tags: `<delete>n</delete>` and `<add>rule</add>` edit its rule list, `<plan>ids</plan>`
asks for actions (0 RESET, 1-4 moves, 5 CONFIRM). Tags are read anywhere in the completion, independently of each
other and in text order; everything else in the completion (`<meta>`, `<keep/>`, prose) is ignored. A tag that is
opened but never closed is not read, so a completion cut short is parsed as far as it is complete. A stop-token
string at the end of the completion (such as `<|im_end|>`) never changes the result.

A plan block runs from the last `<plan>` before its `</plan>`, so a stray opener earlier (the model quoting the
format) does not pull the text in between into the block; a 0 is not read as RESET from a block with words or
parentheses (there it is mostly a coordinate such as "(2,0)"); and the words of a plan block other than ACTIONn are
quoted in a notice.

Both functions are pure and total: they return a value for every `str` and never raise.
"""
from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

PLAN_CAP: Final = 16                # at most this many actions of a plan are executed

_DELETE_RE: Final = re.compile(r"<delete>(.*?)</delete>", re.IGNORECASE | re.DOTALL)
_ADD_RE: Final = re.compile(r"<add>(.*?)</add>", re.DOTALL)
_PLAN_RE: Final = re.compile(r"<plan>((?:(?!<plan>).)*?)</plan>", re.DOTALL)   # from the last opener
# What a clean plan block may contain: ACTIONn words, digit runs and the separators , ; whitespace - > and the
# rightwards arrow (U+2192).
_CLEAN_PARTS_RE: Final = re.compile(r"ACTION\s*\d+|\d+|[,;\s→\->]+", re.IGNORECASE)
_DIGITS_RE: Final = re.compile(r"\d+")
_BARE_RUN_RE: Final = re.compile(r"\s*\d{2,}\s*")
_ADMITTED: Final = frozenset({0, 1, 2, 3, 4, 5})  # the action ids a plan may contain: the environment's actions
_SPLITTABLE: Final = "12345"        # the digits a bare run may be split into: the admitted ids other than RESET
_LETTERS_RE: Final = re.compile(r"[^\W\d_]+")
_PARENTHESIS_RE: Final = re.compile(r"[()]")
_QUOTE_TEXT: Final = 60             # longest text, in characters, the ignored-text notice quotes


@dataclass(frozen=True)
class ParsedResponse:
    """Tag contents of one completion, each tuple in text order.

    `deletes` holds every `<delete>...</delete>` content (tag in any letter case), `adds` every `<add>...</add>`
    content and `plan_blocks` every `<plan>...</plan>` content (both tags in lower case only). Delete and add
    contents are stripped of surrounding whitespace; plan blocks are kept as written.
    """

    deletes: tuple[str, ...]
    adds: tuple[str, ...]
    plan_blocks: tuple[str, ...]


@dataclass(frozen=True)
class PlanResult:
    """The plan of one completion.

    `actions` is None when the completion asks for no action; otherwise it holds 1 to `PLAN_CAP` action ids,
    each in {0, 1, 2, 3, 4, 5}. `notices` tell the model, in its next prompt, what was not executed: first one
    quoting the ignored text, then one per ignored token in token order, then one when the plan was cut to
    `PLAN_CAP` actions. `cap_note` is "" unless the plan was cut; then it says how many trailing actions were cut.
    """

    actions: tuple[int, ...] | None
    notices: tuple[str, ...]
    cap_note: str


def parse_response(text: str) -> ParsedResponse:
    """The `<delete>`, `<add>` and `<plan>` contents of the completion `text` (non-greedy, non-nested:
    `<add>a<add>b</add>` has the single content "a<add>b"; a plan block starts after the last `<plan>` before its
    `</plan>`: `<plan>a<plan>b</plan>` has the plan block "b")."""
    return ParsedResponse(
        deletes=tuple(m.group(1).strip() for m in _DELETE_RE.finditer(text)),
        adds=tuple(m.group(1).strip() for m in _ADD_RE.finditer(text)),
        plan_blocks=tuple(_PLAN_RE.findall(text)),
    )


def _is_clean(block: str) -> bool:
    return not _CLEAN_PARTS_RE.sub("", block)


def parse_plan(blocks: Sequence[str]) -> PlanResult:
    """The plan asked for by the plan blocks of one completion (`ParsedResponse.plan_blocks`).

    Block choice: the last clean block that contains a digit, where clean means nothing but ACTIONn words
    (any letter case), digit runs and the separators `,` `;` whitespace `→` `-` `>`. If no clean block has a
    digit, the last block with a digit is read instead: a list written with brackets or labels still works, and
    digits inside prose (coordinates, counts) become actions too. Blocks without a digit are ignored.

    Tokens are the maximal runs of decimal digits (of any script) of the chosen block. A single digit is
    admitted if its value is 0 (RESET) or 1-5; 6, 7, 8 and 9 are ignored. A run of two or more digits is split into
    single actions only if the chosen block is that one run and nothing else (surrounding whitespace allowed), and
    every digit is one of 1 2 3 4 5, so a split never gives RESET; any other multi-digit run is ignored. Each
    ignored token gives a notice. Of the admitted actions the first `PLAN_CAP` are kept.

    The letters of the chosen block outside ACTIONn words (of the last block with letters, when no block has a
    digit) are quoted in a notice: their runs joined by spaces, cut to 60 characters; and a 0 is ignored when the
    chosen block has such letters or a parenthesis.
    """
    with_digit = [block for block in blocks if _DIGITS_RE.search(block)]
    clean_blocks = [block for block in with_digit if _is_clean(block)]
    text = (clean_blocks or with_digit or [""])[-1]
    bare_run = _BARE_RUN_RE.fullmatch(text) is not None
    actions: list[int] = []
    notices: list[str] = []
    worded = text if with_digit else next((b for b in reversed(blocks) if _LETTERS_RE.search(b)), "")
    rest = _CLEAN_PARTS_RE.sub(" ", worded)
    letters = " ".join(_LETTERS_RE.findall(rest))
    if letters:
        notices.append(f"ignored text in your plan: '{letters[:_QUOTE_TEXT]}' — only action ids are read "
                       "(e.g. 1,4,5)")
    reset_readable = not letters and not _PARENTHESIS_RE.search(rest)
    for token in _DIGITS_RE.findall(text):
        if len(token) >= 2:
            if bare_run and all(ch in _SPLITTABLE for ch in token):
                actions.extend(int(ch) for ch in token)
            else:
                notices.append(f"ignored plan token '{token}' — write one action id per comma (e.g. 1,4,5)")
        elif int(token) == 0 and not reset_readable:
            notices.append("ignored plan token '0' — RESET is not read from a plan with words or parentheses "
                           "(write e.g. 0,1,4)")
        elif int(token) in _ADMITTED:
            actions.append(int(token))
        else:
            notices.append(f"ignored plan token '{token}' — not an available action id")
    if not actions:
        return PlanResult(None, tuple(notices), "")
    cap_note = ""
    if len(actions) > PLAN_CAP:
        written, dropped = len(actions), len(actions) - PLAN_CAP
        notices.append(f"plan cap: your plan had {written} actions; only the first {PLAN_CAP} ran — "
                       f"the last {dropped} were not executed")
        cap_note = f"{dropped} trailing action(s) cut by the plan cap"
        actions = actions[:PLAN_CAP]
    return PlanResult(tuple(actions), tuple(notices), cap_note)
