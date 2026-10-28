"""Reading completions of the reflection call: tag contents and plans (`witness_agent.parse`).

`agent/units/plan_parse` holds the expected plan, notices and cap note of each stored completion, and the
episodes under `agent/episodes/` hold the plan of every answered call; hand-derived values are written here.
"""
from __future__ import annotations

import random
import re

import pytest

from golden import GoldenStore
from witness_agent.parse import PLAN_CAP, ParsedResponse, PlanResult, parse_plan, parse_response

PLAN_CASES = {case["id"]: case for case in GoldenStore().unit_cases("plan_parse")}
TERMINATORS = ("<|im_end|>", "<|endoftext|>")
NOTICE_PATTERNS = (
    re.compile(r"ignored plan token '\d{2,}' — write one action id per comma \(e\.g\. 1,4,5\)"),
    re.compile(r"ignored plan token '\d' — not an available action id"),
    re.compile(r"plan cap: your plan had (\d+) actions; only the first 16 ran — the last (\d+) were not executed"),
    re.compile(r"ignored text in your plan: '[^'\n]{1,60}' — only action ids are read \(e\.g\. 1,4,5\)"),
    re.compile(r"ignored plan token '0' — RESET is not read from a plan with words or parentheses "
               r"\(write e\.g\. 0,1,4\)"),
)
# Atoms of the seeded random plan blocks: digits, multi-digit runs, separators, ACTION words, brackets, prose,
# a non-ASCII digit (ARABIC-INDIC THREE), a fullwidth digit (FULLWIDTH FIVE), a no-break space and tag fragments.
ATOMS = ("1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "12", "35", "10", ",", ", ", " ", "\n", ";", "→", "-",
         "->", ">", "ACTION", "action", "Action ", "[", "]", "(", ")", ".", "x", "go", "RESET", "\u0663", "\uff15",
         "\u00a0", "\t", "<", "/", "then")


def plan_of(response: str) -> PlanResult:
    return parse_plan(parse_response(response).plan_blocks)


def cap_notice(written: int) -> str:
    return (f"plan cap: your plan had {written} actions; only the first 16 ran — "
            f"the last {written - 16} were not executed")


def without_terminator(text: str) -> str:
    for terminator in TERMINATORS:
        if text.endswith(terminator):
            return text[:-len(terminator)]
    return text


def test_tags_are_read_anywhere_in_text_order():
    text = ("<meta>I will <delete>1</delete> later</meta><DELETE> rule 2 </DELETE><add> a<delete>3</delete>b </add>"
            "<ADD>x</ADD><add>y</add>\n<plan>1</plan><|im_end|>")
    assert parse_response(text) == ParsedResponse(deletes=("1", "rule 2", "3"), adds=("a<delete>3</delete>b", "y"),
                                                  plan_blocks=("1",))


@pytest.mark.parametrize("text, expected", [
    ("<delete>1</DELETE>", ParsedResponse(("1",), (), ())),
    ("<Delete>\n 2 \n</delete>", ParsedResponse(("2",), (), ())),
    ("<add>a<add>b</add>", ParsedResponse((), ("a<add>b",), ())),
    ("<add>a</add><add>b</add>", ParsedResponse((), ("a", "b"), ())),
    ("<add>line one\nline two</add>", ParsedResponse((), ("line one\nline two",), ())),
    ("<ADD>x</ADD><Add>y</Add><PLAN>1</PLAN><Plan>2</Plan>", ParsedResponse((), (), ())),
    ("<plan >1</plan><plan>2", ParsedResponse((), (), ())),
    ("<plan> 1, 2 \n</plan>", ParsedResponse((), (), (" 1, 2 \n",))),
    ("<plan>1</plan ><plan>2</plan>", ParsedResponse((), (), ("2",))),
    ("<add>use <plan>1,2</plan> later</add>", ParsedResponse((), ("use <plan>1,2</plan> later",), ("1,2",))),
    ("<delete>rule_number</delete><keep/>", ParsedResponse(("rule_number",), (), ())),
], ids=["mixed_case_delete", "delete_content_stripped", "add_not_nested", "two_adds", "add_spans_lines",
        "upper_case_add_and_plan", "inexact_or_unclosed_plan", "plan_block_raw", "plan_close_with_space",
        "tags_inside_add", "instruction_echo"])
def test_tag_contents_by_case_nesting_and_scope(text, expected):
    assert parse_response(text) == expected


@pytest.mark.parametrize("case_id", PLAN_CASES)
def test_plan_notices_and_cap_note_of_every_stored_completion(case_id):
    case = PLAN_CASES[case_id]
    result = plan_of(case["input"]["response"])
    expected = case["expected"]
    plan = None if expected["plan"] is None else tuple(expected["plan"])
    assert result == PlanResult(plan, tuple(expected["notices"]), expected["cap_note"])


@pytest.mark.parametrize("response, actions", [
    ("<plan>1 → 2</plan><plan>go 3</plan>", (1, 2)),
    ("<plan>1>2</plan><plan>go 3</plan>", (1, 2)),
    ("<plan>1-2</plan><plan>go 3</plan>", (1, 2)),
    ("<plan>1;2</plan><plan>go 3</plan>", (1, 2)),
    ("<plan>action 1</plan><plan>go 2</plan>", (1,)),
    ("<plan>ACTION 1</plan><plan>go 2</plan>", (1,)),
    ("<plan>1,2</plan><plan>3.</plan>", (1, 2)),
    ("<plan>1</plan><plan>[2]</plan>", (1,)),
    ("<plan>1</plan><plan>RESET,2</plan>", (1,)),
], ids=["arrow_is_clean", "greater_than_is_clean", "hyphen_is_clean", "semicolon_is_clean",
        "lower_case_action_word_is_clean", "action_word_with_space_is_clean", "period_is_not_clean",
        "brackets_are_not_clean", "reset_word_is_not_clean"])
def test_the_last_clean_block_wins_over_a_later_unclean_block(response, actions):
    """The separators and the ACTIONn word keep a block clean; a period, brackets and the word RESET do not."""
    assert plan_of(response) == PlanResult(actions, (), "")


@pytest.mark.parametrize("response, actions, notices, cap_note", [
    ("<plan>" + ",".join(["1"] * 22) + "</plan>", (1,) * 16, (cap_notice(22),),
     "6 trailing action(s) cut by the plan cap"),
    ("<plan>" + ",".join(["1"] * 16 + ["2"] * 6) + "</plan>", (1,) * 16, (cap_notice(22),),
     "6 trailing action(s) cut by the plan cap"),
    ("<plan>" + ",".join(["2"] * 16) + "</plan>", (2,) * 16, (), ""),
    ("<plan>" + ",".join(["3"] * 17) + "</plan>", (3,) * 16, (cap_notice(17),),
     "1 trailing action(s) cut by the plan cap"),
    ("<plan>" + "12" * 10 + "</plan>", (1, 2) * 8, (cap_notice(20),), "4 trailing action(s) cut by the plan cap"),
], ids=["22_actions", "16_ones_then_6_twos", "16_actions", "17_actions", "bare_run_of_20"])
def test_the_plan_cap_keeps_the_first_16_actions(response, actions, notices, cap_note):
    assert PLAN_CAP == 16
    assert plan_of(response) == PlanResult(actions, notices, cap_note)


@pytest.mark.parametrize("response, actions, notices", [
    ("<plan>\u0663,\u0664</plan>", (3, 4), ()),
    ("<plan>\uff13,\uff14</plan>", (3, 4), ()),
    ("<plan>\uff16</plan>", None, ("ignored plan token '\uff16' — not an available action id",)),
    ("<plan>\uff13\uff14</plan>",
     None, ("ignored plan token '\uff13\uff14' — write one action id per comma (e.g. 1,4,5)",)),
    ("<plan>\u00b3,1</plan>", (1,), ("ignored text in your plan: '\u00b3' — only action ids are read (e.g. 1,4,5)",)),
    ("<plan>1\u00a02</plan>", (1, 2), ()),
], ids=["arabic_indic_digits", "fullwidth_digits", "fullwidth_six", "fullwidth_run_not_split",
        "superscript_is_not_a_digit", "no_break_space_separates"])
def test_decimal_digits_and_whitespace_of_every_script_count(response, actions, notices):
    assert plan_of(response) == PlanResult(actions, notices, "")


def test_completions_ending_in_a_stop_token_or_cut_short():
    """A completion ending in `<|endoftext|>`, and two recorded completions that give their recorded plans: one
    cut at the token limit (no terminator, no closed plan) and one ending in `<|im_end|>` with every kind of tag."""
    endoftext = PLAN_CASES["endoftext_after"]["input"]["response"]
    assert endoftext.endswith("<|endoftext|>")
    assert plan_of(endoftext).actions == (2, 3, 1, 4, 1, 5)
    truncated, im_end = PLAN_CASES["recorded_truncated"]["input"], PLAN_CASES["recorded_im_end"]["input"]
    assert not truncated["response"].endswith(TERMINATORS) and truncated["recorded_plan"] is None
    assert plan_of(truncated["response"]).actions is None
    assert im_end["response"].endswith("<|im_end|>") and im_end["recorded_plan"] == [4]
    parsed = parse_response(im_end["response"])
    assert (parsed.deletes, len(parsed.adds), parsed.plan_blocks) == (("1",), 2, ("4",))
    assert plan_of(im_end["response"]).actions == (4,)


@pytest.mark.parametrize("case_id", PLAN_CASES)
def test_a_stop_token_at_the_end_never_changes_the_result(case_id):
    text = PLAN_CASES[case_id]["input"]["response"]
    bare = without_terminator(text)
    texts = (text, bare) + tuple(t + terminator for t in (text, bare) for terminator in TERMINATORS)
    assert len({parse_response(t) for t in texts}) == 1
    assert len({plan_of(t) for t in texts}) == 1


@pytest.mark.parametrize("text", ["", "<|im_end|>"], ids=["empty", "stop_token_only"])
def test_an_empty_completion_has_no_tags_and_no_plan(text):
    assert parse_response(text) == ParsedResponse((), (), ())
    assert parse_plan(parse_response(text).plan_blocks) == PlanResult(None, (), "")


def assert_plan_invariants(result: PlanResult) -> None:
    if result.actions is not None:
        assert 1 <= len(result.actions) <= PLAN_CAP
        assert set(result.actions) <= {0, 1, 2, 3, 4, 5}
    else:
        assert result.cap_note == ""
    capped = bool(result.notices) and result.notices[-1].startswith("plan cap:")
    assert (result.cap_note != "") == capped
    for notice in result.notices:
        assert any(p.fullmatch(notice) for p in NOTICE_PATTERNS), notice
    if capped:
        written, dropped = map(int, NOTICE_PATTERNS[2].fullmatch(result.notices[-1]).groups())
        assert written == dropped + PLAN_CAP
        assert result.cap_note == f"{dropped} trailing action(s) cut by the plan cap"


def test_parsing_is_total_and_keeps_its_invariants_on_seeded_random_plans():
    for case in PLAN_CASES.values():
        assert_plan_invariants(plan_of(case["input"]["response"]))
    rng = random.Random(0)
    for _ in range(10_000):
        bodies = tuple("".join(rng.choice(ATOMS) for _ in range(rng.randint(0, 14)))
                       for _ in range(rng.randint(1, 3)))
        parsed = parse_response("".join(f"<plan>{body}</plan>" for body in bodies))
        assert parsed == ParsedResponse((), (), bodies)
        assert_plan_invariants(parse_plan(parsed.plan_blocks))


def test_every_answered_call_of_the_stored_episodes_gives_its_recorded_plan():
    store = GoldenStore()
    checked = 0
    for name in store.episode_names():
        episode = store.episode(name)
        for i, call in enumerate(episode.calls):
            assert plan_of(episode.responses[i]).actions == call.plan, (name, i)
            checked += 1
    assert checked > 0
