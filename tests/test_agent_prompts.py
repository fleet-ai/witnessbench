"""The system prompt and the user message of the reflection call (`witness_agent.prompts`).

`agent/units/kas_lines`, `rules_list`, `notices_block`, `user_message_grid` and `user_message_example` hold the
expected texts for their inputs; the digests of the static texts and hand-derived values are written here.
"""
from __future__ import annotations

import copy
import hashlib
import re
import string

import pytest

from golden import GoldenStore
from test_agent_board import semantics_from_json
from witness_agent.actions import ActionSemantics, MoveSemantics
from witness_agent.knowledge import MAX_RULES, Rule
from witness_agent.parse import PLAN_CAP
from witness_agent.prompts import (CONFIRM_NOTE, MAX_NOTICES_SHOWN, PLAN_CAP_NOTE, RESET_NOTE, STATIC_HEAD,
                                   USER_TEMPLATE, dissonance_warning, kas_text, notices_block, rules_text,
                                   system_prompt, user_message)

FIRST_CALL_KAS = "  ACTION5: CONFIRM/SUBMIT (given), pressed ×0 this level, 0 rejected"
NO_OBSERVATIONS = "No informative observations yet."
BOARD = "Grid: 3×3 dot intersections (cursor moves on dots, not in cells) | Agent at row=2, col=0\nLevel: 1/10"
SHOWCASE = ("Action effects (one observed movement per action; CONFIRM and RESET are described in the system prompt):\n"
            "  ACTION1: @(2,0)→(1,0)")
LEVEL_ENTRY_LINE = "  #0 ACTION5: LEVEL 1 COMPLETE after 9 actions — now on level 2/10: fresh board [level_complete]"
# The seven rules of the worked example: every combination of the three tags, the 0.8 confidence boundary and
# braces in a rule text.
EXAMPLE_RULES = (
    Rule("ACTION1 moves the cursor up one dot.", 0.976, evidence_for=4, source="verified"),
    Rule("Squares of different colors must be separated.", 0.4, source="carried"),
    Rule("exactly at 0.8", 0.8),
    Rule("just below 0.8", 0.7999999999999999),
    Rule("ACTION3 is a no-op {sometimes}", 0.4, evidence_for=1, evidence_against=2),
    Rule("carried and disputed", 0.9, evidence_against=3, source="carried"),
    Rule("carried and confident", 0.974, evidence_for=2, source="carried"),
)
EXAMPLE_RULES_TEXT = ("  [1] ACTION1 moves the cursor up one dot. ✓\n"
                      "  [2] Squares of different colors must be separated. [carried]\n"
                      "  [3] exactly at 0.8 ✓\n"
                      "  [4] just below 0.8\n"
                      "  [5] ACTION3 is a no-op {sometimes} ⚠DISPUTED\n"
                      "  [6] carried and disputed ⚠DISPUTED [carried]\n"
                      "  [7] carried and confident ✓ [carried]")
EXAMPLE_WARNING = ("\n⚠ ATTENTION — The following rules have CONFLICTING evidence and may be wrong. "
                   "Review carefully and consider deleting:\n"
                   "  Rule [5]: \"ACTION3 is a no-op {sometimes}\" (support=1, contradictions=2)\n"
                   "  Rule [6]: \"carried and disputed\" (support=0, contradictions=3)")


def md5(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def cases(topic: str) -> list[dict]:
    return GoldenStore().unit_cases(topic)


def rule(stored: list) -> Rule:
    text, confidence, evidence_for, evidence_against, source = stored
    return Rule(text, confidence, evidence_for, evidence_against, source)


def message_arguments(stored: dict) -> dict:
    return {**stored, "rules": [rule(r) for r in stored["rules"]]}


def move(action: int, status: str = "untested", direction: str | None = None, times: int = 0,
         noop_rate: float | None = None, inverse: int | None = None) -> MoveSemantics:
    return MoveSemantics(action, status, direction, times, noop_rate, inverse)


def message(*, showcase: str = "", window: str = NO_OBSERVATIONS, rules=(), notices=(), attempts: str = "") -> str:
    return user_message(board=BOARD, kas=FIRST_CALL_KAS, showcase=showcase, window=window, rules=rules,
                        notices=notices, attempts=attempts)


def observations_section(user: str) -> tuple[int, str]:
    """(step count of the heading, section body) of a user message."""
    m = re.search(r"\n## Recent Observations \(last (\d+) steps\)\n(.*?)\n\n## Current Knowledge ", user, re.DOTALL)
    assert m is not None
    return int(m.group(1)), m.group(2)


# ----------------------------------------------------------------------------------------------- static texts
def test_static_texts_have_their_exact_digests():
    assert len(STATIC_HEAD) == 4444 and STATIC_HEAD.endswith("planning further.")
    assert md5(STATIC_HEAD) == "1c60c90ca0ae57957756ce1268639bad"
    assert md5(CONFIRM_NOTE) == "ae47fd5b7116b82ec7306ce69194077a"
    assert md5(RESET_NOTE) == "6d81c143c5e14dabfeec1ece33391f1d"
    assert md5(PLAN_CAP_NOTE) == "9495d66e294a3c2adffa5d9338bcb397"
    assert md5(USER_TEMPLATE) == "ae3b05faf4757c3c53532ac193030ae7"
    assert md5(CONFIRM_NOTE + "\n\n" + RESET_NOTE + "\n\n" + PLAN_CAP_NOTE) == "a096928efeb1be142681453fd16b1de1"
    assert md5(STATIC_HEAD + "\n\n## Known Action Semantics\n") == "35751116332e31e97e873f8d2fa68c58"
    assert md5(USER_TEMPLATE[:USER_TEMPLATE.index("## Current Board State")]) == "e5604ed99364fa140fd0bfdfa17e48b9"
    instructions = USER_TEMPLATE[USER_TEMPLATE.index("Analyze the recent"):].replace("{dissonance_warning}", "")
    assert md5(instructions) == "c8b942aaecdf6b3721cec6e27037ea35"
    for text in (STATIC_HEAD, CONFIRM_NOTE, RESET_NOTE, PLAN_CAP_NOTE, USER_TEMPLATE):
        assert not text.endswith("\n")


def test_user_template_has_exactly_ten_plain_fields_in_order():
    parsed = list(string.Formatter().parse(USER_TEMPLATE))
    fields = [(name, format_spec, conversion)
              for _, name, format_spec, conversion in parsed if name is not None]
    assert fields == [(name, "", None) for name in (
        "board_state", "action_semantics", "n_obs", "observations", "n_rules", "max_rules", "knowledge_rules",
        "harness_notices", "attempts_section", "dissonance_warning")]
    assert not any(ch in literal for literal, *_ in parsed for ch in "{}")


def test_numbers_in_the_static_texts_agree_with_the_caps():
    assert f"P1: Plans must be AT MOST {PLAN_CAP} actions." in STATIC_HEAD
    assert f"R4: Maintain AT MOST {MAX_RULES} active rules." in STATIC_HEAD
    assert f"at most {PLAN_CAP} actions per plan" in PLAN_CAP_NOTE
    assert "ACTION5 is the CONFIRM/SUBMIT action" in CONFIRM_NOTE
    assert "ACTION0 is RESET" in RESET_NOTE
    assert MAX_NOTICES_SHOWN == 12


def test_system_prompt_is_the_head_the_action_semantics_and_the_three_notes():
    assert system_prompt("K") == (STATIC_HEAD + "\n\n## Known Action Semantics\nK\n\n" + CONFIRM_NOTE + "\n\n"
                                  + RESET_NOTE + "\n\n" + PLAN_CAP_NOTE)
    first = system_prompt(FIRST_CALL_KAS)
    assert (len(first), md5(first)) == (5424, "5a7f538c4d6e357418eeb8cc51cbf9c1")


# ------------------------------------------------------------------------------------------- action semantics
def test_action_semantics_text_of_every_stored_state():
    stored = cases("kas_lines")
    assert len(stored) == 204
    for case in stored:
        assert kas_text(semantics_from_json(case["input"]["semantics"])) == case["expected"]["text"], case["id"]


def test_action_semantics_lists_known_moves_in_order_and_ends_with_the_confirm_line():
    untested = ActionSemantics(tuple(move(a) for a in (1, 2, 3, 4)), 0)
    assert kas_text(untested) == FIRST_CALL_KAS
    mixed = ActionSemantics((move(1, "observed", "up", 1, 0.0), move(2, "direction_carried", "down", 8, 1 / 8),
                             move(3, "observed", "left", 4, 0.25, 4), move(4, "observed", None, 1, 1.0)), 3)
    assert kas_text(mixed) == (
        "  ACTION1: direction=up, noop=0%, observed ×1 this level\n"
        "  ACTION2: direction=down, noop=12%, direction carried from an earlier level; used ×8 here\n"
        "  ACTION3: direction=left, noop=25%, observed ×4 this level, inv=ACTION4\n"
        "  ACTION4: noop=100%, observed ×1 this level\n"
        "  ACTION5: CONFIRM/SUBMIT (given), pressed ×3 this level, 3 rejected")
    carried = ActionSemantics((move(1), move(2, "carried", "down"), move(3), move(4, "carried", "right")), 1)
    assert kas_text(carried) == (
        "  ACTION2: direction=down, learned on an earlier level — unverified here\n"
        "  ACTION4: direction=right, learned on an earlier level — unverified here\n"
        "  ACTION5: CONFIRM/SUBMIT (given), pressed ×1 this level, 1 rejected")


@pytest.mark.parametrize("rate, shown", [(1 / 8, "12%"), (3 / 8, "38%"), (5 / 8, "62%"), (1 / 3, "33%"),
                                         (2 / 3, "67%"), (1 / 200, "0%"), (7 / 26, "27%"), (0.995, "100%")])
def test_no_op_rates_are_whole_percentages_rounded_half_to_even(rate, shown):
    sem = ActionSemantics((move(1, "observed", "up", 8, rate), move(2), move(3), move(4)), 0)
    assert kas_text(sem).splitlines()[0] == f"  ACTION1: direction=up, noop={shown}, observed ×8 this level"


# ------------------------------------------------------------------------------------------------- rules list
def test_rule_list_of_every_stored_rule_list():
    stored = cases("rules_list")
    assert len(stored) == 203
    for case in stored:
        assert rules_text([rule(r) for r in case["input"]["rules"]]) == case["expected"]["text"], case["id"]


def test_rule_tags_and_the_confidence_boundary():
    assert rules_text(EXAMPLE_RULES) == EXAMPLE_RULES_TEXT
    assert rules_text([]) == "No rules discovered yet."
    assert rules_text([Rule("r", 0.8)]) == "  [1] r ✓"
    assert rules_text([Rule("r", 0.7999999999999999)]) == "  [1] r"
    assert rules_text([Rule("r", 1.0, evidence_for=5, evidence_against=1, source="verified")]) == "  [1] r ⚠DISPUTED"
    assert rules_text([Rule("a\nb ✓", 0.9, source="carried")]) == "  [1] a\nb ✓ ✓ [carried]"


def test_dissonance_warning_lists_the_rules_with_counter_evidence():
    assert dissonance_warning(EXAMPLE_RULES) == EXAMPLE_WARNING
    stored = cases("rules_list")
    assert [c["id"] for c in stored[:2]] == ["example", "example_no_counter_evidence"]
    assert stored[0]["expected"]["dissonance"] == EXAMPLE_WARNING
    assert stored[1]["expected"]["dissonance"] == ""
    for case in stored:
        assert dissonance_warning([rule(r) for r in case["input"]["rules"]]) == case["expected"]["dissonance"], \
            case["id"]


# ---------------------------------------------------------------------------------------------- notices block
def test_notices_block_of_every_stored_notice_list():
    stored = cases("notices_block")
    assert [c["id"] for c in stored] == ["notices-0", "notices-1", "notices-12", "notices-13", "notices-30",
                                         "embedded_newline"]
    for case in stored:
        assert notices_block(case["input"]["notices"]) == case["expected"]["block"], case["id"]


def test_notices_block_shows_twelve_notices_and_counts_the_rest():
    header = "\n## Harness notices (since your last reflection)\n"
    notices = [f"n{i}" for i in range(30)]
    assert notices_block([]) == ""
    assert notices_block(notices[:1]) == header + "- n0\n"
    assert notices_block(notices[:12]) == header + "\n".join(f"- n{i}" for i in range(12)) + "\n"
    assert notices_block(notices[:13]) == notices_block(notices[:12])[:-1] + "\n- (+1 more notices not shown)\n"
    assert notices_block(notices).endswith("\n- n11\n- (+18 more notices not shown)\n")
    assert notices_block(["a\nb"]) == header + "- a\nb\n"


# ----------------------------------------------------------------------------------------------- user message
def test_full_user_message_with_every_section():
    (case,) = cases("user_message_example")
    user = user_message(**message_arguments(case["input"]))
    assert user == case["expected"]["user"]
    assert md5(user) == "226296ad48835912a65adb629960c4df"


def test_user_message_of_every_combination_of_optional_sections():
    stored = cases("user_message_grid")
    assert len(stored) == 40
    for case in stored:
        args = message_arguments(case["input"])
        user = user_message(**args)
        assert user == case["expected"]["user"], case["id"]
        headings = ["Current Board State", "Known Action Semantics", "Recent Observations",
                    f"Current Knowledge ({len(args['rules'])}/15 rules)"]
        headings += ["Harness notices (since your last reflection)"] if args["notices"] else []
        headings += ["Recent Attempts (this level)"] if args["attempts"] else []
        assert re.findall(r"^## (.*?)(?: \(last \d+ steps\))?$", user, re.MULTILINE) == headings + ["Instructions"]
        assert "\n\n\n" not in user, case["id"]
        last_rule = rules_text(args["rules"]).rsplit("\n", 1)[-1]
        if not args["notices"] and not args["attempts"]:
            assert f"{last_rule}\n\n## Instructions\n" in user
        else:
            assert f"{last_rule}\n\n## " in user and "\n\n## Instructions\n" in user
        if dissonance_warning(args["rules"]):
            # the rule list of the disputed cases ends with rule [3], "delta", which has one contradiction
            assert args["rules"][2:] == [Rule("delta", 0.3, evidence_against=1)]
            assert "goals)\n\n⚠ ATTENTION" in user
            assert '\n  Rule [3]: "delta" (support=0, contradictions=1)\nRespond with:' in user
        else:
            assert "toward goals)\n\nRespond with:" in user


@pytest.mark.parametrize("window, steps", [
    (NO_OBSERVATIONS, 0),
    ("Recent actions:\n" + LEVEL_ENTRY_LINE, 0),
    ("Recent actions:\n" + LEVEL_ENTRY_LINE + "\n  #1 ACTION1: @(2,0)→(1,0) | trail +1", 1),
    ("Recent actions:\n" + "\n".join(f"  #{k} ACTION2: no change" for k in range(3, 11)), 8),
], ids=["no_observations", "level_entry_line_only", "level_entry_line_and_one_step", "eight_steps"])
def test_observations_heading_counts_the_step_lines_of_the_window(window, steps):
    for showcase in ("", SHOWCASE):
        count, _ = observations_section(message(showcase=showcase, window=window))
        assert count == steps


def test_observations_are_the_showcase_a_blank_line_and_the_window():
    window = "Recent actions:\n  #1 ACTION1: @(2,0)→(1,0) | trail +1"
    assert observations_section(message(window=window)) == (1, window)
    assert observations_section(message(showcase=SHOWCASE, window=window)) == (1, SHOWCASE + "\n\n" + window)


def test_prompt_functions_are_pure():
    rules = list(EXAMPLE_RULES)
    notices = [f"n{i}" for i in range(13)]
    attempts = "\n## Recent Attempts (this level)\nYou have made 1 attempt on this level so far.\n"
    sem = ActionSemantics((move(1, "observed", "up", 2, 0.5, 2), move(2), move(3, "carried", "left"), move(4)), 2)
    inputs = copy.deepcopy((rules, notices, sem))
    first = (kas_text(sem), rules_text(rules), dissonance_warning(rules), notices_block(notices),
             message(showcase=SHOWCASE, rules=rules, notices=notices, attempts=attempts))
    second = (kas_text(sem), rules_text(rules), dissonance_warning(rules), notices_block(notices),
              message(showcase=SHOWCASE, rules=rules, notices=notices, attempts=attempts))
    assert first == second
    assert (rules, notices, sem) == inputs
