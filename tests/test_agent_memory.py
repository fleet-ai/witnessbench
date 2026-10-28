"""Rules, rule checks, carry-over, trial memory and the transition table (`witness_agent.knowledge`,
`witness_agent.trials`, `witness_agent.history`).

`agent/units/memory_cases` holds the expected results of the memory functions, one case per input, grouped by
`kind`; `agent/memory_segments/` holds the memory events of whole training levels. Both were recorded with
observation version 2: a reflection is its answer applied, then
the check the next prompt makes (`RuleStore.check`), and every check sees no direction (the inputs hold no cursor
movement). Hand-derived values are written here.
"""
from __future__ import annotations

import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import FrozenInstanceError

import pytest

from golden import STORE
from witness_agent.history import TransitionTable
from witness_agent.knowledge import Rule, RuleStore, verify_rule
from witness_agent.trials import TrialMemory


def memory_cases(kind: str) -> list[dict]:
    cases = [case for case in STORE.unit_cases("memory_cases") if case["kind"] == kind]
    assert cases, f"no memory case of kind {kind!r}"
    return cases


def cases_of(kind: str):
    return pytest.mark.parametrize("case", memory_cases(kind), ids=lambda case: case["id"])


def rule_of(fields: list) -> Rule:
    text, confidence, evidence_for, evidence_against, source = fields
    return Rule(text, confidence, evidence_for, evidence_against, source)


def fields_of(rule: Rule) -> list:
    return [rule.text, rule.confidence, rule.evidence_for, rule.evidence_against, rule.source]


def table_of(steps: list) -> TransitionTable:
    table = TransitionTable()
    for before, action, after in steps:
        table.record(before, action, after)
    return table


@contextmanager
def int_digit_limit(limit: int | None) -> Iterator[None]:
    """Set the interpreter's integer-string digit limit (as PYTHONINTMAXSTRDIGITS would) for the block; None keeps
    the current limit."""
    saved = sys.get_int_max_str_digits()
    if limit is not None:
        sys.set_int_max_str_digits(limit)
    try:
        yield
    finally:
        sys.set_int_max_str_digits(saved)


# Digit runs longer than the integer digit limit: the default limit, and the lowest limit the interpreter allows.
LONG_RUNS = pytest.mark.parametrize("limit, digits", [(None, "1" * 4301), (640, "1" * 1000)],
                                    ids=["default_limit", "limit_640"])


# ------------------------------------------------------------------------------------------------------ rules
def test_a_new_rule_has_confidence_one_half_no_evidence_and_is_immutable():
    rule = Rule("walls block")
    assert (rule.text, rule.confidence, rule.evidence_for, rule.evidence_against, rule.source) == \
        ("walls block", 0.5, 0, 0, "reflection")
    with pytest.raises(FrozenInstanceError):
        rule.confidence = 0.9


@cases_of("support")
def test_each_support_moves_the_confidence_as_in_the_golden_cases(case):
    """Golden fixture of repeated supports: the confidence after every support and the final rule."""
    rule = rule_of(case["input"]["rule"])
    confidences = []
    for _ in range(case["input"]["n"]):
        rule = rule.with_support()
        confidences.append(rule.confidence)
    assert confidences == case["expected"]["confidences"]
    assert fields_of(rule) == case["expected"]["rule_after"]


def test_a_starting_list_holds_at_most_fifteen_rules():
    with pytest.raises(ValueError) as err:
        RuleStore([Rule(f"rule number {i}") for i in range(16)])
    assert str(err.value) == "at most 15 rules, got 16"
    assert len(RuleStore([Rule(f"rule number {i}") for i in range(15)]).rules) == 15


@pytest.mark.parametrize("rule", [Rule("walls block", 0.1), Rule("walls block", 1.5),
                                  Rule("ACTION1 moves up", 0.5, -1)], ids=["below", "above", "negative"])
def test_a_starting_rule_with_confidence_outside_the_range_or_negative_evidence_is_rejected(rule):
    with pytest.raises(ValueError) as err:
        RuleStore([rule])
    assert str(err.value) == f"rule confidence must be in [0.2, 1.0] and evidence counts >= 0: {rule!r}"


def test_starting_rules_at_the_confidence_bounds_are_accepted():
    rules = (Rule("walls block", 0.2), Rule("ACTION1 moves up", 1.0))
    assert RuleStore(rules).rules == rules


def test_starting_rules_equal_ignoring_case_are_rejected():
    with pytest.raises(ValueError) as err:
        RuleStore([Rule("walls block"), Rule("WALLS BLOCK")])
    assert str(err.value) == "duplicate rule text (ignoring case): 'WALLS BLOCK'"


def test_the_checks_of_each_starting_rule_come_before_the_duplicate_check():
    with pytest.raises(ValueError) as err:
        RuleStore([Rule("walls block"), Rule("WALLS BLOCK"), Rule("abc")])
    assert str(err.value) == "rule text must be stripped and longer than 3 characters: 'abc'"


@pytest.mark.parametrize("text", [" walls block", "abc"], ids=["unstripped", "short"])
def test_a_starting_rule_text_must_be_stripped_and_longer_than_three_characters(text):
    with pytest.raises(ValueError) as err:
        RuleStore([Rule(text)])
    assert str(err.value) == f"rule text must be stripped and longer than 3 characters: {text!r}"


@pytest.mark.parametrize("rule", [Rule("walls block", 0.5, 1, 0), Rule("walls block", 0.5, 0, 2, "verified")],
                         ids=["support", "contradictions"])
def test_a_starting_rule_naming_no_action_must_have_no_evidence(rule):
    with pytest.raises(ValueError) as err:
        RuleStore([rule])
    assert str(err.value) == f"a rule naming no action ACTION1-ACTION5 must have no evidence: {rule!r}"


def test_a_starting_rule_naming_an_action_may_carry_evidence():
    rule = Rule("ACTION1 moves up", 0.2, 3, 1, "carried")
    assert RuleStore([rule]).rules == (rule,)


# ------------------------------------------------------------------------------------------------ reflections
@cases_of("apply")
def test_memory_operations_of_a_reflection_match_the_golden_cases(case):
    """Golden fixture of reflections: deletes, adds and the rule limits on a given starting list, then the check of
    the next prompt against the given transitions (no direction seen), give exactly these notices (those of the
    answer, then those of the check) and this list."""
    inp = case["input"]
    store = RuleStore([rule_of(r) for r in inp["rules"]])
    notices = store.apply_reflection(inp["deletes"], inp["adds"]) + store.check(table_of(inp["table"]), {})
    assert notices == case["expected"]["notices"]
    assert [fields_of(r) for r in store.rules] == case["expected"]["rules_after"]


@LONG_RUNS
def test_a_delete_number_longer_than_the_integer_digit_limit_names_no_rule(limit, digits):
    store = RuleStore([Rule("alpha rule", 0.9, 0, 0, "verified"), Rule("beta rule", 0.4),
                       Rule("gamma rule", 0.41, 0, 0, "carried")])
    with int_digit_limit(limit):
        notices = store.apply_reflection([digits, "2"], [])
    assert notices == [f"ignored <delete>{digits}</delete>: there is no rule [{digits}] "
                       "(the list above numbers rules 1–3)"]
    assert [r.text for r in store.rules] == ["alpha rule", "gamma rule"]


@pytest.mark.parametrize("limit", [None, 640], ids=["default_limit", "limit_640"])
def test_a_zero_padded_delete_number_names_its_rule_whatever_the_integer_digit_limit(limit):
    store = RuleStore([Rule("alpha rule"), Rule("beta rule"), Rule("gamma rule")])
    with int_digit_limit(limit):
        assert store.apply_reflection(["0" * 999 + "3", "0" * 4299 + "1"], []) == []
    assert [r.text for r in store.rules] == ["beta rule"]


# ------------------------------------------------------------------------------------------------ rule checks
@cases_of("verify")
def test_rule_checks_match_the_golden_cases(case):
    """Golden fixture of rule checks: the check of one rule against a transition table (no direction seen), and the
    rule after the check of a list holding only that rule (null: removed)."""
    rule, table = rule_of(case["input"]["rule"]), table_of(case["input"]["table"])
    check = verify_rule(rule, table, {})
    assert [check.kind, check.confidence, check.contradictions, check.reason] == case["expected"]["result"]
    store = RuleStore([rule])
    store.check(table, {})
    after = case["expected"]["rule_after"]
    assert [fields_of(r) for r in store.rules] == ([] if after is None else [after])


def test_two_transitions_are_too_few_to_confirm_a_rule():
    table = table_of([("s1", 2, "s2"), ("s2", 2, "s3")])
    check = verify_rule(Rule("ACTION2 moves down"), table, {2: "down"})
    assert (check.kind, check.confidence, check.contradictions) == ("inconclusive", 0.4, 0)


def test_contradictions_in_exactly_thirty_percent_of_the_transitions_refute_a_rule():
    table = table_of([(f"m{i}", 1, f"n{i}") for i in range(3)] + [(f"k{i}", 1, f"k{i}") for i in range(7)])
    store = RuleStore([Rule("ACTION1 does nothing")])
    assert store.check(table, {}) == [
        "harness removed rule 'ACTION1 does nothing' — 3/10 transitions contradict this rule (30%) "
        "(your recorded transitions contradict it)"]
    assert store.rules == ()


def test_the_removal_notice_quotes_the_first_120_characters_of_the_rule():
    text = ("ACTION3 does nothing at all; pressing it leaves the path and every coloured dot exactly as it was a "
            "moment ago, so skip it")
    assert len(text) > 120
    store = RuleStore([Rule(text)])
    assert store.check(table_of([("s1", 3, "s5"), ("s2", 3, "s6"), ("s3", 3, "s3")]), {}) == [
        "harness removed rule 'ACTION3 does nothing at all; pressing it leaves the path and every coloured dot "
        "exactly as it was a moment ago, so skip ' — 2/3 transitions contradict this rule (67%) (your recorded "
        "transitions contradict it)"]
    assert store.rules == ()


def test_a_confirmed_rule_drops_the_contradictions_of_earlier_checks_before_its_support():
    """A check counts the contradictions of the current table only: the rule confirmed at 0.92 (ACTION1 moved up from
    three states, no contradiction) loses the two contradictions an earlier check counted, so its support raises it to
    0.7 + 0.3 x 0.92 (with them the support alone would have lowered it to about 0.51)."""
    table = table_of([("s1", 1, "s2"), ("s2", 1, "s3"), ("s3", 1, "s4"), ("s4", 1, "s4")])
    store = RuleStore([Rule("ACTION1 moves the cursor up", 0.5, 0, 2)])
    assert store.check(table, {1: "up"}) == []
    assert store.rules == (Rule("ACTION1 moves the cursor up", 0.976, 1, 0, "verified"),)


def test_a_text_claiming_both_that_an_action_always_moves_and_never_moves_is_an_always_moves_claim():
    """Action 5 moved from three states and stayed in one. As an always-moves claim only the stay contradicts the
    rule: 1 of 4, inconclusive at 1 - 2 x 1/4, and not counted against a rule without support. As a no-op claim
    the three moves would refute it."""
    text = "ACTION5 never moves the cursor, it always submits the path"
    table = table_of([("a", 5, "b"), ("b", 5, "c"), ("c", 5, "d"), ("d", 5, "d")])
    check = verify_rule(Rule(text), table, {})
    assert (check.kind, check.confidence, check.contradictions, check.reason) == ("inconclusive", 0.5, 1, "")
    store = RuleStore([Rule(text)])
    assert store.check(table, {}) == []
    assert store.rules == (Rule(text, 0.5, 0, 0, "reflection"),)


def test_a_conditional_claim_that_an_action_does_not_move_is_not_a_no_op_claim():
    """The wall makes the claim conditional, so the three moves of action 2 do not contradict it; the check does not
    test a conditional claim either, so the rule is not confirmed and stays at most 0.4."""
    text = "ACTION2 does not move the cursor next to a wall"
    table = table_of([("a", 2, "b"), ("b", 2, "c"), ("c", 2, "d")])
    check = verify_rule(Rule(text), table, {})
    assert (check.kind, check.confidence, check.contradictions, check.reason) == ("inconclusive", 0.4, 0, "")
    store = RuleStore([Rule(text)])
    assert store.check(table, {}) == []
    assert store.rules == (Rule(text, 0.4, 0, 0, "reflection"),)


@LONG_RUNS
def test_an_action_number_longer_than_the_integer_digit_limit_names_no_action(limit, digits):
    with int_digit_limit(limit):
        alone = verify_rule(Rule(f"ACTION{digits} moves"), table_of([("s1", 1, "s2")] * 3), {})
        mixed = verify_rule(Rule(f"ACTION1 and ACTION{digits} always move the cursor"),
                            table_of([("s1", 1, "s2"), ("s2", 1, "s3"), ("s3", 1, "s4")]), {})
    assert (alone.kind, alone.confidence, alone.contradictions) == ("inconclusive", 0.4, 0)
    assert (mixed.kind, mixed.confidence, mixed.contradictions) == ("confirmed", 0.915, 0)


@pytest.mark.parametrize("limit", [None, 640], ids=["default_limit", "limit_640"])
def test_a_zero_padded_action_number_names_its_action_whatever_the_integer_digit_limit(limit):
    with int_digit_limit(limit):
        check = verify_rule(Rule("ACTION" + "0" * 999 + "1 moves the cursor up"),
                            table_of([("s1", 1, "s2"), ("s2", 1, "s3"), ("s3", 1, "s4")]), {1: "up"})
    assert (check.kind, check.confidence, check.contradictions) == ("confirmed", 0.915, 0)


# ------------------------------------------------------------------------------------------- transition table
@cases_of("outcomes")
def test_transition_table_counts_the_latest_outcome_of_each_action_per_state(case):
    """Golden fixture of transition-table outcomes: per action, (moved, stayed) over the before-keys, where only
    the latest outcome from each key counts; every recorded triple is kept in order."""
    table = table_of(case["input"]["table"])
    expected = case["expected"]
    counts = {action: list(table.outcome_counts(int(action))) for action in expected["outcome_counts"]}
    assert counts == expected["outcome_counts"]
    assert table.transitions() == tuple(map(tuple, expected["transitions"]))


def test_transition_table_keeps_repeated_steps_in_recording_order():
    table = TransitionTable()
    steps = [("a", 1, "b"), ("b", 2, "a"), ("a", 1, "b")]
    for step in steps:
        table.record(*step)
    assert table.transitions() == tuple(steps)
    assert (table.outcome_counts(1), table.outcome_counts(2)) == ((1, 0), (1, 0))


def test_transition_table_counts_a_state_whose_action_moved_then_stayed_as_stayed():
    table = TransitionTable()
    table.record("a", 1, "b")
    table.record("a", 1, "a")
    assert table.outcome_counts(1) == (0, 1)
    assert table.transitions() == (("a", 1, "b"), ("a", 1, "a"))


# ------------------------------------------------------------------------------------------------- carry-over
@cases_of("carry_over")
def test_carry_over_to_the_next_level_matches_the_golden_cases(case):
    """Golden fixture of level changes: the "Level change" notice (null: none) and the carried rules."""
    store = RuleStore([rule_of(r) for r in case["input"]["rules"]])
    assert store.carry_over() == case["expected"]["notice"]
    assert [fields_of(r) for r in store.rules] == case["expected"]["rules_after"]


def test_a_rule_exactly_at_the_store_threshold_after_the_level_end_support_is_carried():
    """A rule confirmed four times, then contradicted by 1 of 4 states over nine reflections: the level-end support
    gives 0.7 x 5/14 + 0.3 x 0.5 = 0.4 exactly, which the store threshold keeps, and the verified rule is carried
    at 0.8 x 0.4."""
    store = RuleStore([Rule("ACTION1 always moves the cursor up", 0.5, 4, 9, "verified")])
    assert store.carry_over() == ("Level change: 1 of your rules were carried over from the previous level (tagged "
                                  "[carried]; confidence reduced — verify them here).")
    assert store.rules == (Rule("ACTION1 always moves the cursor up", 0.32000000000000006, 0, 0, "carried"),)


def test_a_rule_below_the_store_threshold_that_names_coordinates_counts_as_low_confidence():
    """A new rule after one reflection on which 3 of its 11 recorded states contradicted it: the level-end support
    gives 0.7 x 1/4 + 0.3 x (1 - 2 x 3/11), below the store threshold, so the rule is not carried and counts as
    low-confidence, not as naming coordinates."""
    store = RuleStore([Rule("ACTION1 always moves the cursor up from (3,4)", 1 - 3 / 11 * 2, 0, 3)])
    assert store.carry_over() == ("Level change: 0 of your rules were carried over from the previous level (tagged "
                                  "[carried]; confidence reduced — verify them here); 1 low-confidence rule(s) were "
                                  "not carried.")
    assert store.rules == ()


# ----------------------------------------------------------------------------------------------- trial memory
@cases_of("trials")
def test_attempts_section_matches_the_golden_cases_after_every_event(case):
    """Golden fixture of trial-memory event sequences: the attempts section before the first event and after
    every event."""
    trials = TrialMemory()
    sections = [trials.section()]
    for event in case["input"]["events"]:
        name, args = event[0], event[1:]
        if name == "open":
            plan, start_key, cap_note = args
            trials.open(plan, start_key, cap_note)
        elif name == "rejected_confirm":
            trials.note_rejected_confirm(*args)
        elif name == "close_if_exhausted":
            trials.close_if_exhausted(*args)
        elif name == "reset":
            trials.reset()
        else:
            raise ValueError(f"unknown trial event {name!r}")
        sections.append(trials.section())
    assert sections == case["expected"]["sections"]


# ----------------------------------------------------------------------------------------- whole-level memory
@pytest.mark.parametrize("name", STORE.memory_segment_names())
def test_whole_level_memory_events_match_the_golden_segments(name):
    """Golden fixture of the memory events of whole training levels: replayed through a rule list starting from
    the rules at level entry, a trial memory and a transition table, every reflection sees the recorded
    attempts section, checks the rules as its prompt is assembled (no direction seen) and then applies its answer,
    giving the recorded notices (those of the check, then those of the answer) and rules, and the level change
    gives the recorded notice and carried rules."""
    segment = STORE.memory_segment(name)
    store = RuleStore([rule_of(r) for r in segment["seed_rules"]])
    trials, table = TrialMemory(), TransitionTable()
    for i, event in enumerate(segment["events"]):
        where = f"{name} event {i}"
        kind = event["t"]
        if kind == "step":
            trials.close_if_exhausted(event["remaining"])
        elif kind == "reflection":
            assert trials.section() == event["section"], where
            notices = store.check(table, {})
            notices += store.apply_reflection(event["deletes"], event["adds"])
            assert notices == event["notices"], where
            assert [fields_of(r) for r in store.rules] == event["rules_after"], where
            if event["plan"] is not None:
                trials.open(event["plan"], event["start_key"], event["cap_note"])
        elif kind == "transition":
            table.record(event["before"], event["action"], event["after"])
        elif kind == "rejected_confirm":
            trials.note_rejected_confirm(event["remaining"])
        elif kind == "level_solved":
            assert store.carry_over() == event["notice"], where
            assert [fields_of(r) for r in store.rules] == event["rules_after"], where
        else:
            raise ValueError(f"unknown memory event {kind!r} ({where})")
