"""The wording of the version-2 prompts: the reworded system-prompt rules, the instructions, the budget line and the
rule ticks and contradictions shown to the model. Every expected value is written here by hand."""
from __future__ import annotations

import pytest

import obs_support
from witness_agent import EpisodeLimits, run_episode
from witness_agent.board import budget_line
from witness_agent.history import TransitionTable
from witness_agent.knowledge import Rule, RuleStore, verify_rule
from witness_agent.loop import MAX_STEPS_PER_LEVEL
from witness_agent.parse import parse_plan, parse_response
from witness_agent.prompts import STATIC_HEAD, system_prompt
from witness_env import make_game


class Scripted:
    def __init__(self, answers):
        self.answers = list(answers)
        self.calls = []

    def __call__(self, system, user):
        self.calls.append((system, user))
        return self.answers.pop(0) if self.answers else None


def play(answers, cap=None, levels=1, game="tw01"):
    policy = Scripted(answers)
    result = run_episode(make_game(game), seed=0, limits=EpisodeLimits(levels, max_reflection_calls=cap),
                         policy=policy)
    return result, policy.calls


def test_the_budget_line_names_both_limits():
    assert budget_line(7, 300, 12, 1000) == "Budget: actions on this level 7 of 300 | reflection calls used 12 of 1000"
    assert budget_line(0, 300, 0, None) == "Budget: actions on this level 0 of 300 | reflection calls used 0 (no limit)"


def test_version_2_shows_the_budget_after_the_level_line():
    _, calls = play(["<plan>4,4</plan>", "<keep/>"], cap=5)
    lines = calls[1][1].split("\n")
    level = next(i for i, line in enumerate(lines) if line.startswith("Level: "))
    assert lines[level + 1] == (f"Budget: actions on this level 2 of {MAX_STEPS_PER_LEVEL} | reflection calls used "
                                "1 of 5")
    assert calls[0][1].split("\n")[level + 1].endswith("actions on this level 0 of 300 | reflection calls used 0 of 5")


def test_at_the_call_cap_version_2_runs_the_last_plan_to_its_end():
    result, _ = play(["<plan>4,2,4</plan>"], cap=1)
    assert [s.action for s in result.steps] == [4, 2, 4] and result.stop_reason == "call_budget"


# The system-prompt rules as reworded for version 2 (the last one ends the fixed instructions).
REWORDED_RULES = [
    "- R3: A rule marked ⚠DISPUTED has conflicting evidence — test it or delete it; do not plan on it.\n",
    "- R7: If you are not sure a rule holds, say so in its wording (e.g. \"probably\").\n",
    "- P3: Every action of a plan is executed before you see any result; when you are unsure what an action does, "
    "plan fewer actions.\n",
    "- G1: The board shows the actions used on this level (the level ends the episode when they run out) and the "
    "reflection calls used; when little is left, spend it on your best complete attempt.\n",
    "- G3: When the same (state, action) produces different results, this is COGNITIVE DISSONANCE — resolve the "
    "contradiction in your next reflection before planning further.",
]


@pytest.mark.parametrize("line", REWORDED_RULES, ids=["R3", "R7", "P3", "G1", "G3"])
def test_version_2_rewords_each_rule_the_model_could_not_follow(line):
    """No rule names a quantity the model never sees (rule confidence, a budget share) or asks for what it cannot do
    (abandon a plan mid-way, trigger a reflection)."""
    assert STATIC_HEAD.count(line) == 1 and system_prompt("KAS").count(line) == 1


def test_version_2_says_what_follows_an_answer_without_a_plan():
    """Instruction 3 says what follows an answer without a plan (the model could not know)."""
    _, calls = play(["<keep/>"], cap=1)
    assert ("3. Optionally, if you have a specific strategy to try right now (an answer without a plan is followed by "
            "one random move):\n") in calls[0][1]


def test_version_2_column_numbers_stand_over_their_node_columns():
    _, calls = play(["<keep/>"], cap=1)
    lines = calls[0][1].split("\n")
    header, first_row = lines[lines.index("Board:") + 1], lines[lines.index("Board:") + 2]
    assert header.rstrip() == "   0 1 2" and first_row.startswith(" 0 ")
    assert all(header[3 + 2 * c] == str(c) for c in range(3))


def recent_attempts(user):
    """The lines of the Recent Attempts section of a user message between its heading and its footer."""
    lines = user.split("\n")
    start = lines.index("## Recent Attempts (this level)")
    return lines[start + 1:lines.index("(Attempt history is provided automatically — do not store attempts as "
                                       "knowledge rules.)")]


def test_version_2_counts_only_plans_with_confirm_as_attempts_and_only_resets_sent_to_the_engine():
    answers = ["<plan>4,0</plan>", "<plan>0,4</plan>", "<plan>4,5</plan>", "<keep/>"]
    result, calls = play(answers, cap=4)
    assert [(s.action, s.noop_reset) for s in result.steps[:4]] == [(4, False), (0, False), (0, True), (4, False)]
    chunks = " ({} of the plans below ran without a submit — partial chunks, not counted as attempts.)"
    first = "- chunk before attempt #1 (no submit): [4,0] → ran to its end — continue from the resulting state"
    second = "- chunk before attempt #1 (no submit): [0,4] → ran to its end — continue from the resulting state"
    assert recent_attempts(calls[1][1]) == ["You have made 0 attempts on this level so far." + chunks.format(1),
                                            first + "; included 1 reset(s)"]
    assert recent_attempts(calls[2][1]) == ["You have made 0 attempts on this level so far." + chunks.format(2),
                                            first + "; included 1 reset(s)", second]
    assert recent_attempts(calls[3][1]) == ["You have made 1 attempt on this level so far." + chunks.format(1),
                                            second, "- attempt #1: [4,5] → submit rejected at action 2/2",
                                            "    submit at action 2: submitted path (3 nodes): (2,0) → (2,1) → "
                                            "(2,2)"]


def test_version_2_quotes_each_rule_dropped_at_a_level_change_for_naming_coordinates():
    rules = [Rule("Squares of different colours must be separated (0,1)", 0.9),
             Rule("ACTION4 moves the line right", 0.9),
             Rule("Entering an earlier trail dot other than the immediate predecessor is blocked, confirmed at both "
                  "(3,1) and (2,2) on this level", 0.9)]
    carried = ("Level change: 1 of your rules were carried over from the previous level (tagged [carried]; confidence "
               "reduced — verify them here); 2 rule(s) naming specific coordinates were dropped as level-specific")
    assert RuleStore(rules).carry_over() == (
        carried + ": 'Squares of different colours must be separated (0,1)', 'Entering an earlier trail dot other "
        "than the immediate predecessor is blocked, confirmed at both (3,1) and (2,2) on this'.")


def test_version_2_names_the_dropped_rule_in_the_first_prompt_of_the_next_level():
    answer = ("<add>Squares of different colours must be separated (0,1)</add>"
              + obs_support.plan_text(obs_support.SOLUTIONS["tw01"][0]))
    notice = ("- Level change: 0 of your rules were carried over from the previous level (tagged [carried]; confidence "
              "reduced — verify them here); 1 rule(s) naming specific coordinates were dropped as level-specific")
    _, calls = play([answer, "<keep/>"], cap=2, levels=2)
    assert notice + ": 'Squares of different colours must be separated (0,1)'." in calls[1][1].split("\n")


def test_version_2_reads_a_plan_block_from_the_last_opener_before_its_closer():
    stray = 'The prompt says "Use <plan>...". Rule [7] holds at (2,0).\n<plan>1,1,4,5</plan>'
    assert parse_response(stray).plan_blocks == ("1,1,4,5",)
    assert parse_plan(parse_response(stray).plan_blocks).actions == (1, 1, 4, 5)
    twice = "<plan>3,3,5\n</meta>\n<plan>3,3,5</plan>"
    assert parse_response(twice).plan_blocks == ("3,3,5",)
    assert parse_plan(parse_response(twice).plan_blocks).actions == (3, 3, 5)


TEXT = "ignored text in your plan: '{}' — only action ids are read (e.g. 1,4,5)"
ZERO = "ignored plan token '0' — RESET is not read from a plan with words or parentheses (write e.g. 0,1,4)"


@pytest.mark.parametrize("block, expected", [
    ("D,R,R,5", ((5,), (TEXT.format("D R R"),))),
    ("ACTION4 x3, ACTION5", ((4, 3, 5), (TEXT.format("x"),))),
    ("go from start (3,0) toward END (0,0) and submit",
     ((3,), (TEXT.format("go from start toward END and submit"), ZERO, ZERO, ZERO))),
    ("(2,0)", ((2,), (ZERO,))),
    ("[0, 4, 4, 5]", ((0, 4, 4, 5), ())),
    ("0,4,4,5.", ((0, 4, 4, 5), ())),
    ("4,7,5", ((4, 5), ("ignored plan token '7' — not an available action id",))),
    ("147", (None, ("ignored plan token '147' — write one action id per comma (e.g. 1,4,5)",))),
    ("RIGHT, DOWN", (None, (TEXT.format("RIGHT DOWN"),))),
    ("first take the long way round along the top edge and then come back down: 1,4",
     ((1, 4), (TEXT.format("first take the long way round along the top edge and then co"),))),
])
def test_version_2_reads_plan_tokens_as_listed(block, expected):
    result = parse_plan((block,))
    assert (result.actions, result.notices) == expected


def test_the_loop_reads_each_plan_from_its_last_opener_and_quotes_its_ignored_text():
    answers = ["Use <plan> for (2,0)\n<plan>4,4</plan>", "<plan>D,R,4</plan>"]
    result, calls = play(answers)
    assert [s.action for s in result.steps] == [4, 4, 4]
    assert "- " + TEXT.format("D R") in calls[2][1].split("\n")


def table_of(*transitions):
    table = TransitionTable()
    for before, action, after in transitions:
        table.record(before, action, after)
    return table


MOVED_1 = table_of(("a", 1, "b"), ("b", 1, "c"), ("c", 1, "d"))       # ACTION1 moved from three states
STAYED_2 = table_of(("a", 2, "a"), ("b", 2, "b"), ("c", 2, "c"))      # ACTION2 stayed in three states


@pytest.mark.parametrize("text, table, directions, kind", [
    ("ACTION1 moves the cursor up", MOVED_1, {1: "up"}, "confirmed"),
    ("ACTION1 = UP (row-1), confirmed at (5,1)", MOVED_1, {1: "up"}, "confirmed"),
    ("ACTION1 always moves", MOVED_1, {}, "confirmed"),
    ("ACTION2 does nothing", STAYED_2, {}, "confirmed"),
    ("ACTION1 moves the cursor down", MOVED_1, {1: "up"}, "inconclusive"),
    ("ACTION1 moves the cursor up", MOVED_1, {}, "inconclusive"),
    ("ACTION1 always moves the cursor up", MOVED_1, {1: "down"}, "inconclusive"),
    ("ACTION1 moves up, never down", MOVED_1, {1: "up"}, "inconclusive"),
    ("ACTION1 does not move up", MOVED_1, {1: "up"}, "inconclusive"),
    ("ACTION1 probably moves up", MOVED_1, {1: "up"}, "inconclusive"),
    ("ACTION1 = UP, ACTION5 = CONFIRM", MOVED_1, {1: "up"}, "inconclusive"),
    ("ACTION1 wins the level once the path touches the end", MOVED_1, {1: "up"}, "inconclusive"),
])
def test_version_2_confirms_only_the_claims_its_check_tests(text, table, directions, kind):
    check = verify_rule(Rule(text), table, directions)
    assert (check.kind, check.confidence if kind == "inconclusive" else None) == (
        kind, 0.4 if kind == "inconclusive" else None)


def rule_lines(user):
    return [line.strip() for line in user.split("\n") if line.startswith("  [")]


def test_version_2_ticks_neither_a_wrong_direction_nor_a_win_condition():
    """A scripted episode on tw01 level 1, where ACTION1 moves up."""
    answers = ["<plan>4,5,4,4,5,1,1,5,2,1,4,1,1,3</plan>",
               "<add>ACTION1 moves the cursor down by one dot</add>"
               "<add>ACTION5 wins the level as soon as the path touches the end node</add><plan>2</plan>",
               "<add>The path must visit every yellow dot before submitting</add><plan>1</plan>",
               "<add>ACTION1 moves the cursor up one dot</add><plan>2</plan>", "<keep/><plan>1</plan>"]
    rules = ["[1] ACTION1 moves the cursor down by one dot", "[2] ACTION5 wins the level as soon as the path touches "
             "the end node", "[3] The path must visit every yellow dot before submitting",
             "[4] ACTION1 moves the cursor up one dot"]
    _, calls = play(answers)
    assert rule_lines(calls[5][1]) == rules[:3] + [rules[3] + " ✓"]


def test_version_2_counts_a_contradiction_once_and_tags_by_the_transitions_shown():
    """A scripted episode on ft20 level 1: the rule is confirmed by four moves up, then ACTION1 is blocked once at
    the top and never taken again. The rule is ticked in the prompt right after the four moves, and one contradiction
    is shown from the prompt right after the bump on."""
    answers = ["<add>ACTION1 always moves the cursor up one dot</add><plan>1,1,1,1</plan>", "<keep/><plan>3</plan>",
               "<keep/><plan>1</plan>", "<keep/><plan>3</plan>", "<keep/><plan>4</plan>", "<keep/><plan>3</plan>",
               "<keep/><plan>4</plan>", "<keep/><plan>3</plan>"]
    rule = "[1] ACTION1 always moves the cursor up one dot"
    disputed = '  Rule [1]: "ACTION1 always moves the cursor up one dot" (support=2, contradictions={})'
    _, calls = play(answers, game="ft20")
    assert [rule_lines(user) for _, user in calls[1:3]] == [[rule + " ✓"]] * 2
    assert all(rule_lines(user) == [rule + " ⚠DISPUTED"] and disputed.format(1) in user.split("\n")
               for _, user in calls[3:])
