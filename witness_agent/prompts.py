"""The two messages of a reflection call: the system prompt and the user message.

Every reflection call sends exactly these two messages, both rebuilt from the harness state; no conversation
history is sent. The system prompt holds fixed instructions, the current action semantics and three notes (on
CONFIRM, RESET and the plan cap). The user message holds the board, the action semantics, the recent
observations, the rule list, the harness notices, the recent attempts and the instructions.

The static texts are fixed strings: the numbers in them (the plan cap 16, the rule cap 15) are part of the text,
not substituted. Every function is pure.
"""
from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Final

from .actions import ActionSemantics, MoveSemantics, confirm_phrase, usage_phrase
from .knowledge import MAX_RULES, Rule

MAX_NOTICES_SHOWN: Final = 12       # notices listed in one user message; the rest are counted in one line
_CONFIDENT: Final = 0.8             # confidence from which a rule without counter-evidence gets the check-mark tag

# A step line of the observation window ("  #k ACTION..."), except the level-entry line "#0 ".
_STEP_LINE_RE: Final = re.compile(r"^\s*#(?!0 )\d+ ", re.MULTILINE)

# The fixed instructions at the start of the system prompt.
STATIC_HEAD: Final = (
    "You are an interactive reasoning agent playing a puzzle video game. Each turn you receive a TEXT observation: "
    "an ASCII board with a legend and, on games that expose them, ground-truth lists of the board's markers and cell "
    "content, plus your recent actions with their effects and your own knowledge rules. Your goal: discover the "
    "rules and solve each level efficiently.\n"
    "\n"
    "## Core Priors\n"
    "\n"
    "P1 OBJECTNESS: The environment has discrete bounded objects (the board elements named in the legend and, when "
    "present, the cell-content list). Objects persist between steps unless explicitly changed. They have: color, "
    "position, size, shape. Objects can appear, disappear, or change — track these transitions.\n"
    "\n"
    "P2 AGENTNESS: You control something in the environment. Your actions (ACTION1-ACTION5) cause state changes. You "
    "must discover WHAT each action does by observing before/after. Goal: reach a winning state (the environment "
    "will signal level completion).\n"
    "\n"
    "P3 NUMBERS: Objects can be counted. Quantities may be constraints (exactly N, at least N, at most N). "
    "Arithmetic relationships between quantities may matter.\n"
    "\n"
    "P4 GEOMETRY: Spatial relationships matter — adjacency, containment, alignment, symmetry "
    "(horizontal/vertical/rotational). Distances can be Manhattan or Euclidean. Boundaries may block, wrap, or "
    "reflect. Coordinate system: (row, col), 0-indexed, (0,0) at TOP-LEFT — the same order as every coordinate in "
    "the observation. On a dot board the cursor moves on dot intersections; cell (r,c) is the square between dots "
    "(r,c), (r,c+1), (r+1,c) and (r+1,c+1).\n"
    "\n"
    "## Hypothesis-Test Protocol\n"
    "\n"
    "Each turn:\n"
    "1. OBSERVE: Read the current state carefully\n"
    "2. HYPOTHESIZE: What rule explains the observations so far?\n"
    "3. PREDICT: What will my chosen action cause?\n"
    "4. ACT: Choose the most informative action\n"
    "5. VERIFY: Was prediction correct? Update hypothesis confidence.\n"
    "\n"
    "## Exploration Strategy\n"
    "\n"
    "- Simple first: try basic actions before complex strategies\n"
    "- Systematic: test each available action from new states\n"
    "- Contrastive: different actions in the same state reveal what each action does\n"
    "- Errors are signals: negative feedback (visual changes, state resets) = direct evidence of constraint "
    "boundaries\n"
    "- Hypothesis-driven: once you have a hypothesis, test it efficiently\n"
    "- AVOID CYCLES: if you've been to a state before, try a DIFFERENT action\n"
    "- Map actions early: in the first few steps, try each of ACTION1-4 to learn what they do\n"
    "\n"
    "## Reasoning Principles\n"
    "\n"
    "- Occam's Razor: prefer the simplest hypothesis that explains all observations\n"
    "- One rule at a time: test hypotheses individually, not in combination\n"
    "- Distinguish mechanics from goals: HOW things work vs WHAT you need to achieve\n"
    "- Failed attempts are the most informative: they reveal constraint boundaries\n"
    "- Look for patterns across levels: rules often stay the same, only configurations change\n"
    "\n"
    "## HARD RULES — Mandatory Constraints\n"
    "\n"
    "These rules are ABSOLUTE. Violating any of them wastes budget and degrades performance.\n"
    "\n"
    "### Rule Discovery\n"
    "- R1: Output AT MOST 2 new rules per reflection. Quality over quantity.\n"
    "- R2: DELETE any rule where counter-evidence >= supporting evidence. Do not keep disputed rules alive.\n"
    "- R3: A rule marked ⚠DISPUTED has conflicting evidence — test it or delete it; do not plan on it.\n"
    "- R4: Maintain AT MOST 15 active rules. If at capacity, delete the weakest before adding.\n"
    "- R5: NEVER create a rule from a single observation. Require at least 2 supporting observations.\n"
    '- R6: Correlation is NOT causation. "X happened after Y" does not mean Y caused X — verify with a deliberate '
    "test.\n"
    '- R7: If you are not sure a rule holds, say so in its wording (e.g. "probably").\n'
    "\n"
    "### Planning\n"
    "- P1: Plans must be AT MOST 16 actions. Longer plans accumulate prediction errors.\n"
    "- P2: Each action in a plan MUST have a justification grounded in a known rule.\n"
    "- P3: Every action of a plan is executed before you see any result; when you are unsure what an action does, "
    "plan fewer actions.\n"
    "\n"
    "### Global\n"
    "- G1: The board shows the actions used on this level (the level ends the episode when they run out) and the "
    "reflection calls used; when little is left, spend it on your best complete attempt.\n"
    "- G2: If 10 consecutive actions produce no state change, SWITCH strategy (different actions, different "
    "region).\n"
    "- G3: When the same (state, action) produces different results, this is COGNITIVE DISSONANCE — resolve the "
    "contradiction in your next reflection before planning further."
)

# The notes that close the system prompt: CONFIRM, RESET and the plan cap.
CONFIRM_NOTE: Final = (
    "ACTION5 is the CONFIRM/SUBMIT action: it does not move anything — press it to submit your COMPLETED solution "
    "and win the level. Plan it as the FINAL action, and only once your path satisfies every rule (confirming an "
    "incomplete/incorrect path fails the attempt)."
)

RESET_NOTE: Final = (
    "ACTION0 is RESET: it restarts the CURRENT level from scratch. Your drawn path/progress on the board is erased, "
    "but everything you have LEARNED (rules, action semantics) is kept. IMPORTANT: all actions you have already "
    "taken on this level still count toward your step budget and efficiency score — resetting does not refund them. "
    "Use it deliberately, when your current path is unsalvageable and restarting is cheaper than undoing it move by "
    "move."
)

PLAN_CAP_NOTE: Final = (
    "PLAN LENGTH: at most 16 actions per plan are executed; anything beyond the first 16 is dropped. Plan in chunks "
    "of at most 16 actions and continue from the resulting state."
)

# The user message; its ten fields are filled by `user_message`.
USER_TEMPLATE: Final = (
    "You are analyzing an unknown game environment to discover its rules and mechanics.\n"
    "You must figure out everything from observation — no prior knowledge of this game exists.\n"
    "\n"
    "## Current Board State\n"
    "{board_state}\n"
    "\n"
    "## Known Action Semantics\n"
    "{action_semantics}\n"
    "\n"
    "## Recent Observations (last {n_obs} steps)\n"
    "{observations}\n"
    "\n"
    "## Current Knowledge ({n_rules}/{max_rules} rules)\n"
    "{knowledge_rules}\n"
    "{harness_notices}{attempts_section}\n"
    "## Instructions\n"
    "Analyze the recent observations in context of the board state and existing knowledge.\n"
    "Look for:\n"
    "- How actions affect the game state (movement patterns, special behaviors)\n"
    "- Spatial patterns (board structure, special positions, boundaries)\n"
    "- Win condition clues (what might lead to level completion)\n"
    "- Effective strategies (sequences that make progress toward goals)\n"
    "{dissonance_warning}\n"
    "Respond with:\n"
    "1. <meta>Your concise analysis</meta>\n"
    "2. Memory operations (one or more of these):\n"
    "   <add>new rule or pattern you discovered</add>\n"
    "   <delete>rule_number</delete>  (to remove an outdated/wrong rule)\n"
    "   <keep/>  (if no changes to knowledge needed)\n"
    "3. Optionally, if you have a specific strategy to try right now (an answer without a plan is followed by one "
    "random move):\n"
    "   Use <plan>...</plan> with action IDs as comma-separated numbers.\n"
    "\n"
    "Reminders (full constraints in system prompt — R1-R7, P1-P3, G1-G3):\n"
    "- Update knowledge via delete+add (don't duplicate existing rules); keep rules one sentence each.\n"
    "- DISPUTED rules (⚠) have conflicting evidence — verify before relying."
)


def kas_text(semantics: ActionSemantics) -> str:
    """The "Known Action Semantics" text: one line per known move action in ascending order, then the CONFIRM
    line; lines joined by newlines, no trailing newline.

    A move is known unless it is untested. A carried move (direction learned on an earlier level, not yet used
    here) shows its direction; a move used on this level shows its direction when one is known, its no-op rate
    as a whole percentage, how often it was used and, when known, the action that undoes it. The CONFIRM line
    counts the CONFIRMs pressed on this level, every one of which was rejected.
    """
    lines = [_move_line(move) for move in semantics.moves if move.status != "untested"]
    n = semantics.confirm_presses
    lines.append(f"  ACTION5: CONFIRM/SUBMIT (given), {confirm_phrase(n)}")
    return "\n".join(lines)


def _move_line(move: MoveSemantics) -> str:
    parts = [] if move.direction is None else [f"direction={move.direction}"]
    if move.status == "carried":
        parts.append(usage_phrase(move))
    else:
        parts += [f"noop={format(move.noop_rate, '.0%')}", usage_phrase(move)]
        if move.inverse is not None:
            parts.append(f"inv=ACTION{move.inverse}")
    return f"  ACTION{move.action}: " + ", ".join(parts)


def system_prompt(kas: str) -> str:
    """The system prompt: the fixed instructions, the "Known Action Semantics" section holding `kas`, then the
    CONFIRM, RESET and plan-cap notes, separated by blank lines. It changes whenever `kas` changes."""
    return (STATIC_HEAD + "\n\n## Known Action Semantics\n" + kas
            + "\n\n" + CONFIRM_NOTE + "\n\n" + RESET_NOTE + "\n\n" + PLAN_CAP_NOTE)


def rules_text(rules: Sequence[Rule]) -> str:
    """The rule list: one line `  [i] text` per rule, numbered from 1 in list order (the numbers the model uses
    in `<delete>`), or "No rules discovered yet." when there is none.

    The text is printed as stored. It is followed by " ⚠DISPUTED" when the rule has counter-evidence, otherwise
    by " ✓" when its confidence is at least 0.8, and then by " [carried]" when it was kept from the previous
    level.
    """
    if not rules:
        return "No rules discovered yet."
    lines = []
    for i, rule in enumerate(rules, 1):
        tag = " ⚠DISPUTED" if rule.evidence_against > 0 else (" ✓" if rule.confidence >= _CONFIDENT else "")
        if rule.source == "carried":
            tag += " [carried]"
        lines.append(f"  [{i}] {rule.text}{tag}")
    return "\n".join(lines)


def dissonance_warning(rules: Sequence[Rule]) -> str:
    """The warning about rules with counter-evidence: "" when there is none; otherwise a newline, a warning line
    and one line per such rule with its number, text and evidence counts (no trailing newline)."""
    disputed = [(i, rule) for i, rule in enumerate(rules, 1) if rule.evidence_against > 0]
    if not disputed:
        return ""
    lines = ["\n⚠ ATTENTION — The following rules have CONFLICTING evidence and may be wrong. "
             "Review carefully and consider deleting:"]
    lines += [f"  Rule [{i}]: \"{rule.text}\" (support={rule.evidence_for}, contradictions={rule.evidence_against})"
              for i, rule in disputed]
    return "\n".join(lines)


def notices_block(notices: Sequence[str]) -> str:
    """The "Harness notices" section: "" when there is no notice; otherwise a newline, the heading, one line
    `- notice` for each of the first `MAX_NOTICES_SHOWN` notices (printed as given), a line counting the rest if
    there are more, and a final newline."""
    if not notices:
        return ""
    shown = notices[:MAX_NOTICES_SHOWN]
    more = len(notices) - len(shown)
    return ("\n## Harness notices (since your last reflection)\n"
            + "\n".join(f"- {notice}" for notice in shown)
            + (f"\n- (+{more} more notices not shown)" if more > 0 else "") + "\n")


def user_message(*, board: str, kas: str, showcase: str, window: str, rules: Sequence[Rule],
                 notices: Sequence[str], attempts: str) -> str:
    """The user message of one reflection call.

    `board` is the board text, `kas` the same action-semantics text as in the system prompt, `showcase` the
    action-effects text ("" when there is none) and `window` the recent-actions text, passed separately: the
    observations section is `showcase`, a blank line and `window`, or `window` alone. Its heading counts the
    step lines of the window ("#k ACTION...", not the level-entry line "#0"). `rules` is the rule list before
    this call's answer is applied, `notices` the pending harness notices, and `attempts` the recent-attempts
    section ("" or a text that starts and ends with a newline). Every value is inserted as given.
    """
    observations = f"{showcase}\n\n{window}" if showcase else window
    return USER_TEMPLATE.format(
        board_state=board,
        action_semantics=kas,
        n_obs=len(_STEP_LINE_RE.findall(observations)),
        observations=observations,
        n_rules=len(rules),
        max_rules=MAX_RULES,
        knowledge_rules=rules_text(rules),
        harness_notices=notices_block(notices),
        attempts_section=attempts,
        dissonance_warning=dissonance_warning(rules),
    )
