"""The natural-language rules the model keeps about the game while it plays a level.

`RuleStore` holds the rule list of the current level. After every answered reflection it applies the model's
`<delete>` and `<add>` tags and keeps the list within its limits (at most `MAX_NEW_RULES` new rules per reflection,
no two rules equal ignoring case, at most `MAX_RULES` rules). When a prompt is assembled it checks every rule against
the transitions recorded on the level (`RuleStore.check`, `verify_rule`), so the tags shown agree with the
transitions recorded up to that prompt; the check confirms only the claims it tests (an action always moves, does
nothing, or moves in the direction the action tracker saw on this level; a rule naming no action is capped at
confidence 0.4) and counts the contradicting transitions of the current table. When a level is solved, `carry_over`
moves the rules to the next level with reduced confidence.

The next prompt shows notices of what the harness did to the list: every delete it ignores, every add it rewrites
or drops (except an add of 3 characters or fewer, which is dropped silently), every rule it removes, and a summary
of each level change. Confidence and evidence changes show only through the tags of the rendered rule list
(`prompts.rules_text`).
"""
from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Final, Literal

from .history import TransitionTable

MAX_RULES: Final = 15               # the rule list holds at most this many rules; the "(n/15 rules)" denominator
MAX_NEW_RULES: Final = 2            # <add> tags taken per reflection
NEW_RULE_CONFIDENCE: Final = 0.5    # confidence of a rule the model adds
STORE_THRESHOLD: Final = 0.4        # a rule below this confidence at the end of a solved level is not carried

_RULE_ACTIONS: Final = (1, 2, 3, 4, 5)  # the ids n of the ACTIONn a rule can name; other ids are ignored
_SHORTEST_RULE: Final = 4           # characters of the shortest rule text the list holds
_MIN_EVIDENCE: Final = 3            # transitions a check needs before it can confirm or refute a rule
_REFUTE_RATE: Final = 0.3           # share of contradicting transitions that removes a rule
_INCONCLUSIVE_CAP: Final = 0.4      # confidence ceiling of a rule that a check can neither confirm nor refute
_MAX_DIGIT_RUN: Final = 4300        # longest digit run read as a number (Python's default int() digit limit)
_CARRY_FACTOR_VERIFIED: Final = 0.8
_CARRY_FACTOR_OTHER: Final = 0.5
_MIN_CONFIDENCE: Final = 0.2        # lowest confidence a listed rule can have: STORE_THRESHOLD x _CARRY_FACTOR_OTHER
_SUPPORT_SHARE_WEIGHT: Final = 0.7  # weight of the share of confirming evidence in `Rule.with_support`
_SUPPORT_PRIOR_WEIGHT: Final = 0.3  # weight of the previous confidence there (1 - 0.7 is not 0.3 in floating point)
_CONFIRMED_BASE: Final = 0.9        # confidence a confirming check supports before its evidence bonus
_CONFIRMED_BONUS: Final = 0.1       # evidence bonus per _CONFIRMED_BONUS_SPAN counted transitions
_CONFIRMED_BONUS_SPAN: Final = 20
_CONFIRMED_CAP: Final = 0.95        # highest confidence a confirming check supports

# Longest text, in characters, that a notice quotes.
_QUOTE_RULE: Final = 120            # an add R1 drops, a rule R4 or a check removes, a rule dropped at a level change
_QUOTE_DUPLICATE: Final = 80        # a duplicate add
_QUOTE_DELETE: Final = 60           # the content of an unreadable <delete>
_QUOTE_STRIPPED: Final = 30         # an add before and after its leading rule number is removed

RuleSource = Literal["reflection", "verified", "carried"]

# Notices (model-visible). "R1" and "R4" name the hard rules of the system prompt; their numbers restate
# MAX_NEW_RULES and MAX_RULES and change only together with them.
_NOTICE_DELETE_UNREADABLE: Final = ("ignored <delete>{content}</delete>: could not read a rule number "
                                    "(write <delete>3</delete> with the number shown in the list)")
_NOTICE_DELETE_NO_RULE: Final = ("ignored <delete>{n}</delete>: there is no rule [{n}] "
                                 "(the list above numbers rules 1–{count})")
_NOTICE_ADD_STRIPPED: Final = ("stored your rule without its leading number ('{before}…' → '{after}…'): "
                               "the list numbers rules for you")
_NOTICE_DUPLICATE: Final = "ignored duplicate <add>: '{text}' is already in the list"
_NOTICE_R1: Final = "R1 (max 2 new rules per reflection) dropped your add: '{text}'"
_NOTICE_CAP_TIED: Final = ("R4 (max 15 rules) removed the oldest of the {tied} rules tied at the lowest confidence "
                           "({conf:.2f}): '{text}'")
_NOTICE_CAP_SINGLE: Final = "R4 (max 15 rules) removed the lowest-confidence rule ({conf:.2f}): '{text}'"
_REFUTED_REASON: Final = "{contra}/{total} transitions contradict this rule ({rate:.0%})"
_NOTICE_REMOVED: Final = "harness removed rule '{text}' — {reason} (your recorded transitions contradict it)"
_LEVEL_CHANGE_PREFIX: Final = "Level change: "
_LEVEL_CHANGE_CARRIED: Final = ("{carried} of your rules were carried over from the previous level "
                                "(tagged [carried]; confidence reduced — verify them here)")
_LEVEL_CHANGE_LAYOUT: Final = "{n} rule(s) naming specific coordinates were dropped as level-specific: {quotes}"
_LEVEL_CHANGE_LOW: Final = "{n} low-confidence rule(s) were not carried"

# Reading <delete> and <add> contents.
_DELETE_CLEAN = re.compile(r"(?i)rules?[_\s#]*|\br(?=[_\s#]*\d)|[_#\[\]():.]|\band\b|[,;]")
_DIGIT_RUN = re.compile(r"\d+")
_DELETE_SHAPE = re.compile(r"[\d\s]+")
_ADD_NUMBER_PREFIX = re.compile(
    r"^\s*(?:rule\s*\[?\d+\]?\s*[:.)\-—]|\[\d+\]\s*[:.)\-—]?|\d+\s*(?:[:)]|\.(?!\d)|[-—](?=\s)))\s*",
    re.IGNORECASE)

# Rules naming board coordinates, which are not carried to the next level.
_LAYOUT_PATTERNS = (re.compile(r"\(\d+\s*,\s*\d+\)", re.IGNORECASE),
                    re.compile(r"row\s+\d+.*col\s+\d+", re.IGNORECASE),
                    re.compile(r"position\s+\d+", re.IGNORECASE),
                    re.compile(r"cell\s+\(\d+", re.IGNORECASE),
                    re.compile(r"at\s+\d+\s*,\s*\d+", re.IGNORECASE))

# Claim classifiers of the rule check; all but the first apply to the lower-cased rule text.
_ACTION_REF = re.compile(r"ACTION\s*(\d+)", re.IGNORECASE)
_HEDGE = re.compile(
    r"unknown whether|unclear whether|not sure|whether or not|might be|may be|\?|noop\s*=\s*(false|0|no)\b"
    r"|\b(?:probably|perhaps|possibly|apparently|seemingly|seems?|seemed|appears?|appeared|i think|i guess"
    r"|i believe|i suspect|maybe)\b")
_VACUOUS_TAIL = re.compile(
    r"\b(?:when|once|if|whenever|after|because)\s+(?:it is\s+|it's\s+)?(?:pressed|used|invoked|called|applied"
    r"|nothing changed|nothing changes|nothing happens)(?=\s*(?:[.,;:!)]|$))")
_CONDITION = re.compile(
    r"\b(?:unless|until|when|whenever|if|while|once|after|before|except|only|because|so far|has been|likely|with"
    r"|without|given)\b|\b(?:edge|edges|wall|walls|boundar\w*|border|borders|blocked|blocking|obstacle\w*|top"
    r"|bottom|upper|lower|first|last|start|end|middle|row|rows|column|columns|col \d+|off[- ]board|off[- ]grid"
    r"|out of bounds|corner\w*)\b")
_NEGATED_MOVE_WORDS = re.compile(
    r"(?:does not|doesn't|never|cannot|can't|did not|didn't|no)\s+(?:move|moves|movement)\w*")
_NEGATION = re.compile(r"\bnot\b|n't\b")
# Direction words of a direction claim, by the direction they name.
_DIRECTION_WORDS: Final = MappingProxyType({
    word: direction for direction, words in (("up", ("up", "upward", "upwards", "north")),
                                             ("down", ("down", "downward", "downwards", "south")),
                                             ("left", ("left", "leftward", "leftwards", "west")),
                                             ("right", ("right", "rightward", "rightwards", "east")))
    for word in words})
_DIRECTION_RE = re.compile(r"\b(" + "|".join(_DIRECTION_WORDS) + r")\b")
_NOOP_ANCHORED_TEMPLATE: Final = (
    r"action\s*<A>\b[^.;,]{0,25}?(?:(?:is|=|:|remains|acts as)\s+(?:a\s+|an\s+|consistently\s+a\s+|likely\s+a\s+"
    r"|always\s+a\s+|the\s+)?)?(?:noop|no-op|nop|no effect|has no effect|does nothing|no change|no_change)")
_NOOP_NEGATED_MOVE_TEMPLATE: Final = (
    r"action\s*<A>\b[^.;,]{0,30}?(?:does not move|doesn't move|never moves|cannot move|can't move|no movement"
    r"|did not move|didn't move)")
_NOOP_ANCHORED = MappingProxyType({a: re.compile(_NOOP_ANCHORED_TEMPLATE.replace("<A>", str(a)))
                                   for a in _RULE_ACTIONS})
_NOOP_NEGATED_MOVE = MappingProxyType({a: re.compile(_NOOP_NEGATED_MOVE_TEMPLATE.replace("<A>", str(a)))
                                       for a in _RULE_ACTIONS})
_MOVE_WORDS = ("moves", "move the", "moves the", "always move", "causes movement", "direction")
_UNIVERSAL_WORDS = ("always", "every time", "never fails", "never blocked", "in all cases", "unconditionally",
                    "regardless of")
_POSITIVE_MOVE_WORDS = _MOVE_WORDS + ("moving", " up", " down", " left", " right", "=up", "=down", "=left", "=right",
                                     "north", "south", "east", "west", "upward", "downward", "leftward", "rightward",
                                     "toward", "row index", "column index", "row-1", "row+1", "col-1", "col+1",
                                     "one cell", "one dot", "one step")


@dataclass(frozen=True)
class Rule:
    """One rule of the list.

    `confidence` is the harness's confidence in the rule. `evidence_for` counts the checks on this level that
    confirmed the rule. `evidence_against` counts the recorded transitions that contradicted it at the latest check:
    a check that finds contradicting transitions but keeps the rule sets it to their number when `evidence_for` is
    positive or when there are at least two of them, and to 0 otherwise. Both counts restart at 0 when the rule is
    carried to the next level.

    `source` says where the rule stands: "reflection" (written by the model), "verified" (written by the model and
    confirmed by the recorded transitions) or "carried" (kept from the previous level).
    """

    text: str
    confidence: float = NEW_RULE_CONFIDENCE
    evidence_for: int = 0
    evidence_against: int = 0
    source: RuleSource = "reflection"

    def with_support(self) -> Rule:
        """The rule after one more confirming check: `evidence_for` grows by one and the confidence becomes
        0.7 x (share of confirming evidence) + 0.3 x (previous confidence), kept within [0, 1]."""
        evidence_for = self.evidence_for + 1
        total = evidence_for + self.evidence_against
        confidence = min(1.0, max(0.0, _SUPPORT_SHARE_WEIGHT * (evidence_for / total)
                                     + _SUPPORT_PRIOR_WEIGHT * self.confidence))
        return replace(self, evidence_for=evidence_for, confidence=confidence)


@dataclass(frozen=True)
class RuleCheck:
    """The result of checking one rule against the transitions of the level.

    `kind` is "confirmed" (no counted transition contradicts the rule), "refuted" (at least 30% of them do; `reason`
    says how many) or "inconclusive" (fewer than three transitions of the named actions, fewer than 30%
    contradictions, a rule that names no action, or a claim the check does not test). `confidence` is
    the confidence the check supports and `contradictions` the number of contradicting transitions.
    """

    kind: Literal["confirmed", "refuted", "inconclusive"]
    confidence: float
    contradictions: int = 0
    reason: str = ""


def verify_rule(rule: Rule, table: TransitionTable, directions: Mapping[int, str]) -> RuleCheck:
    """Check `rule` against the latest outcome of each action from each recorded state and `directions` (action id
    to the direction the action tracker saw on this level).

    The check counts the recorded outcomes of every action ACTION1-ACTION5 the rule names. A rule claiming that an
    action always moves is contradicted by each state where it stayed; a rule claiming that an action does nothing
    is contradicted by each state where it moved; hedged and conditional claims are not counted as either. A
    confirmed rule gets 0.9 + 0.005 per counted transition (at most 0.95); a rule with contradictions below 30%
    gets 1 - 2 x (their share); a rule without enough evidence, or naming no action, gets at most 0.4.

    The check confirms only what it tests. Each action the rule names must be claimed to always move, to do
    nothing, or to move in one direction (`_claimed_directions`) that is its direction in `directions`; and a
    direction claimed for an action must be that one. Otherwise the rule gets at most 0.4, as a rule naming no
    action does (contradictions still refute it).
    """
    actions = _action_ids(rule.text)
    if not actions:
        return RuleCheck("inconclusive", min(_INCONCLUSIVE_CAP, rule.confidence))
    text = rule.text.lower()
    claimed = _claimed_directions(text)
    total = contra = 0
    unchecked = False
    for action in actions:
        moved, stayed = table.outcome_counts(action)
        total += moved + stayed
        tested = False
        if _claims_always_moves(text, action):
            contra += stayed
            tested = True
        elif _claims_noop(text, action):
            contra += moved
            tested = True
        if action in claimed:
            tested = claimed[action] is not None and claimed[action] == directions.get(action)
        unchecked = unchecked or not tested
    if total < _MIN_EVIDENCE:
        return RuleCheck("inconclusive", min(_INCONCLUSIVE_CAP, rule.confidence), contra)
    if contra > 0:
        rate = contra / total
        if rate >= _REFUTE_RATE:
            return RuleCheck("refuted", 0.0, contra, _REFUTED_REASON.format(contra=contra, total=total, rate=rate))
        return RuleCheck("inconclusive", min(1.0 - rate * 2, _INCONCLUSIVE_CAP) if unchecked else 1.0 - rate * 2,
                         contra)
    if unchecked:
        return RuleCheck("inconclusive", min(_INCONCLUSIVE_CAP, rule.confidence))
    bonus = (total / _CONFIRMED_BONUS_SPAN) * _CONFIRMED_BONUS
    return RuleCheck("confirmed", min(_CONFIRMED_CAP, _CONFIRMED_BASE + bonus))


class RuleStore:
    """The rules of the current level in list order; a rule's 1-based position is its number in <delete>."""

    def __init__(self, rules: Sequence[Rule] = ()) -> None:
        """Start from `rules` (empty for a new episode). Raises ValueError unless the rules could be the list of
        a running episode: at most `MAX_RULES` rules, confidences in [0.2, 1.0], non-negative evidence counts,
        stripped texts longer than 3 characters, no evidence on a rule naming no action, and no two texts equal
        ignoring case."""
        self._rules = list(rules)
        _check_starting_rules(self._rules)

    @property
    def rules(self) -> tuple[Rule, ...]:
        """The current list."""
        return tuple(self._rules)

    def apply_reflection(self, deletes: Sequence[str], adds: Sequence[str]) -> list[str]:
        """Apply one answered reflection and return its notices in order.

        `deletes` and `adds` are the stripped contents of the response's `<delete>` and `<add>` tags (both empty
        for a response without tags). In order: the deletes remove the rules they number (numbers refer to the
        list before this reflection); every add loses a leading rule number and is dropped if 3 characters or
        shorter; the first `MAX_NEW_RULES` adds are appended unless the list already holds the same text ignoring
        case (a duplicate still uses up a place), the rest are dropped; the list is cut to the `MAX_RULES` most
        confident rules (the newest win ties). The rules are checked when the next prompt is assembled (`check`).
        """
        notices: list[str] = []
        rules = _delete(self._rules, deletes, notices)
        candidates = _add_candidates(adds, notices)
        for text in candidates[:MAX_NEW_RULES]:
            if any(text.lower() == r.text.lower() for r in rules):
                notices.append(_NOTICE_DUPLICATE.format(text=text[:_QUOTE_DUPLICATE]))
            else:
                rules.append(Rule(text))
        notices.extend(_NOTICE_R1.format(text=text[:_QUOTE_RULE]) for text in candidates[MAX_NEW_RULES:])
        self._rules = _cap(rules, notices)
        return notices

    def check(self, table: TransitionTable, directions: Mapping[int, str]) -> list[str]:
        """Check every rule against `table` and the directions the action tracker saw on this level (`verify_rule`)
        and return the notices of removed rules, in list order. Refuted rules are removed; inconclusive ones are
        capped at the check's confidence and count its contradictions as described for `Rule.evidence_against`;
        confirmed ones lose their contradictions, are raised to the check's confidence and then gain one support
        (`Rule.with_support`); a model-written confirmed rule becomes "verified"."""
        notices: list[str] = []
        self._rules = _check_all(self._rules, table, notices, directions)
        return notices

    def carry_over(self) -> str | None:
        """Move the rules of a solved level to the next level; return its "Level change" notice, or None when
        the list was empty.

        Every rule first gains one support (`Rule.with_support`). Rules below `STORE_THRESHOLD` and rules naming
        board coordinates are dropped; the others restart as "carried" rules without evidence, at 0.8 x their
        confidence if they were verified on this level and 0.5 x otherwise. The notice quotes each rule dropped
        for naming coordinates (at most 120 characters, in list order).
        """
        if not self._rules:
            return None
        supported = [rule.with_support() for rule in self._rules]
        stored = [rule for rule in supported if rule.confidence >= STORE_THRESHOLD]
        low = len(supported) - len(stored)
        carried: list[Rule] = []
        layout: list[str] = []
        for rule in stored:
            if any(p.search(rule.text) for p in _LAYOUT_PATTERNS):
                layout.append(rule.text)
                continue
            factor = _CARRY_FACTOR_VERIFIED if rule.source == "verified" else _CARRY_FACTOR_OTHER
            carried.append(Rule(rule.text, rule.confidence * factor, source="carried"))
        self._rules = carried
        parts = [_LEVEL_CHANGE_CARRIED.format(carried=len(carried))]
        if layout:
            quotes = ", ".join(f"'{text[:_QUOTE_RULE]}'" for text in layout)
            parts.append(_LEVEL_CHANGE_LAYOUT.format(n=len(layout), quotes=quotes))
        if low:
            parts.append(_LEVEL_CHANGE_LOW.format(n=low))
        return _LEVEL_CHANGE_PREFIX + "; ".join(parts) + "."


# -------------------------------------------------------------------------------------------- list operations
def _check_starting_rules(rules: Sequence[Rule]) -> None:
    """Raise ValueError at the first property of `RuleStore`'s starting list that `rules` lacks."""
    if len(rules) > MAX_RULES:
        raise ValueError(f"at most {MAX_RULES} rules, got {len(rules)}")
    for rule in rules:
        if not _MIN_CONFIDENCE <= rule.confidence <= 1.0 or rule.evidence_for < 0 or rule.evidence_against < 0:
            raise ValueError(f"rule confidence must be in [{_MIN_CONFIDENCE}, 1.0] and evidence counts >= 0: {rule!r}")
        if rule.text != rule.text.strip() or len(rule.text) < _SHORTEST_RULE:
            raise ValueError(f"rule text must be stripped and longer than 3 characters: {rule.text!r}")
        if not _action_ids(rule.text) and (rule.evidence_for or rule.evidence_against):
            raise ValueError(f"a rule naming no action ACTION1-ACTION5 must have no evidence: {rule!r}")
    seen: set[str] = set()
    for rule in rules:
        if rule.text.lower() in seen:
            raise ValueError(f"duplicate rule text (ignoring case): {rule.text!r}")
        seen.add(rule.text.lower())


def _delete(rules: Sequence[Rule], deletes: Sequence[str], notices: list[str]) -> list[Rule]:
    """The rules without the ones the deletes number; a tag without a readable number, and every number that names
    no rule, gets a notice (a digit run longer than `_MAX_DIGIT_RUN` names no rule)."""
    count = len(rules)
    doomed: set[int] = set()
    for content in deletes:
        cleaned = _DELETE_CLEAN.sub(" ", content)
        numbers = _DIGIT_RUN.findall(cleaned)
        if not numbers or not _DELETE_SHAPE.fullmatch(cleaned.strip()):
            notices.append(_NOTICE_DELETE_UNREADABLE.format(content=content[:_QUOTE_DELETE]))
            continue
        for n in numbers:
            value = _number_at_most(n, count)
            if value is not None and value >= 1:
                doomed.add(value - 1)
            else:
                notices.append(_NOTICE_DELETE_NO_RULE.format(n=n, count=count))
    return [rule for i, rule in enumerate(rules) if i not in doomed]


def _add_candidates(adds: Sequence[str], notices: list[str]) -> list[str]:
    """The add texts without a leading rule number (with a notice when one was removed), longer than 3
    characters."""
    candidates: list[str] = []
    for text in adds:
        stripped = _ADD_NUMBER_PREFIX.sub("", text).strip()
        if stripped != text:
            notices.append(_NOTICE_ADD_STRIPPED.format(before=text[:_QUOTE_STRIPPED], after=stripped[:_QUOTE_STRIPPED]))
        if len(stripped) >= _SHORTEST_RULE:
            candidates.append(stripped)
    return candidates


def _cap(rules: list[Rule], notices: list[str]) -> list[Rule]:
    """The `MAX_RULES` most confident rules in list order; among equal confidences the newest stay. Each removed
    rule gets a notice, in list order."""
    if len(rules) <= MAX_RULES:
        return rules
    order = sorted(range(len(rules)), key=lambda i: (-rules[i].confidence, -i))
    evicted = set(order[MAX_RULES:])
    for i in sorted(evicted):
        rule = rules[i]
        tied = sum(1 for other in rules if abs(other.confidence - rule.confidence) < 1e-9)
        template = _NOTICE_CAP_TIED if tied > 1 else _NOTICE_CAP_SINGLE
        notices.append(template.format(tied=tied, conf=rule.confidence, text=rule.text[:_QUOTE_RULE]))
    return [rule for i, rule in enumerate(rules) if i not in evicted]


def _check_all(rules: Sequence[Rule], table: TransitionTable, notices: list[str],
               directions: Mapping[int, str]) -> list[Rule]:
    """Every rule after its check against `table`, in list order; refuted rules are removed with a notice. A kept
    rule's `evidence_against` is the count of this check (0 when it does not count, and for a confirmed rule), not
    a running sum."""
    checked: list[Rule] = []
    for rule in rules:
        check = verify_rule(rule, table, directions)
        if check.kind == "refuted":
            notices.append(_NOTICE_REMOVED.format(text=rule.text[:_QUOTE_RULE], reason=check.reason))
            continue
        if check.kind == "inconclusive":
            counted = check.contradictions > 0 and (rule.evidence_for > 0 or check.contradictions >= 2)
            rule = replace(rule, confidence=min(rule.confidence, check.confidence),
                           evidence_against=check.contradictions if counted else 0)
        else:
            rule = replace(rule, evidence_against=0, confidence=max(rule.confidence, check.confidence)).with_support()
            rule = replace(rule, source="verified" if rule.source == "reflection" else rule.source)
        checked.append(rule)
    return checked


# ------------------------------------------------------------------------------------------ claim classifiers
def _action_ids(text: str) -> list[int]:
    """The action ids 1-5 that `text` names as ACTIONn, ascending (a digit run longer than `_MAX_DIGIT_RUN` names
    none)."""
    values = (_number_at_most(m, max(_RULE_ACTIONS)) for m in _ACTION_REF.findall(text))
    return sorted({v for v in values if v in _RULE_ACTIONS})


def _number_at_most(digits: str, high: int) -> int | None:
    """The value of the digit run `digits` if it is at most `high`, else None; a run longer than `_MAX_DIGIT_RUN`
    is None whatever its value. int() reads the run only after its leading zeros are dropped and only when no
    more digits than those of `high` remain, so the interpreter's integer-string digit limit (which can be set
    below 4300) never applies."""
    if len(digits) > _MAX_DIGIT_RUN:
        return None
    significant = "".join(str(int(ch)) for ch in digits).lstrip("0")   # ASCII form, as int() reads any script
    if len(significant) > len(str(high)):
        return None
    value = int(significant or "0")
    return value if value <= high else None


def _claims_always_moves(text: str, action: int) -> bool:
    """`text` claims, without hedge or condition, that `action` always moves."""
    if f"action{action}" not in text.replace(" ", ""):
        return False
    if _hedged(text) or _conditional(text):
        return False
    return any(w in text for w in _MOVE_WORDS) and any(w in text for w in _UNIVERSAL_WORDS)


def _claims_noop(text: str, action: int) -> bool:
    """`text` claims, without hedge, that `action` does nothing: an unnegated, unconditional no-op phrase after the
    action, or an unconditional "does not move" after it, in a text that claims no movement elsewhere."""
    if f"action{action}" not in text.replace(" ", ""):
        return False
    if _positive_move_claim(text) or _hedged(text):
        return False
    conditional = _conditional(text)
    anchored = not _NEGATION.search(text) and not conditional and _NOOP_ANCHORED[action].search(text)
    negated_move = not conditional and _NOOP_NEGATED_MOVE[action].search(text)
    return bool(anchored or negated_move)


def _hedged(text: str) -> bool:
    return _HEDGE.search(text) is not None


def _conditional(text: str) -> bool:
    """A condition word or a board-position word, ignoring a clause-final "when pressed" and the like."""
    return _CONDITION.search(_VACUOUS_TAIL.sub(" ", text)) is not None


def _positive_move_claim(text: str) -> bool:
    """A movement word outside a negated movement phrase."""
    unnegated = _NEGATED_MOVE_WORDS.sub(" ", text)
    return any(w in unnegated for w in _POSITIVE_MOVE_WORDS)


def _claimed_directions(text: str) -> dict[int, str | None]:
    """For each action ACTION1-ACTION5 that `text` (lower case) names with a direction word between one
    of its mentions and the next action mention, the direction those words name; None when they name more than
    one, a negation stands between such a mention and the next, or the text is hedged."""
    mentions = [(m, _number_at_most(m.group(1), max(_RULE_ACTIONS))) for m in _ACTION_REF.finditer(text)]
    found: dict[int, set[str]] = {}
    negated: set[int] = set()
    for i, (mention, action) in enumerate(mentions):
        if action not in _RULE_ACTIONS:
            continue
        segment = text[mention.end():mentions[i + 1][0].start() if i + 1 < len(mentions) else len(text)]
        named = {_DIRECTION_WORDS[w] for w in _DIRECTION_RE.findall(segment)}
        if named:
            found.setdefault(action, set()).update(named)
            if _NEGATION.search(segment):
                negated.add(action)
    hedged = _hedged(text)
    return {action: next(iter(named)) if len(named) == 1 and action not in negated and not hedged else None
            for action, named in found.items()}
