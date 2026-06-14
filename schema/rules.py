"""The rule schema: what a San Francisco parking sign stack can say, as data.

Design notes:
- A pole holds a stack of Restriction objects (top to bottom matters for rendering,
  not for semantics).
- Answering "can I park here at time T?" applies every restriction independently;
  the most severe applicable verdict wins (TOW > NO_PARK > NO_STOP > TIME_LIMIT > FREE).
- v1 scope: sign-stack-only. Curb paint and meters are out of frame; queries that
  depend on them must be answered ABSTAIN by a correct model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, time
from enum import Enum


class Day(Enum):
    MON, TUE, WED, THU, FRI, SAT, SUN = range(7)


WEEKDAYS = frozenset({Day.MON, Day.TUE, Day.WED, Day.THU, Day.FRI})
EVERY_DAY = frozenset(Day)


class Kind(Enum):
    NO_STOPPING = "no_stopping"            # CA R26(S): no stopping at any time in window
    NO_PARKING = "no_parking"              # CA R26: no parking in window
    TOW_AWAY = "tow_away"                  # modifier or standalone: violation = tow
    TIME_LIMIT = "time_limit"              # CA R30: e.g. 2-hour parking 9am-6pm
    STREET_CLEANING = "street_cleaning"    # CA R32: no parking, specific weekday window
    PERMIT_EXEMPT_LIMIT = "permit_limit"   # RPP: time limit EXCEPT vehicles with area permit
    LOADING_ONLY = "loading_only"          # passenger/commercial loading zone window


@dataclass(frozen=True)
class Window:
    """A recurring weekly time window, e.g. Tue 8:00-10:00."""
    days: frozenset[Day]
    start: time
    end: time
    weeks: frozenset[int] = frozenset()  # which weeks of the month (1-5); empty = every week

    def contains(self, dt: datetime) -> bool:
        if Day(dt.weekday()) not in self.days or not (self.start <= dt.time() < self.end):
            return False
        if self.weeks:  # "2nd & 4th Monday" style: nth occurrence of the weekday in the month
            week_of_month = (dt.day - 1) // 7 + 1
            if week_of_month not in self.weeks:
                return False
        return True


@dataclass(frozen=True)
class Restriction:
    kind: Kind
    window: Window
    limit_minutes: int | None = None    # TIME_LIMIT / PERMIT_EXEMPT_LIMIT
    permit_area: str | None = None      # RPP area letter, e.g. "S"
    tow: bool = False                   # tow-away enforcement on violation


@dataclass
class SignStack:
    """Everything on one pole."""
    restrictions: list[Restriction] = field(default_factory=list)
    pole_id: str | None = None


class Verdict(Enum):
    TOW_RISK = "tow_risk"           # parking now risks a tow
    NO = "no"                       # no parking (citation risk)
    LIMITED = "limited"             # ok up to N minutes
    OK = "ok"                       # no restriction applies right now
    ABSTAIN = "abstain"             # not decidable from the sign stack alone


@dataclass
class Answer:
    verdict: Verdict
    limit_minutes: int | None = None   # for LIMITED
    until: datetime | None = None      # next moment the verdict changes (v2)
    reason: str = ""


SEVERITY = [Verdict.TOW_RISK, Verdict.NO, Verdict.LIMITED, Verdict.OK]


def can_park(stack: SignStack, when: datetime, permit_areas: frozenset[str] = frozenset()) -> Answer:
    """The resolver: ground truth for every generated question."""
    verdicts: list[Answer] = []
    for r in stack.restrictions:
        if not r.window.contains(when):
            continue
        if r.kind in (Kind.NO_STOPPING, Kind.NO_PARKING, Kind.STREET_CLEANING, Kind.LOADING_ONLY):
            v = Verdict.TOW_RISK if (r.tow or r.kind is Kind.NO_STOPPING) else Verdict.NO
            verdicts.append(Answer(v, reason=f"{r.kind.value} in effect"))
        elif r.kind is Kind.TIME_LIMIT:
            verdicts.append(Answer(Verdict.LIMITED, limit_minutes=r.limit_minutes,
                                   reason=f"{r.limit_minutes}min limit"))
        elif r.kind is Kind.PERMIT_EXEMPT_LIMIT:
            # permit_area must be a hashable scalar; malformed model output (e.g. a list)
            # is treated as "no matching permit"
            if isinstance(r.permit_area, str) and r.permit_area in permit_areas:
                verdicts.append(Answer(Verdict.OK, reason=f"permit {r.permit_area} exempts"))
            else:
                verdicts.append(Answer(Verdict.LIMITED, limit_minutes=r.limit_minutes,
                                       reason=f"{r.limit_minutes}min limit without permit {r.permit_area}"))
    if not verdicts:
        return Answer(Verdict.OK, reason="no restriction in effect")
    verdicts.sort(key=lambda a: SEVERITY.index(a.verdict))
    best = verdicts[0]
    if best.verdict is Verdict.LIMITED:
        # multiple limits: strictest applies. Tolerate None (malformed model reads).
        limits = [a.limit_minutes for a in verdicts
                  if a.verdict is Verdict.LIMITED and a.limit_minutes is not None]
        best.limit_minutes = min(limits) if limits else None
    return best
