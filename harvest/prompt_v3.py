"""Teacher prompt v3: reason-then-answer, per-sign confidence, sharper rules.

Improvements over v1:
  - brief reasoning pass before the JSON (read each plate aloud) -> better on stacks
  - each restriction carries "confidence": high|low  -> low ones can be filtered
  - explicit normalization (dates->weekday, ANY TIME, X HR/MIN limits, ranges)
  - final answer fenced in ```json so reasoning text never confuses the parser
"""

TEACHER_PROMPT_V3 = """You are reading a San Francisco street pole to extract its PARKING rules.

Step 1. Briefly list each sign you can see on the pole and read its text aloud. Note which
are parking-related and which are not (speed limit, stop, street name, one-way: NOT parking).

Step 2. For every PARKING-related sign, produce one JSON object:
{
  "kind": one of [no_parking, no_stopping, tow_away, time_limit, permit_limit, street_cleaning, loading_only],
  "days": ["MON","TUE",...] the weekdays it applies,
  "start": "HH:MM" 24-hour, "end": "HH:MM" 24-hour,
  "limit_minutes": integer or null,
  "permit_area": single letter or null,
  "tow": true if the sign threatens towing else false,
  "weeks": list of which weeks of the month it applies, e.g. [2,4] for "2nd & 4th MONDAY"; null for every week,
  "confidence": "high" if every field is clearly legible, else "low"
}

Rules:
- kind = permit_limit when there is a permit exemption ("EXCEPT VEHICLES WITH AREA X PERMIT");
  time_limit when it is a plain hour/minute limit with no permit.
- "X HOUR PARKING" -> limit_minutes = X*60. "30 MINUTE" -> 30.
- "NO PARKING ANY TIME" / "TOW AWAY NO STOPPING ANY TIME" -> days all seven, start 00:00, end 23:59.
- Calendar dates (e.g. 6/17/26) -> convert to the matching weekday(s) in days.
- A day range "MON THRU FRI" -> all five weekdays. "TUE & THU" -> those two.
- "2nd & 4th MONDAY" / "1st & 3rd TUE" -> days has the weekday, weeks = [2,4] / [1,3].
- If a field is present on the sign but you cannot read it, set it null and confidence "low".
- Do NOT emit an object unless at least one of days/start/end is legible.
- Ignore non-parking signs entirely.

Step 3. Output ONLY the final answer as a JSON array inside a fenced block:
```json
[ ... ]
```
If no parking sign is legible, output ```json
[]
```"""
