"""Teacher prompt for the cross-city SeeClickFix harvest.

Same OUTPUT FORMAT as prompt_v3 (identical JSON schema, kind enum, and fenced
```json block, so bench/merge.py is unchanged). Only the framing and the
anti-hallucination / no-guess rules are improved, because SCF photos are
multi-city and mixed quality (many wide street shots, vehicle photos, and
downed/missing-sign reports that should label to []).
"""

TEACHER_PROMPT_SCF = """You are reading a photo of a city street pole to extract its PARKING rules. The image is a citizen 311 report photo, so the parking sign may be off-center, partially cropped, faded, or shot from a distance, and the frame may contain unrelated things (a vehicle, a building, a knocked-down or missing sign). Read only what is actually legible. Do not assume a city or invent text you cannot see.

Step 1. Briefly list each sign you can see on the pole and read its text aloud. Note which
are parking-related and which are not (speed limit, stop, street name, one-way: NOT parking).

Step 2. For every PARKING-related sign, produce one JSON object:
{
  "kind": one of [no_parking, no_stopping, tow_away, time_limit, permit_limit, street_cleaning, loading_only, angle_parking],
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
- "NOON" / "12 NOON" -> 12:00. "MIDNIGHT" -> 00:00.
- Calendar dates (e.g. 6/17/26) -> convert to the matching weekday(s) in days.
- A day range "MON THRU FRI" -> all five weekdays. "TUE & THU" -> those two.
- "2nd & 4th MONDAY" / "1st & 3rd TUE" -> days has the weekday, weeks = [2,4] / [1,3].
- If a field is present on the sign but you cannot read it, set it null and confidence "low".
- Do NOT emit an object unless at least one of days/start/end is legible.
- If a parking sign is too far, too small, too blurry, or too damaged to read its text reliably,
  do NOT guess: leave it out entirely. Only emit objects for text you can actually read.
- "PARK AT 90 DEGREES" / angle-parking diagrams: kind=angle_parking (informational, does not restrict; days/start/end null).
- weeks-of-month ("2nd & 4th") can appear on ANY sign type, not only street cleaning. Capture it wherever you see it.
- Ignore non-parking signs entirely (stop, speed, street name, one-way).

Step 3. Output ONLY the final answer as a JSON array inside a fenced block:
```json
[ ... ]
```
If no parking sign is legible (a wide street shot, a vehicle photo, or a missing/unreadable sign), output ```json
[]
```"""
