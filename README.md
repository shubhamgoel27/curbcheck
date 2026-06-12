# curbcheck 🅿️

**Can a small VLM tell you if you can park here?**

San Francisco parking sign stacks are a perception + logic + time-reasoning gauntlet:
read four stacked signs, combine the restrictions, apply them to "Tuesday, 5:30pm,
no permit," and decide if you're about to get towed. Frontier models are decent at it.
Small open VLMs are not. This project measures the gap, then closes some of it.

## The four artifacts

1. **The exam**: a benchmark of SF parking-sign images. Synthetic sign stacks rendered
   from official Caltrans/FHWA vector specs, seeded with real rule distributions from
   SFMTA's 144k-sign inventory, plus a real-photo test set (city PDDL photos, 311 citizen
   reports as a hard split, Mapillary/Flickr CC, and our own labeled photo walks).
   Three question layers: read (extract rules to JSON), reason (can I park now? until
   when?), abstain (what can't be determined from the signs alone?).
2. **The report card**: a leaderboard of small VLMs vs frontier reference, with the
   money chart: accuracy vs number of signs on the pole.
3. **The student**: a small VLM fine-tuned on synthetic stacks, evaluated on real
   photos. The headline question: does synthetic training survive contact with a faded,
   sticker-covered Mission Street pole?
4. **The demo**: upload a sign photo, pick a day and time, get an answer.

## Layout

- `schema/rules.py`: the rule schema + `can_park()` resolver (ground truth for all questions)
- `harvest/`: real-photo collection scripts (see `notes/sources.md` for the licensing map)
- `render/`: synthetic sign-stack renderer (Caltrans R-series specs)
- `data/`: gitignored; run `python harvest/pilot.py` to pull a sample

## Status

Early. Pilot harvest works (30/30 valid images across DPW + 311 sources). Renderer next.
