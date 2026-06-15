# Data sources and licensing map

Verified 2026-06-12 via four research routes (academic datasets, community CV platforms,
CC photo repositories, SF civic data).

## Tier 1: images republishable in the dataset

| Source | Volume | License | Notes |
|---|---|---|---|
| DPW Street Space Permit Photos ([pigs-fac7](https://data.sfgov.org/City-Infrastructure/Parking-Signs-Street-Space-Permit-Photos/pigs-fac7)) | 2,934 | PDDL | city-taken photos of posted temporary parking signs |
| Mapillary API v4 | thousands (SF bbox) | CC BY-SA 4.0 | filter `map_features` by bbox + `object_values=regulatory--no-parking--*` etc.; attribution + share-alike required |
| Flickr CC (API, license=4,5,9,10) | ~1.2k "no parking sign", ~90 SF | CC BY / CC BY-SA / CC0 per photo | harvest with per-photo license record |
| Wikimedia Commons (Parking signs in California/US cats) | low hundreds | per-file CC/PD | filter out MUTCD diagrams |
| Own photo walks | as needed | ours (publish CC BY) | the human-anchored core test set |

## Tier 2: free to use, distribute as URL lists only (no pixel rehosting)

| Source | Volume | Why URL-only |
|---|---|---|
| SF 311 "MTA Parking Traffic Signs" cases ([vw6y-z8j6](https://data.sfgov.org/City-Infrastructure/311-Cases/vw6y-z8j6)) | ~21.8k cases, ~5k parking-relevant with live photos | citizen submitters retain copyright (license granted to City only). Photos on spot-sf-res.cloudinary.com resolve; verintcloudservices.com URLs are auth-walled (skip). Subtypes = free labels: street_cleaning, no_parking, permit_parking, tow_away. Defaced/faded/bent cases = natural hard split |
| Kaggle "SF Parking Sign Detection" (Appen mirror) | 1,934 tiles | CC0-tagged but Google Street View-derived; internal dev only |

## Renderer references (public domain)

- FHWA 2024 Standard Highway Signs: vector SVG/EPS artwork, US federal work, public domain.
  https://mutcd.fhwa.dot.gov/kno-shs_2024-release-status/index.htm
- Caltrans CA sign spec PDFs (SF uses CA codes, not federal): R26 (no parking), R30 (time
  limit), R32 (street sweeping). https://dot.ca.gov/programs/safety-programs/sign-specs
- SFMTA Street Signs inventory ([m48z-6ji4](https://data.sfgov.org/City-Infrastructure/Street-Signs/m48z-6ji4)):
  144,333 geocoded signs with CA sign codes + legend text, PDDL. Drives realistic rule
  distributions for the generator and cross-checks photo labels by location.

## Ruled out

- Google Street View anything: ToS prohibits dataset redistribution.
- Waymo Open (license blocks republishing), LISA (no parking classes), nuScenes (no SF).
- SFMTA Flickr (non-commercial policy, historic transit content anyway).
- Unsplash/Pexels: license prohibits dataset-style rehosting; could use by reference only.

## Leads

- UW Tacoma parking-sign dataset (4,191 images, 27 symbol classes, unreleased):
  email uwcheng@uw.edu, cite IJCAI-2022 AI4AD workshop paper.

## Cross-CA expansion (2026-06-14)

- **Mapillary: SHELVED.** Tested across Oakland/Berkeley. Tons of parking-sign detections
  (222 in 4 tiles) but only ~3% are >=110px (readable); wide dashcam FOV captures signs too
  small/distant even after MVT-geometry crops. Our task needs close-ups. `harvest/mapillary_ca.py`
  kept for reference, marked shelved.
- **The lesson:** SF 311/DPW worked because those systems collect deliberate sign close-ups.
  The right CA expansion is other cities' 311 / SeeClickFix systems with photo attachments,
  not street-level imagery. Oakland (data.oaklandca.gov) and San Jose (data.sanjoseca.gov)
  portals are live; photo availability per-dataset is the open question (needs a sourcing sweep
  like the original SF hunt). NYC 311 confirmed to have NO photo field.
