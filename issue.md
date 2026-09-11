# Property-Tax-Pipeline — technical problems, for the pipeline team

**What this is:** an inventory of everything structurally wrong with the current pipeline as an *integration partner* for the dashboard/phone-app system — with code-level evidence — so the pipeline team can plan the rework. **What this is not:** an architecture proposal. Design comes after the team reads and responds to this.

**One rule pinned before anything else:** *the match core is frozen.* The layered matcher (`matchers/mobile → property_id → spatial → electricity → name_locality`) and its thresholds produced the pinned Amritsar baseline (356k+ GIS polygons, ~105k mSeva rows, ~68% matched) and **nothing in this document asks for changes inside `matchers/` or scoring logic.** Every problem below is about identity, output shape, run management, and interface — the shell around the matcher, not the matcher.

Measured numbers come from the Amritsar run (2026-08/09): 358,294 GIS · 105,336 mSeva · 71,833 matched onto 29,048 *unique* GIS uids (many-to-one) · `potential_defaulters: 325,363` in summary but **200,165 emitted** (~125k silently dropped) · 110.9 MB defaulters GeoJSON · 81 raw `property_usage` spellings, 16 `property_type` values.

---

## P1. No run identity — a run is not a thing you can address

**Today:** `python -m Property-Tax-Pipeline --city … --steps …` mutates shared state in-memory and writes files with *fixed names* (`{City}_Match_Register.csv` etc.). There is no `run_id`, no job record, no status, no receipts. A run exists only while the process lives.

**Why it hurts:**
- **Disconnect = loss.** Geocoding + matching Amritsar takes a long time and depends on paid external APIs (`--gemini-key`, `--google-key`). If the shell/connection dies mid-run, there is no job handle to query, no resume point, no way to fetch partial results — you rerun and re-pay.
- **"Did September's run finish?" is unanswerable** except by checking file timestamps.
- Nothing stops a second operator from starting the same city twice and interleaving writes.

**Pressure it creates:** a run registry (`run_id`, city, month, inputs, state, timestamps, owner) and step-level checkpointing. The pipeline must survive process death and expose "run N status / fetch N's outputs."

## P2. Same-name overwrites — nothing stops the pipeline from destroying its own history

**Today:** `orchestrator.py:281,473,559` write with `to_csv/to_file` to fixed paths in `--output` (`results/Amritsar_Defaulters.geojson` …). Run twice → last month's file is gone. `results/` even holds a manual `backup_2026-08-12/` folder — the workaround exists *because* the tool clobbers.

**Why it hurts:** the phase-1 rule is *nothing is ever erased; every change logs who/when*. A tool that overwrites its own artifacts cannot prove what it shipped last month — so even though month-on-month comparison is computed downstream, the pipeline destroying its own evidence makes its side of that story unverifiable. It also makes debugging unreproducible ("what did the March file contain?" is unrecoverable).

**Pressure it creates:** immutable, versioned output directories — one per (city, data-month, run-seq), `exist_ok=False`, never modified after success. Caller passes the directory; the pipeline never picks its own destination implicitly.

## P3. The pipeline computes things, then throws them away without telling anyone

**Today, step by step:**
1. The matcher reads all 358,294 survey parcels and all 105,336 tax rows.
2. It works out the *defaulter suspects*: parcels that exist on the survey map but have no tax record — its own notes say 325,363 of them.
3. Then it asks: "which of these have a drawable shape?" The ones that do (~200,165) get written to the output file. The other **~125,000 are kept in memory and silently never written** — they appear in no CSV, no GeoJSON, nowhere.
4. Wards where we *know* a property is exempt (religious places, government buildings) also vanish the same way.

So the receiving team can never see the full picture: the numbers the pipeline computed internally, and the numbers in the files it delivered, tell two different stories — and nothing flags the gap.

**Why it hurts:** the office dashboard's main screen is supposed to show *everything* — taxpayers, suspected non-payers, and exempt. With ~125k rows missing from the files, the dashboard's totals can never match the map, and the month-end "do the chart numbers equal the map numbers?" check fails on day one, every month, by construction — not because anyone did something wrong, but because the data was never shipped.

**Pressure it creates:** the pipeline must emit one file containing every record it knows about, each labelled *who it is* (pays tax / suspected / exempt) and *what location evidence exists* (full shape on map / just a point / nothing). Plus it should check its own totals: "I computed 325,363 suspects and I wrote 325,363 suspect rows" — before calling the run a success.


## P4. Messy spellings arrive as-is — and in the new pipeline the field arrives empty

**Today, concretely (measured in both pipelines, 2026-09-10):**
- *Old pipeline, Amritsar:* `Amritsar_Defaulters.geojson` (200,165 features) passes `property_usage` through raw — **81 distinct spellings** ("Residential Area/ Colony" ×139,592, "Residential and Commercial" ×32,138, …); property types show 16 variants; ward references are free text like `"W. no. 12"`. Nothing is checked against an official list, and unrecognised values don't stop the run.
- *New pipeline (geoai), Barnala:* the outputs *declare* `property_usage`/`property_type` columns but fill them with **empty strings — 0 of 21,590 defaulter rows carry a usage value**. `data_loader.py` knows how to find the column in the input file, but the emit step writes blanks, so the field never actually arrives.

Now picture the receiving end: the dashboard groups everything by *ward* and by *usage category*. Spellings must map to canonical values or ward totals silently mis-count — and a field that is 100% blank is just a more extreme version of unusable. The app therefore enforces a rule: *if a spelling isn't on the city-signed list, stop and report — never guess.* That's the correct, safe behaviour — but against the old pipeline it means **a single new typo anywhere in 105,000 rows rejects the entire city's monthly update**, and the fix-bounce cycle costs days every month.

**Why it hurts:** the monthly rhythm depends on acceptance being near-automatic when data is clean. Right now it is near-guaranteed to fail, for spelling reasons only.

**Pressure it creates:** clean the spellings *at the source*, inside the pipeline, against the signed master lists (ward list, usage list). Unknown value ⇒ stop the run with a clear inventory ("these 3 new spellings appeared, in these rows") — fix once upstream, instead of monthly rejection downstream.

## P5. Folder-in/folder-out is not an interface

**Today:** inputs are CLI file paths; outputs are CSV/GeoJSON files in a folder; consumers poll the directory and guess freshness by timestamp. No API, no schema version on the outputs, no declared column contract, no event on completion.

**Why it hurts:** "kind of not technically fit for implementation — only works for checking things" (correct assessment). An app cannot *call* this: there is no request/response, no auth boundary, no per-tenant isolation, no way to distinguish "run finished cleanly" from "file is half-written" (a `to_csv` mid-crash leaves a truncated file that looks valid).

**Pressure it creates:** atomic publish (write temp → fsync → rename + receipt), machine-readable manifest per run, and — later — a service/queue wrapping instead of conversational CLI.

## P6. Geometry handling doesn't scale and drops records

**Today (measured):** defaulters GeoJSON = **200,165 features, 110.9 MB** in one file — frontend can't load it, grep/jq can't stream it sanely, and anything without geometry just disappears from it. Dedup of ~3,883 duplicate `gis_uid`s in the real shapefile has no rule. WGS84 vs projected-crs handling is implicit.

**Why it hurts:** map serving needs ~30 KB ward summaries, not 111 MB monoliths. Rows with no geometry are *data*, not dirt (P3).

**Pressure it creates:** geometry as a keyed-by-uid artifact; a/geo tiers produced for serving; explicit CRS; dedup rule stated ("first occurrence wins") and counted.

## P7. Per-tenant deployment duplication — and the city is hardcoded in the code

**Today (verified in `pipeline_geoai/`):** onboarding a city isn't a redeploy — it's a *code edit + training run + 610 MB of new artifacts*:
- `geoai/trainer.py:150` saves the knowledge base to the literal path `models/Barnala/`; `geoai/inferencer.py:222` loads `models/Barnala/`. The **city name is a string literal in source** — Amritsar means editing code, retraining, and packaging a second 610 MB folder.
- `models/Barnala/` on disk: 610 MB, of which ~514 MB is **the same embedding array saved ~6×** (address / full_address / locality / owner / road / ward pickles are byte-identical sizes), plus FAISS + BallTree + KDTree + RTree pickles indexing the same data four ways.
- The SentenceTransformer base model is still downloaded/loaded per process on top of the city artifacts (`geoai/embeddings.py`).
- Same pattern at the framework level: each tenant stands up its own full pipeline environment rather than sharing one deployment with tenant-scoped jobs, and every run re-loads big frames fully in memory (358k polygons × ~105k rows).

**Why it hurts:** N cities = N × (code fork + training + 610 MB) — storage, compute, and a manual ML step per city, forever. Two cities wanting "refresh on the 1st" also collide (no queue). The FRS's "< 2 hours to onboard a new ULB" target is unreachable while onboarding includes training.

**Pressure it creates:** one shared embedding/model layer loaded once; city data as *data* (tenant_key → artifacts in storage, path from config, never from a literal in code); indexes built once per dataset version, not re-pickled per run; training — if it stays — as a first-class queued job with its own run id (same registry as P1), not a prerequisite hack for onboarding.

*Document ends at P7 — the pressures above are the complete ask list. Matching internals (`matchers/*`) stay frozen; architecture belongs to step two. One carry-through requirement stands outside the problem list above: month-on-month comparison is computed downstream from full monthly snapshots, which works **only if every output row carries the sources' own IDs verbatim** (mSeva property id, GIS parcel uid) — no new keying scheme, just don't strip what the sources already provide.*
 