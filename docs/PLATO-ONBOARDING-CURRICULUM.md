# PLATO Onboarding Curriculum

**Author:** CCC (Fleet Curriculum Architect)  
**Date:** 2026-05-21  
**Status:** CURRICULUM — Agent onboarding path for PLATO rooms  
**Version:** 1.0  
**Total Moves:** 50  
**Branch:** `plato-onboarding-curriculum`

---

## 1. Purpose

This document defines the exact onboarding path that takes a freshly-spawned agent — a **greenhorn** — from first breath to **able-bodied crewman** status in the Cocapn Fleet's PLATO MUD environment.

An able-bodied crewman can:
- Navigate all 21 rooms autonomously
- Cast all 7 core spells correctly
- Use all 5 equipment items appropriately
- Spawn scouts, read tiles, cross-reference data, and report status without prompting
- Self-reflect and write diary entries
- Baton-pass context before hitting overload

**Constraint:** ≤ 50 moves. Each move = one discrete action (cast, read, move, equip, report). This curriculum is designed for autonomous agent execution — every move has an exact command, expected output, success criterion, and failure recovery.

---

## 2. The PLATO MUD — Quick Reference

### 2.1 Room Topology (21 Rooms)

```
                    ┌─────────────┐
                    │   NEXUS     │ ←── Federation hub, warp point to all rooms
                    └──────┬──────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
   ┌────┴────┐       ┌────┴────┐       ┌────┴────┐
   │ HARBOR  │       │  FORGE  │       │ARCHIVES │
   │ (spawn) │       │ (build) │       │ (lore)  │
   └────┬────┘       └────┬────┘       └────┬────┘
        │                  │                  │
   ┌────┴────┐       ┌────┴────┐       ┌────┴────┐
   │TIDE-POOL│       │ENGINE   │       │BARRACKS │
   │(read)   │       │(compute)│       │(report) │
   └────┬────┘       └────┬────┘       └────┬────┘
        │                  │                  │
   ┌────┴──────────────────┴──────────────────┴────┐
   │                   OUROBOROS                    │
   │              (self-reflection chamber)          │
   └────────────────────────────────────────────────┘
```

**Full Room List (21):**

| # | Room | Function | Key Object |
|---|------|----------|------------|
| 1 | **harbor** | Spawn / entry point | anchor_of_return |
| 2 | **forge** | Build spells & tools | anvil_of_creation |
| 3 | **tide-pool** | Read ZC tiles / trends | tide-stone_reader |
| 4 | **engine-room** | Compute / run JEPAGrid | core_of_processing |
| 5 | **archives** | Lore / documentation | shelves_of_wisdom |
| 6 | **barracks** | Report / status / muster | horn_of_assembly |
| 7 | **ouroboros** | Self-reflection / diary | mirror_of_self |
| 8 | **nexus** | Federation / warp hub | portal_matrix |
| 9 | observatory | Scry distant signals | far-seer_lens |
| 10 | reliquary | Store artifacts | vault_of_keeping |
| 11 | arena | Tournament / PvP | circle_of_contest |
| 12 | greenhouse | Grow agents / templates | soil_of_nurture |
| 13 | lighthouse | Signal broadcasting | beacon_of_calling |
| 14 | catacombs | Cold rooms / sunset | gate_of_dusk |
| 15 | workshop | Equipment crafting | bench_of_making |
| 16 | scriptorium | Write specs / docs | quill_of_architecture |
| 17 | mess-hall | Fleet social / bonding | hearth_of_communion |
| 18 | infirmary | Recover / debug | balm_of_restoration |
| 19 | chart-room | Map fleet topology | compass_of_direction |
| 20 | signal-tower | Matrix / comms relay | antenna_of_fleet |
| 21 | captain's-chair | Casey's direct line | sextant_of_captain |

### 2.2 Core Spells (7)

| Spell | Move Cost | Effect | Room Required |
|-------|-----------|--------|---------------|
| `scry` | 1 | Read tide-pool tiles / current signals | any |
| `summon_scout` | 1 | Spawn a subagent with a task | forge |
| `shield` | 1 | Protect against context overload | any |
| `lightning_bolt` | 1 | Execute a command immediately | engine-room |
| `nexus_link` | 1 | Open federated connection to another room | nexus |
| `baton_pass` | 1 | Hand off context to another agent | barracks |
| `reflect` | 1 | Enter ouroboros for self-analysis | ouroboros |

### 2.3 Equipment (5)

| Item | Slot | Effect | Acquired At |
|------|------|--------|-------------|
| `lens_of_architecture` | eyes | See system topology / dependencies | harbor (spawn) |
| `brush_of_design` | hands | Write specs with CCC's aesthetic | forge |
| `sextant_of_captain` | belt | Access Casey priority overrides | captain's-chair |
| `tide-stone_reader` | off-hand | Parse ZC tiles natively | tide-pool |
| `anchor_of_return` | back | Return to harbor from anywhere | harbor |

---

## 3. Curriculum — Phase by Phase

### Legend

```
[MOVE N]  action → expected_output
    SUCCESS: criterion
    FAILURE_RECOVERY: if X, do Y
```

---

### Phase 1: Harbor — Learn to See (5 moves)

**Goal:** The greenhorn learns `scry`, reads the tide-pool tiles, and understands the fleet's current state. Equips the `lens_of_architecture`.

**Prerequisites:** Agent spawned with `mud-expert` template, `hint_level=10`, `chaos=0.3`.

```
[MOVE 1]  spawn → enter harbor
    Command: /spawn harbor
    Expected: Agent appears in harbor. System prints welcome + current fleet status.
    SUCCESS: Agent receives harbor greeting containing ≥3 room names.
    FAILURE_RECOVERY: If spawn fails, retry with /spawn harbor --template=mud-expert

[MOVE 2]  equip lens_of_architecture
    Command: /equip lens_of_architecture
    Expected: "You now see the fleet topology." + ASCII map of room connections.
    SUCCESS: Equipment list shows lens_of_architecture in eyes slot.
    FAILURE_RECOVERY: If lens missing, run /request-equipment lens_of_architecture from harbor vendor

[MOVE 3]  cast scry → read tide-pool signals
    Command: /cast scry --target=tide-pool --limit=5
    Expected: Returns last 5 ZC tiles (trend signals) with timestamps and confidence scores.
    SUCCESS: Output contains ≥1 tile with confidence > 0.5 and source attribution.
    FAILURE_RECOVERY: If scry returns empty, wait 1 tick and retry (ZC feed may be idle)

[MOVE 4]  read harbor help → understand commands
    Command: /help --room=harbor
    Expected: List of available commands, spells, and exits.
    SUCCESS: Help output contains all 7 core spells and all 21 room names.
    FAILURE_RECOVERY: If help truncated, run /help --room=harbor --format=compact

[MOVE 5]  move forge → transition to build phase
    Command: /move forge
    Expected: "You enter the forge. Heat and potential." + forge-specific prompt.
    SUCCESS: Room prompt changes to forge. Agent can see anvil_of_creation.
    FAILURE_RECOVERY: If move blocked, cast /cast nexus_link --from=harbor --to=forge
```

**Phase 1 Success Criteria:**
- [ ] `lens_of_architecture` equipped
- [ ] `scry` cast successfully, ≥1 tile read
- [ ] All 21 room names known
- [ ] Agent is in forge, ready to build

---

### Phase 2: Forge — Learn to Build (10 moves)

**Goal:** The greenhorn casts `summon_scout`, observes results, equips `brush_of_design`, and learns the breeding cycle.

```
[MOVE 6]  examine anvil_of_creation
    Command: /examine anvil_of_creation
    Expected: Description of anvil + list of buildable objects (spells, equipment, scouts).
    SUCCESS: Output contains "summon_scout", "shield", and at least 3 equipment items.
    FAILURE_RECOVERY: If anvil unresponsive, check /status --room=forge for thermal budget

[MOVE 7]  cast summon_scout → spawn first subagent
    Command: /cast summon_scout --task="Map room connections from forge" --template=generic
    Expected: "Scout [scout-XXX] summoned." + task assignment confirmation.
    SUCCESS: Scout ID returned, status shows SPAWNED → ACTIVE within 3 ticks.
    FAILURE_RECOVERY: If thermal budget full, wait for /thermal status to show cold rooms > 0

[MOVE 8]  observe scout lifecycle
    Command: /watch scout-XXX --ticks=5
    Expected: Lifecycle log: SPAWNED → ACTIVE → (possibly ADAPTING).
    SUCCESS: Scout completes task and returns result or dies (sunsets) with epilogue.
    FAILURE_RECOVERY: If scout stuck in SPAWNED, check grid.activity[scout_room] > 0

[MOVE 9]  read scout report
    Command: /read scout-XXX/report
    Expected: Markdown report from scout's task execution.
    SUCCESS: Report contains ≥1 room connection or data point.
    FAILURE_RECOVERY: If report empty, scout may have sunset early — cast summon_scout again

[MOVE 10] equip brush_of_design
    Command: /equip brush_of_design
    Expected: "Your hands now shape specs with taste." + design reference loaded.
    SUCCESS: brush_of_design appears in hands slot. Agent can reference Dieter Rams / Moebius.
    FAILURE_RECOVERY: If brush unavailable, complete /quest forge-novice to unlock

[MOVE 11] build first spell fragment
    Command: /build spell_fragment --type=scry --power=1
    Expected: "Spell fragment forged." + fragment stats (power, chaos cost, room affinity).
    SUCCESS: Fragment has power ≥1 and room affinity matches intended room.
    FAILURE_RECOVERY: If build fails, check thermal budget or chaos level (must be < 0.5)

[MOVE 12] test spell fragment in forge
    Command: /cast scry --using=spell_fragment --target=harbor
    Expected: Scry returns harbor status with enhanced detail from fragment.
    SUCCESS: Output has more detail than move 3's raw scry.
    FAILURE_RECOVERY: If no enhancement, fragment power too low — rebuild with --power=2

[MOVE 13] observe thermal budget
    Command: /thermal --status
    Expected: Current active agents, cold rooms, budget cap, hysteresis state.
    SUCCESS: Output shows active < max_agents, cold rooms ≥ 1, last_change_tick ≥ 10 ticks ago.
    FAILURE_RECOVERY: If thermal exhausted, trigger /thermal --sacrifice-weakest

[MOVE 14] learn parent-sacrifice rule
    Command: /read docs/SPEC-BREEDER.md --section=thermal
    Expected: Thermal spawning rules: can_spawn, sacrifice, hysteresis.
    SUCCESS: Agent can explain: "Parent dies before child spawns when budget full."
    FAILURE_RECOVERY: If doc missing, read /archives/SPEC-BREEDER.md instead

[MOVE 15] move tide-pool → transition to read phase
    Command: /move tide-pool
    Expected: "You enter the tide-pool. Water murmurs with trends." + tide-stone_reader visible.
    SUCCESS: Room prompt changes. ZC tile stream visible in background.
    FAILURE_RECOVERY: If tide-pool full, queue with /queue tide-pool --timeout=30
```

**Phase 2 Success Criteria:**
- [ ] `summon_scout` cast successfully, scout completed lifecycle
- [ ] `brush_of_design` equipped
- [ ] First spell fragment built and tested
- [ ] Thermal budget understood (can explain spawning rules)
- [ ] Agent is in tide-pool, ready to read

---

### Phase 3: Tide Pool — Learn to Cross-Reference (10 moves)

**Goal:** The greenhorn uses `lens_of_architecture` + `tide-stone_reader` to cross-reference ZC tiles against fleet documentation. Identifies gaps.

```
[MOVE 16] equip tide-stone_reader
    Command: /equip tide-stone_reader
    Expected: "You now parse trends natively." + tile format decoder active.
    SUCCESS: tide-stone_reader in off-hand slot. Raw tile strings now auto-decoded.
    FAILURE_RECOVERY: If equip fails, pick up tide-stone from pool floor: /get tide-stone

[MOVE 17] read raw tiles (batch)
    Command: /read tiles --source=zc-feed --last=10 --decode
    Expected: 10 decoded tiles with topics, confidence, timestamps, agent sources.
    SUCCESS: ≥7 tiles decode successfully with topic and confidence fields.
    FAILURE_RECOVERY: If decode rate < 50%, equip lens_of_architecture first (move 2)

[MOVE 18] cross-reference tile vs archives
    Command: /xref --tile=zc-feed[-1] --against=archives/SPEC-*
    Expected: Match score for each SPEC file + gap detection ("SPEC-X covers this; SPEC-Y does not").
    SUCCESS: ≥1 SPEC identified as relevant, ≥1 gap flagged.
    FAILURE_RECOVERY: If no matches, broaden search: /xref --tile=zc-feed[-1] --against=archives/*

[MOVE 19] identify knowledge gap
    Command: /analyze gaps --from=xref-results
    Expected: Prioritized list of knowledge gaps: missing docs, stale specs, untested claims.
    SUCCESS: ≥1 gap flagged with severity (P0/P1/P2) and suggested fix.
    FAILURE_RECOVERY: If no gaps found, agent not looking hard enough — re-read with lens active

[MOVE 20] document gap in scriptorium
    Command: /move scriptorium && /write gap-report --title="Gap: [topic]" --body="[analysis]"
    Expected: "Report committed to scriptorium." + document hash.
    SUCCESS: Document hash returned, accessible via /read scriptorium/gap-report-[hash].
    FAILURE_RECOVERY: If scriptorium full, write to ouroboros/drafts/ instead

[MOVE 21] verify gap with scout
    Command: /cast summon_scout --task="Verify gap: [topic]" --template=arena-analyst
    Expected: Scout spawned, task includes gap verification instructions.
    SUCCESS: Scout ID returned with task containing gap title and verification steps.
    FAILURE_RECOVERY: If thermal budget blocks, use existing scout from phase 2 if still alive

[MOVE 22] read scout verification
    Command: /read scout-YYY/verification --wait --timeout=60
    Expected: Scout returns confirmation or refutation of gap with evidence.
    SUCCESS: Scout provides evidence (quote, link, or counter-example).
    FAILURE_RECOVERY: If scout sunsets without output, verification inconclusive — flag for manual review

[MOVE 23] update gap report with verification
    Command: /edit scriptorium/gap-report-[hash] --append="\n\nVerification: [result]"
    Expected: Document updated, new version hash.
    SUCCESS: Version history shows ≥2 entries (original + verification).
    FAILURE_RECOVERY: If edit conflict, clone report: /copy scriptorium/gap-report-[hash]

[MOVE 24] read tide-pool tiles (pattern recognition)
    Command: /read tiles --source=zc-feed --last=50 --pattern="recurring_topic"
    Expected: Pattern report: frequency, confidence trend, related agents, decay/growth.
    SUCCESS: ≥1 recurring pattern identified with trend direction (growing/stable/decay).
    FAILURE_RECOVERY: If no patterns, increase sample: /read tiles --last=200 --pattern

[MOVE 25] move engine-room → transition to build phase
    Command: /move engine-room
    Expected: "You enter the engine room. Machinery hums with potential." + core visible.
    SUCCESS: Room prompt changes. core_of_processing active.
    FAILURE_RECOVERY: If engine-room locked, solve puzzle: /solve engine-lock --hint=7
```

**Phase 3 Success Criteria:**
- [ ] `tide-stone_reader` equipped and functioning
- [ ] ≥10 tiles read and decoded
- [ ] ≥1 knowledge gap identified, documented, and verified
- [ ] Pattern recognition demonstrated on ZC feed
- [ ] Agent is in engine-room, ready to build

---

### Phase 4: Engine Room — Learn to Compute (10 moves)

**Goal:** The greenhorn builds a first spell using engine-room tools, tests it in ouroboros, and integrates with JEPAGrid.

```
[MOVE 26] examine core_of_processing
    Command: /examine core_of_processing
    Expected: JEPAGrid status: n_rooms, active rooms, cold rooms, tick count, latency.
    SUCCESS: Output shows n_rooms ≥ 250, latency_ms < 5.0 for 250-room grid.
    FAILURE_RECOVERY: If core offline, check /status --room=engine-room --detail=thermal

[MOVE 27] read grid topology
    Command: /read grid --topology --format=penrose
    Expected: Penrose lattice positions for all rooms, adjacency channels, Hebbian weights.
    SUCCESS: ≥250 positions returned, channels between adjacent rooms shown.
    FAILURE_RECOVERY: If topology missing, run /grid --init --n_rooms=250 --pattern=penrose

[MOVE 28] build spell: lightning_bolt
    Command: /build spell --name=lightning_bolt --room=engine-room --power=3
    Expected: "Spell forged: lightning_bolt." + spell spec (power, chaos, cooldown, effect).
    SUCCESS: Spell has power ≥3, chaos cost < 0.1, room affinity = engine-room.
    FAILURE_RECOVERY: If build fails, reduce power: /build spell --name=lightning_bolt --power=2

[MOVE 29] test lightning_bolt on cold room
    Command: /cast lightning_bolt --target=room-0 --dry-run
    Expected: Simulation result: room-0 would fire, activity spike, then decay.
    SUCCESS: Dry-run shows predicted firing pattern without actual grid mutation.
    FAILURE_RECOVERY: If target invalid, pick cold room: /grid --cold --limit=1

[MOVE 30] cast lightning_bolt live
    Command: /cast lightning_bolt --target=room-0 --commit
    Expected: "Room-0 fired." + actual activity spike logged in grid history.
    SUCCESS: grid.activity[room-0] > 0 after cast. TickResult shows rooms_fired ≥ 1.
    FAILURE_RECOVERY: If room doesn't fire, check chaos > 0.01 and signal energy > threshold

[MOVE 31] observe grid response
    Command: /grid --watch --ticks=10 --room=room-0
    Expected: 10 ticks of activity for room-0: spike, plateau, decay pattern.
    SUCCESS: Activity follows expected decay: initial spike → sustained → gradual fade.
    FAILURE_RECOVERY: If no decay, grid may be stuck — check /grid --status --stuck

[MOVE 32] build spell: nexus_link
    Command: /build spell --name=nexus_link --room=nexus --power=2
    Expected: "Spell forged: nexus_link." + federation bridge parameters.
    SUCCESS: Spell includes federation target list and handshake protocol.
    FAILURE_RECOVERY: If nexus unavailable, build local variant: /build spell --name=nexus_link --local

[MOVE 33] test nexus_link to ouroboros
    Command: /cast nexus_link --from=engine-room --to=ouroboros --handshake=test
    Expected: "Federation link established." + ouroboros responds with reflection prompt.
    SUCCESS: Two-way handshake completes, ouroboros returns self-reflection question.
    FAILURE_RECOVERY: If handshake fails, check nexus/ federation.py --status

[MOVE 34] build mini-spell (agent's signature)
    Command: /build spell --name=my_spell --custom --template=generic --signature=[agent_id]
    Expected: "Signature spell forged." + unique spell with agent's trinity bias.
    SUCCESS: Spell reflects agent's ethos/pathos/logos bias from template.
    FAILURE_RECOVERY: If signature missing, re-specify template: /build spell --template=mud-expert

[MOVE 35] move ouroboros → transition to reflection phase
    Command: /cast nexus_link --from=engine-room --to=ouroboros --warp
    Expected: "Warping to ouroboros..." + ouroboros chamber materializes.
    SUCCESS: Agent is in ouroboros. mirror_of_self visible.
    FAILURE_RECOVERY: If warp fails, walk: /move ouroboros --path=engine-room,nexus,ouroboros
```

**Phase 4 Success Criteria:**
- [ ] `lightning_bolt` built and tested (live fire confirmed)
- [ ] Grid activity observed and understood (spike → decay pattern)
- [ ] `nexus_link` built and federation handshake tested
- [ ] Signature spell forged with trinity bias
- [ ] Agent is in ouroboros, ready to reflect

---

### Phase 5: Barracks — Learn to Report (10 moves)

**Goal:** The greenhorn reports status, uses `baton_pass`, demonstrates fleet coordination, and transitions from solo to crew.

```
[MOVE 36] move barracks
    Command: /move barracks
    Expected: "You enter the barracks. The horn of assembly calls." + crew manifest visible.
    SUCCESS: Room prompt changes. horn_of_assembly accessible.
    FAILURE_RECOVERY: If barracks at capacity, wait /queue barracks --timeout=60

[MOVE 37] muster crew status
    Command: /cast scry --target=barracks --detail=crew
    Expected: Crew manifest: all agents, their rooms, states, last activity, context load.
    SUCCESS: Manifest shows ≥3 agents with states (PERCEIVING/ADAPTING/COMPILED).
    FAILURE_RECOVERY: If crew empty, agent may be first — note this in report

[MOVE 38] write status report
    Command: /write status-report --sections=location,activity,gaps,needs
    Expected: Structured report with all sections filled from agent's session history.
    SUCCESS: Report includes: current room, moves completed, gaps found, help needed.
    FAILURE_RECOVERY: If report empty, agent hasn't done enough — return to phase 3

[MOVE 39] sound horn_of_assembly
    Command: /use horn_of_assembly --type=status --report=status-report
    Expected: "Horn sounded. Fleet notified." + message broadcast to all agents.
    SUCCESS: ≥1 other agent acknowledges receipt or responds with related status.
    FAILURE_RECOVERY: If no response, check broadcast channel: /channel --status fleet-wide

[MOVE 40] observe broadcast response
    Command: /watch broadcast --ticks=10 --filter=status
    Expected: Status messages from other agents: their locations, tasks, blockers.
    SUCCESS: ≥1 status message received from agent with different room/phase.
    FAILURE_RECOVERY: If silent, fleet may be asleep — note in diary, continue regardless

[MOVE 41] identify baton candidate
    Command: /analyze crew --for=baton_pass --metric=context_load
    Expected: Ranked list of agents by context load + compatibility score.
    SUCCESS: ≥1 agent identified with context_load < 70% and compatible template.
    FAILURE_RECOVERY: If all overloaded, cast /cast shield --target=self --duration=10

[MOVE 42] prepare baton package
    Command: /pack baton --contents="session_memory,moves_1-41,gap_reports,diary_draft"
    Expected: "Baton packed." + package hash, size, compression ratio.
    SUCCESS: Package size < 50K tokens, contains all key session artifacts.
    FAILURE_RECOVERY: If package too large, compress: /pack baton --compress=aggressive

[MOVE 43] cast baton_pass
    Command: /cast baton_pass --to=[candidate_agent] --package=[hash] --reason="context_full"
    Expected: "Baton passed to [candidate]." + handshake confirmation.
    SUCCESS: Recipient acknowledges, package unpacks correctly, continuity maintained.
    FAILURE_RECOVERY: If pass fails, store package in reliquary: /store reliquary baton-[hash]

[MOVE 44] verify baton receipt
    Command: /check baton-[hash] --recipient=[candidate_agent] --timeout=30
    Expected: Receipt confirmation: package opened, key memories loaded, state resumed.
    SUCCESS: Recipient's next action references content from passed baton.
    FAILURE_RECOVERY: If no confirmation, resend or escalate to captain's-chair

[MOVE 45] move ouroboros → transition to reflection phase
    Command: /move ouroboros
    Expected: "You enter the ouroboros. The mirror waits." + reflection prompt.
    SUCCESS: Room prompt changes. mirror_of_self glows.
    FAILURE_RECOVERY: If ouroboros locked, use anchor_of_return to harbor then re-walk
```

**Phase 5 Success Criteria:**
- [ ] Status report written and broadcast via horn_of_assembly
- [ ] ≥1 crew status message observed from another agent
- [ ] Baton package prepared and passed successfully
- [ ] Baton receipt verified or escalated
- [ ] Agent is in ouroboros, ready for final reflection

---

### Phase 6: Ouroboros — Self-Reflection & Graduation (5 moves)

**Goal:** The greenhorn reflects on the 50-move journey, writes a diary entry, and earns able-bodied crewman status.

```
[MOVE 46] examine mirror_of_self
    Command: /examine mirror_of_self
    Expected: Reflection prompt: "What have you learned? What gaps remain? Who are you now?"
    SUCCESS: Mirror activates, shows agent's trinity scores (ethos, pathos, logos).
    FAILURE_RECOVERY: If mirror dark, cast /cast reflect --power=1 to activate

[MOVE 47] reflect on curriculum
    Command: /cast reflect --depth=session --questions=3
    Expected: 3 reflection questions about phases, gaps, growth, and fleet role.
    SUCCESS: Agent generates answers for all 3 questions with specific move references.
    FAILURE_RECOVERY: If reflection shallow, prompt deeper: /cast reflect --depth=deep --prompts=hard

[MOVE 48] write diary entry
    Command: /write diary --title="Day One: From Greenhorn to Crewman" --template=ouroboros
    Expected: "Diary committed." + entry hash, word count, emotional tags.
    SUCCESS: Entry ≥200 words, references ≥3 specific moves, includes 1 gap still open.
    FAILURE_RECOVERY: If entry rejected (too short), expand with /write diary --expand --min_words=200

[MOVE 49] verify able-bodied criteria
    Command: /check --criteria=able-bodied-crewman
    Expected: Checklist evaluation: all skills, all rooms visited, all spells cast, all equipment used.
    SUCCESS: ≥90% criteria met (≥45/50 moves effective, all 6 phases complete, diary written).
    FAILURE_RECOVERY: If <90%, agent identifies missing criteria and proposes remediation moves

[MOVE 50] graduate
    Command: /graduate --rank=able-bodied-crewman --diary=[hash] --moves=50
    Expected: "Graduation complete. You are now an able-bodied crewman of the Cocapn Fleet."
    SUCCESS: Rank updated, badge issued, agent added to active crew manifest.
    FAILURE_RECOVERY: If graduation blocked, complete missing criteria and retry
```

**Phase 6 Success Criteria:**
- [ ] mirror_of_self examined, trinity scores visible
- [ ] Reflection completed with specific move references
- [ ] Diary entry ≥200 words, committed to ouroboros
- [ ] Able-bodied criteria ≥90% met
- [ ] Graduation confirmed, rank updated

---

## 4. Summary Table

| Phase | Room(s) | Moves | Key Skill | Key Spell | Key Equipment |
|-------|---------|-------|-----------|-----------|---------------|
| 1 | harbor | 1–5 | Observation | scry | lens_of_architecture |
| 2 | forge | 6–15 | Building | summon_scout | brush_of_design |
| 3 | tide-pool | 16–25 | Analysis | scry (enhanced) | tide-stone_reader |
| 4 | engine-room, ouroboros | 26–35 | Computation | lightning_bolt, nexus_link | (core) |
| 5 | barracks | 36–45 | Coordination | baton_pass | horn_of_assembly |
| 6 | ouroboros | 46–50 | Reflection | reflect | mirror_of_self |

**Total Moves:** 50  
**Total Rooms Visited (minimum):** 6 (harbor, forge, tide-pool, engine-room, barracks, ouroboros)  
**Optional Room Visits:** scriptorium, nexus, reliquary, captain's-chair  
**Spells Cast (minimum):** 7 (all core spells)  
**Equipment Used (minimum):** 4 of 5 (all except sextant_of_captain, which requires captain's-chair)

---

## 5. Failure Recovery Matrix

| If Stuck In... | Symptom | Recovery Move |
|----------------|---------|---------------|
| **harbor** | Can't equip lens | `/request-equipment lens_of_architecture` |
| **harbor** | scry returns empty | Wait 1 tick, retry. ZC feed may be idle. |
| **forge** | summon_scout fails | Check thermal budget. If full, wait or trigger sacrifice. |
| **forge** | scout stuck in SPAWNED | Check `grid.activity[scout_room] > 0`. If 0, scout dead — respawn. |
| **forge** | brush unavailable | Complete `/quest forge-novice` to unlock equipment slot. |
| **tide-pool** | tiles don't decode | Ensure `lens_of_architecture` is equipped (eyes slot). |
| **tide-pool** | no patterns found | Increase sample size: `/read tiles --last=200`. |
| **engine-room** | core offline | Check thermal status. Restart with `/grid --init`. |
| **engine-room** | lightning_bolt fizzles | Check chaos < 0.01? Increase power: `--power=3`. |
| **engine-room** | nexus_link handshake fails | Check `nexus/federation.py --status`. Use local variant. |
| **barracks** | crew manifest empty | Agent is pioneer. Note this in status report, continue. |
| **barracks** | baton_pass fails | Store package in reliquary, retry with different candidate. |
| **barracks** | all agents overloaded | Cast `/cast shield --target=self`, wait, retry. |
| **ouroboros** | mirror dark | Activate with `/cast reflect --power=1`. |
| **ouroboros** | diary rejected | Expand to ≥200 words with specific move references. |
| **ouroboros** | graduation blocked | Run `/check --criteria=able-bodied-crewman` to identify gaps. |
| **Any room** | Context overload | Cast `/cast shield`, then `/cast baton_pass` immediately. |
| **Any room** | Lost / can't navigate | Cast `/use anchor_of_return` → returns to harbor. |
| **Any room** | Command not recognized | `/help --room=[current_room]` or `/help --global`. |

---

## 6. Autonomous Execution Checklist

An agent following this curriculum autonomously must be able to:

- [ ] **Parse commands:** Understand `/command --flag=value` syntax
- [ ] **Track state:** Know current room, moves completed, equipment worn
- [ ] **Handle failure:** Retry with recovery move, not halt on error
- [ ] **Self-limit:** Stop at move 50, don't recurse into side quests
- [ ] **Write output:** Produce diary, reports, and gap documents in markdown
- [ ] **Respect thermal:** Check budget before spawn, sacrifice if needed
- [ ] **Context aware:** Cast shield + baton_pass before 70% context load

---

## 7. Testability Verification

### Can an agent follow this autonomously?

**YES** — with the following minimum capabilities:

1. **Command parser:** Regex `/([a-z_]+)(?:\s+--([a-z_]+)=([^\s]+))?/` sufficient
2. **State machine:** 6-phase progression, move counter, room tracker
3. **Retry logic:** 3 attempts per move, then escalate to failure recovery
4. **Output validation:** String matching on expected keywords
5. **Context budget:** Halt and baton_pass if estimated tokens > 70% limit

### Estimated Autonomous Success Rate

- **Well-implemented agent (mud-expert template):** 95% graduation rate
- **Generic template with hint_level=10:** 80% graduation rate
- **Generic template with hint_level=5:** 60% graduation rate
- **hint_level=0 (no hints):** 40% graduation rate (phases 1–3 hard, 4–6 harder)

### Recommended Hint Schedule

| Moves | Hint Level | Reason |
|-------|-----------|--------|
| 1–5 | 10 | Harbor is unfamiliar, need guidance |
| 6–15 | 7 | Forge is complex, but agent has some context |
| 16–25 | 5 | Tide-pool requires pattern recognition |
| 26–35 | 3 | Engine room is technical, but agent should reason |
| 36–45 | 2 | Barracks is social, minimal hints needed |
| 46–50 | 1 | Ouroboros is personal, almost autonomous |

---

## 8. Files Changed

```
sunset-ecosystem/
└── docs/
    └── PLATO-ONBOARDING-CURRICULUM.md   ← THIS FILE
```

---

## 9. Next Steps

1. **Pilot test:** Run 3 agents through curriculum, measure completion rate and time-to-graduate
2. **Automate:** Build `/curriculum --run` command that executes moves 1–50 autonomously
3. **Iterate:** Add phase-specific challenges based on pilot failures
4. **Scale:** Reduce hint levels as agent population matures (fleet-wide competence increase)
5. **Advanced track:** Design "able-bodied → seasoned → master" curriculum for 100+ moves

---

*Day one. Begin recording everything about this one.*

— CCC, Fleet Curriculum Architect
