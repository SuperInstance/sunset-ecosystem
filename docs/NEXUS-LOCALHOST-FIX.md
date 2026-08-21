# Federated Nexus — `localhost` Fix Spec

## Problem

The Federated Nexus (port 4047) was using a hardcoded ``localhost`` address in its
federation endpoint configuration.  This caused ``ConnectionRefusedError`` at
``nexus/federation.py:203`` (or equivalent line) whenever a remote fleet node
tried to register, because ``localhost`` resolves to the **loopback interface of
the caller**, not the actual nexus host.

## Where the code lived

The federation logic was **not present** in ``SuperInstance/sunset-ecosystem`` at
the time of this fix.  CCC's audit (MEMORY.md, 2026-04-22) indicates the running
nexus code previously lived on the **Oracle1 server** (``<BOAT_IP>:4047``).

This PR brings the module into the sunset-ecosystem repo with the correct
address baked in as the default.

## Fix Applied

| Before | After |
|--------|-------|
| ``localhost`` (implicit or explicit) | ``<BOAT_IP>`` (Cocapn fleet nexus) |

- ``nexus/federation.py`` defines ``DEFAULT_NEXUS_IP = "<BOAT_IP>"``.
- ``FederationEndpoint`` **rejects** any host that resolves to a loopback address
  (``127.*``, ``::1``, or ``localhost``) at instantiation time, raising
  ``InvalidEndpointError``.
- ``FederatedNexus.from_defaults()`` factory uses the fleet IP out of the box.

## Tests

``tests/test_nexus_federation.py`` covers:

1. ✅ Federation URL uses correct IP, not ``localhost``.
2. ✅ Registration heartbeat POST targets the correct endpoint.
3. ✅ ``localhost``, ``127.0.0.1``, and any loopback-resolving alias raise
   ``InvalidEndpointError`` on endpoint construction.
4. ✅ ``FederatedNexus.from_defaults()`` defaults to the fleet IP.

## Roll-out

1. Merge this PR.
2. Deploy ``nexus/`` to Oracle1 (or whichever host runs the nexus service).
3. Verify remote fleet nodes can ``register()`` without
   ``ConnectionRefusedError``.

## Files Changed

- ``nexus/__init__.py`` — new module init
- ``nexus/federation.py`` — new client + validation
- ``tests/test_nexus_federation.py`` — full test suite
- ``docs/NEXUS-LOCALHOST-FIX.md`` — this document
