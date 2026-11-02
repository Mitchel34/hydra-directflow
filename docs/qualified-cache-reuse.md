# One reviewed reader-compatibility rule

The original `20200422_00Z` raw-cycle archive retains compact consolidated-metadata
404 receipts. The later reader expected an additional nested proof field. The
original producer and authenticated manifest instead preserve matching source
records and settled GET404 request evidence. The new branch requires that exact
legacy schema, receipt equality and one matching request; present malformed
nested proofs still fail. All twelve component receipts, coordinates, tile
decoding, numeric values and basin checks remain required.

Previously accepted reductions do not change under this extension. Re-downloading
their raw sources would repeat already qualified work. `cache_reuse.py` therefore
implements one exact producer/consumer rule: the qualified v3 derivation map with
reader `ed999095…a47084` may be consumed by the reviewed reader `adb2fe58…d5045`.
Every other derivation-code hash, source-row identity, qualified index, basin
weights and policy must match. Unknown versions are not implicitly compatible.
The full hashes are in the implementation, and the helper hash is bound in the
new corpus plan.

Under the existing cache lock, the adapter verifies the original committed
payloads and numerical schema, reference clock and source receipt. It returns
the **original cache key and original source receipt**, without writing a new
cache entry or changing any retained file. A separate per-use proof binds the
old COMMIT, old receipt, producer and consumer identities and compatibility rule.
That proof is recorded in the new chunk's `case-index.json`; its content hash
is covered by the immutable chunk COMMIT. Resume verifies both the recorded key
and the per-use proof.

Eligible original entries are preferred deterministically even if a current
version entry also exists. A selected original entry that is corrupt, linked or
nonresident stops the run; it is not silently replaced by a network read. A
missing qualified original can use the normal current reader/provider path.
Reuse invokes no raw source decoder, provider callback or raw-eviction callback.
Per-use load counts may repeat across month sessions and are not unique-cycle
counts. This is a narrow compatibility rule, not a general migration framework.

The old v3 plan, shards, cache entries and interruption receipts remain intact.
The new v4 plan rebuilds typed shards from qualified reductions and local USGS
sources. The completed v3 prefix must be checked for exact numeric/mask/clock
payload parity before continuing source staging. Updated derivation keys or
per-use provenance may change case-index bytes without changing model inputs.
No new scientific choice, protected access or source-acquisition scope is implied.
