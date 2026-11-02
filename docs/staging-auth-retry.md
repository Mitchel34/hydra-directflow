# Bounded source-token refresh

A cached CLI token can already be near expiry when acquired. On the first HTTP401 in one object transfer, the provider may clear only its in-memory token and retry once through the normal budgeted loop. The next attempt is durably precharged before the unchanged `gcloud auth print-access-token` command runs again. There is no login, account/configuration change, IAM mutation or alternate credential route.

The refresh consumes an existing attempt slot; it does not add one beyond the three-attempt maximum. A repeated401 stops. HTTP403 never refreshes or retries. Existing transient-response handling shares the same attempt limit, and a depleted persistent budget prevents the extra GET. Failed reservations remain charged.

The existing `error_type` column records a portable label such as `HTTPError_401` or `HTTPError_403`; the database schema is unchanged. Error bodies, URLs, authorization headers and tokens are neither read for diagnostics nor logged. Tests use fake HTTP responses and a mocked credential command, with no network or live authentication.
