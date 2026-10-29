# First observability increment

`RunJournal` writes an actual-time event log, a current state file, finite scalar
metrics in Prometheus text format and artifact digests. It performs no network
requests and starts no service. Keep journals under ignored `runs/`.

```python
from hydra_directflow.telemetry import RunJournal

run = RunJournal("runs/example", {"stage": "development_smoke"},
                 max_wall_seconds=300, max_rss_bytes=4 * 2**30)
try:
    with run.stage("load"):
        pass  # bounded data loading
    run.metric("valid_cases", 3)
    run.finish()
except BaseException:
    if run.status == "RUNNING":
        run.finish("FAILED")
    raise
```

This is cooperative instrumentation. `check()` must run at stage/update
boundaries; it is not an OS memory limit or a substitute for an external watchdog.
A hard-killed process can retain RUNNING state with a stale heartbeat. Treat that
as an unknown outcome until the process/artifacts are inspected. A resume gets a
new explicit attempt and references its parent's checkpoint evidence.

Publish only the current attempt's `metrics.prom` through a textfile collector,
or add a bounded exporter later. Multiple unlabeled historical files must not be
scraped as if they were one current job. Keep individual case IDs and unrestricted
run IDs in logs/artifacts rather than metric labels.

Suggested future dashboards cover data progress/exclusions, loader versus compute
time, train/validation metrics, memory/disk/accelerator measurements, heartbeat,
checkpoint age and resource-hours. Protected assessment skill stays disabled
until the final evaluation phase. Budget checks and owned-job shutdown are
separate controls; a dashboard alert does not enforce a spending limit.

MLflow integration and Grafana/OpenTelemetry deployment remain later increments.
No hosted monitoring account, listener or public dashboard is configured here.
