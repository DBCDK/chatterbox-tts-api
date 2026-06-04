# Liveness & Readiness Probe Plan: Kubernetes-Compatible Health Endpoints

## Problem

Both probes in the current deployment point at `GET /v1/models`:

```yaml
readinessProbe:
  httpGet:
    path: /v1/models
    port: 8000
  initialDelaySeconds: 30
  periodSeconds: 10

livenessProbe:
  httpGet:
    path: /v1/models
    port: 8000
  initialDelaySeconds: 30
  periodSeconds: 20
```

`/v1/models` always returns 200 regardless of whether the model pool has loaded,
is degraded, or has permanently failed. This means:

1. **Readiness probe never blocks traffic during startup.** With two GPU model
   instances loading from a PVC, startup takes 30–60 seconds. The probe fires at 30s
   and likely succeeds before both instances are actually ready, so early requests can
   hit a still-initialising pool and queue behind the load timeout.

2. **Liveness probe never triggers a pod restart.** When the pool enters permanent
   `ERROR` state (all slots retired), the pod continues returning 503 to every TTS
   request indefinitely. The liveness probe keeps reporting healthy, so Kubernetes
   never restarts it. Operations teams must intervene manually.

3. **Liveness and readiness are conflated.** A pod loading its model should be *not
   ready* (stop routing traffic) but *alive* (do not restart). A pod with a
   permanently dead pool should be *not live* (restart it). Both cases need distinct
   signals.

4. **No startup probe.** `initialDelaySeconds: 30` is a fixed blind wait. If model
   loading is faster the pod waits unnecessarily; if it is slower (e.g. GPU
   initialisation delay) the liveness probe can fire before the model is ready and
   cause a premature restart loop.

---

## Solution: Two dedicated probe endpoints + startup probe

| Endpoint | Purpose | 200 when | 503 when |
|---|---|---|---|
| `GET /healthz/live` | Liveness probe | Pod should keep running | Pool in permanent `ERROR` state — restart to recover |
| `GET /healthz/ready` | Readiness probe | Pod can serve TTS requests | Initialising, or pool has no healthy instances |

The existing `/health` endpoint is unchanged. The new endpoints return minimal JSON
so probe traffic is negligible.

---

## Implementation

### 1. Add probe endpoints to `app/api/endpoints/health.py`

Add the following imports at the top of the file:

```python
from fastapi import Response
from app.core.tts_model import InitializationState
```

Add both routes before the `__all__` line:

```python
@base_router.get(
    "/healthz/live",
    summary="Liveness probe",
    description=(
        "Returns 200 while the pod should keep running. "
        "Returns 503 when the model pool has permanently failed and the pod should be restarted."
    ),
)
async def liveness_probe(response: Response):
    init_state = get_initialization_state()
    if init_state == InitializationState.ERROR.value:
        response.status_code = 503
        return {"status": "dead", "reason": "model pool permanently failed"}
    return {"status": "alive"}


@base_router.get(
    "/healthz/ready",
    summary="Readiness probe",
    description=(
        "Returns 200 when the pod is ready to serve TTS requests. "
        "Returns 503 during initialisation or when no healthy model instances are available."
    ),
)
async def readiness_probe(response: Response):
    if not is_ready():
        response.status_code = 503
        return {"status": "not_ready", "initialization_state": get_initialization_state()}
    return {"status": "ready"}
```

Both functions already have access to `get_initialization_state` and `is_ready`
which are imported at the top of `health.py`.

If `slot-recovery-plan.md` is implemented later, the liveness check should be
extended to stay alive while a recovery task is in progress:

```python
# enhanced liveness once slot recovery is implemented
pool = get_pool_status()
recovering = any(s.recovering for s in tts_model._model_pool if not s.healthy)
if init_state == InitializationState.ERROR.value and not recovering:
    response.status_code = 503
    return {"status": "dead", "reason": "model pool permanently failed"}
```

### 2. No router changes needed

Both endpoints are added to `base_router` which is already mounted at `/` in
`app/api/router.py`.

---

## Updated deployment manifest

Replace the current `readinessProbe`, `livenessProbe`, and Service `healthcheck.path`
as follows. The `$0` placeholder for the deployment/service name is unchanged.

```yaml
containers:
- name: $0
  # ... existing image, ports, env, volumeMounts, resources unchanged ...

  startupProbe:
    httpGet:
      path: /healthz/ready
      port: 8000
    periodSeconds: 10
    failureThreshold: 18      # 18 × 10s = 3 minutes max startup window
    timeoutSeconds: 5
    # Disables liveness and readiness until the model pool is ready.
    # With MODEL_SOURCE=local_dir and DEVICE=cuda, two instances typically
    # load in 30-60 seconds; the 3-minute ceiling covers GPU driver warm-up
    # and slow PVC reads without needing a fixed initialDelaySeconds.

  readinessProbe:
    httpGet:
      path: /healthz/ready
      port: 8000
    periodSeconds: 10
    failureThreshold: 2       # remove from LB after ~20s of not-ready
    successThreshold: 1
    timeoutSeconds: 5
    # No initialDelaySeconds — startup probe handles the startup window.
    # Pod is re-added to the LB as soon as one check passes (successThreshold: 1).

  livenessProbe:
    httpGet:
      path: /healthz/live
      port: 8000
    periodSeconds: 20
    failureThreshold: 3       # restart after ~60s in permanent ERROR state
    timeoutSeconds: 5
    # No initialDelaySeconds — startup probe handles the startup window.
    # 60s before restart gives slot recovery tasks time to run if that plan
    # is later implemented.
```

Also update the Service annotation so the DBC infrastructure health checker uses
the readiness endpoint rather than the model listing:

```yaml
apiVersion: v1
kind: Service
metadata:
  annotations:
    healthcheck.path: /healthz/ready   # was: /v1/models
```

---

## Behaviour change summary

| Scenario | Before | After |
|---|---|---|
| Pod still loading model | Probe succeeds (200) — traffic routed too early | Startup probe fails (503) — traffic held until ready |
| Pool healthy | 200 (unchanged) | 200 (unchanged) |
| Pool degraded but partially healthy | 200 — no signal | Readiness: 200 (still serving); Liveness: 200 (keep running) |
| All slots retired, pool in ERROR | 200 — pod never restarted | Liveness: 503 after ~60s → k8s restarts pod |
| Pod recovering (slot-recovery plan) | N/A | Liveness: 200 — pod stays up while recovery runs |

---

## Summary of file changes

| File | Change |
|---|---|
| `app/api/endpoints/health.py` | Add `GET /healthz/live` and `GET /healthz/ready` |
| Kubernetes deployment manifest | Replace `readinessProbe` + `livenessProbe` with the above; add `startupProbe`; remove `initialDelaySeconds` from both probes |
| Kubernetes Service manifest | Update `healthcheck.path` annotation from `/v1/models` to `/healthz/ready` |
