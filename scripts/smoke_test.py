import json
import os
import sys
import urllib.error
import urllib.request
import uuid

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8001").rstrip("/")
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:8080").rstrip("/")


def check(label, condition, detail=""):
    if condition:
        print(f"PASS {label}")
    else:
        print(f"FAIL {label}" + (f": {detail}" if detail else ""))
        sys.exit(1)


def request(method, url, body=None, token=None):
    data = None
    headers = {}
    if body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status, resp.read(), resp.headers
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read(), exc.headers


def build_payload(schema):
    features = {}
    for field in schema["raw_fields"]:
        allowed = field.get("allowed_values")
        if allowed:
            features[field["name"]] = allowed[-1]
            continue
        value = (field["train_min"] + field["train_max"]) / 2
        if field["type"] == "int":
            value = int(round(value))
        features[field["name"]] = value
    return {"features": features}


def main():
    status, _, _ = request("GET", f"{BACKEND_URL}/health")
    check("backend /health returns 200", status == 200, f"status={status}")

    email = f"smoke-{uuid.uuid4().hex}@example.com"
    password = uuid.uuid4().hex + uuid.uuid4().hex
    status, body, _ = request(
        "POST", f"{BACKEND_URL}/auth/register", {"email": email, "password": password}
    )
    check("register returns 201", status == 201, f"status={status} body={body[:200]}")

    status, body, _ = request(
        "POST", f"{BACKEND_URL}/auth/login", {"email": email, "password": password}
    )
    check("login returns 200", status == 200, f"status={status}")
    token = json.loads(body)["access_token"]

    schemas = {}
    for disease in ("cvd", "diabetes", "alzheimers", "liver"):
        status, body, _ = request("GET", f"{BACKEND_URL}/schema/{disease}")
        check(f"GET /schema/{disease} returns 200", status == 200, f"status={status}")
        schemas[disease] = json.loads(body)

    for disease in ("cvd", "diabetes", "alzheimers", "liver"):
        payload = build_payload(schemas[disease])
        status, body, _ = request(
            "POST", f"{BACKEND_URL}/compare/{disease}", payload, token
        )
        data = json.loads(body) if body else {}
        ml = data.get("ml", {})
        dl = data.get("dl", {})
        ok = (
            status == 200
            and isinstance(ml.get("risk_score"), (int, float))
            and isinstance(dl.get("risk_score"), (int, float))
            and 0 <= ml["risk_score"] <= 1
            and 0 <= dl["risk_score"] <= 1
            and bool(ml.get("risk_level"))
            and bool(dl.get("risk_level"))
            and bool(data.get("comparison"))
        )
        check(
            f"POST /compare/{disease} returns 200 with ML and DL scores",
            ok,
            f"status={status} body={body[:200]}",
        )

    famhist = schemas["cvd"].get("family_history_field", "famhist")
    incomplete = build_payload(schemas["cvd"])
    incomplete["features"].pop(famhist, None)
    status, body, _ = request(
        "POST", f"{BACKEND_URL}/compare/cvd", incomplete, token
    )
    check("missing family-history field returns 422", status == 422, f"status={status}")

    status, body, _ = request("GET", f"{BACKEND_URL}/history", token=token)
    records = json.loads(body) if body else []
    check(
        "history has exactly 4 records",
        status == 200 and len(records) == 4,
        f"status={status} count={len(records)}",
    )
    leaks = [key for record in records for key in record if "feature" in key.lower()]
    check("history stores no input features", not leaks, f"keys={leaks}")

    profile_data = {"shared.age": 40, "cvd.famhist": "Present"}
    status, _, _ = request("PUT", f"{BACKEND_URL}/profile", {"data": profile_data}, token)
    check("PUT /profile returns 200", status == 200, f"status={status}")

    status, body, _ = request("GET", f"{BACKEND_URL}/profile", token=token)
    saved = json.loads(body) if body else {}
    check(
        "GET /profile returns saved data",
        status == 200 and saved.get("data") == profile_data,
        f"status={status}",
    )

    status, _, _ = request("DELETE", f"{BACKEND_URL}/profile", token=token)
    check("DELETE /profile returns 204", status == 204, f"status={status}")

    status, body, _ = request("GET", f"{BACKEND_URL}/profile", token=token)
    cleared = json.loads(body) if body else {}
    check(
        "GET /profile after DELETE is empty",
        status == 200 and cleared.get("data") == {},
        f"status={status}",
    )

    status, _, headers = request("GET", f"{FRONTEND_URL}/")
    check(
        "frontend / returns HTML 200",
        status == 200 and "text/html" in headers.get("Content-Type", ""),
        f"status={status}",
    )
    status, _, _ = request("GET", f"{FRONTEND_URL}/history")
    check("frontend /history SPA fallback returns 200", status == 200, f"status={status}")


if __name__ == "__main__":
    main()
