"""
Live verification script for Phase 11 Database and Transaction History.
"""
import sys
import time
import requests
import json

BASE_URL = "http://127.0.0.1:8000"

# Legitimate sample
LEGIT_PAYLOAD = {
    "transaction_id": "tx-live-legit-001",
    "Time": 406.0,
    "Amount": 67.88,
    "V1": -2.3122, "V2": 1.9519, "V3": -1.6098, "V4": 3.9979,
    "V5": -0.5221, "V6": -1.4265, "V7": -2.5373, "V8": 1.3916,
    "V9": -2.7700, "V10": -2.7722, "V11": 3.2020, "V12": -2.8999,
    "V13": -0.5952, "V14": -4.2892, "V15": 0.3897, "V16": -1.1407,
    "V17": -2.8300, "V18": -0.0168, "V19": 0.4169, "V20": 0.1269,
    "V21": 0.5172, "V22": -0.0350, "V23": -0.4652, "V24": 0.3201,
    "V25": 0.0445, "V26": 0.1778, "V27": 0.2611, "V28": -0.1432
}

# High-risk / fraud sample
FRAUD_PAYLOAD = {
    "transaction_id": "tx-live-fraud-002",
    "Time": 472.0,
    "Amount": 529.00,
    "V1": -3.0435, "V2": -3.1573, "V3": 1.0884, "V4": 2.2886,
    "V5": 1.3598, "V6": -1.0648, "V7": 0.3255, "V8": -0.0677,
    "V9": -0.2709, "V10": -0.8385, "V11": -0.4145, "V12": -0.5031,
    "V13": 0.6765, "V14": -1.6920, "V15": 2.0006, "V16": 0.6667,
    "V17": 0.5997, "V18": 1.7253, "V19": 0.2833, "V20": 2.1023,
    "V21": 0.6616, "V22": 0.4354, "V23": 1.3759, "V24": -0.2938,
    "V25": 0.2797, "V26": -0.1453, "V27": -0.2527, "V28": 0.0357
}

def run_checks():
    print("=" * 60)
    print("PHASE 11 LIVE VERIFICATION SEQUENCE")
    print("=" * 60)

    # 1. Health check
    print("\n1. Verifying /health ...")
    h_resp = requests.get(f"{BASE_URL}/health", timeout=5)
    assert h_resp.status_code == 200, f"Health check failed: {h_resp.text}"
    health_data = h_resp.json()
    print(f"   [PASS] /health response: status={health_data['status']}, model={health_data['model_algorithm']}, threshold={health_data['decision_threshold']}")

    # 2. Docs check
    print("\n2. Verifying /docs ...")
    docs_resp = requests.get(f"{BASE_URL}/docs", timeout=5)
    assert docs_resp.status_code == 200, f"/docs failed: {docs_resp.status_code}"
    print(f"   [PASS] /docs returned HTTP 200 (OpenAPI/Swagger UI)")

    # 3. Initial stats check
    print("\n3. Verifying /stats (initial) ...")
    stats_resp = requests.get(f"{BASE_URL}/stats", timeout=5)
    assert stats_resp.status_code == 200
    initial_stats = stats_resp.json()
    print(f"   [PASS] Current stats: total={initial_stats['total_transactions']}, fraud={initial_stats['fraud_count']}, legit={initial_stats['legitimate_count']}, rate={initial_stats['fraud_rate']}%")

    # 4. Predict transaction 1 (legitimate)
    print("\n4. Submitting transaction 1 to /predict ...")
    p1_resp = requests.post(f"{BASE_URL}/predict", json=LEGIT_PAYLOAD, timeout=10)
    assert p1_resp.status_code == 200, f"/predict failed: {p1_resp.text}"
    p1_data = p1_resp.json()
    print(f"   [PASS] Transaction 1 scored: tx_id={p1_data['transaction_id']}, prob={p1_data['fraud_probability']:.4f}, decision={p1_data['decision']}, risk={p1_data['risk_level']}")
    print(f"          Top SHAP driver: {p1_data['top_contributing_features'][0]['feature']} (shap={p1_data['top_contributing_features'][0]['shap_value']:.4f})")

    # 5. Predict transaction 2 (fraud sample)
    print("\n5. Submitting transaction 2 to /predict ...")
    p2_resp = requests.post(f"{BASE_URL}/predict", json=FRAUD_PAYLOAD, timeout=10)
    assert p2_resp.status_code == 200, f"/predict failed: {p2_resp.text}"
    p2_data = p2_resp.json()
    print(f"   [PASS] Transaction 2 scored: tx_id={p2_data['transaction_id']}, prob={p2_data['fraud_probability']:.4f}, decision={p2_data['decision']}, risk={p2_data['risk_level']}")
    print(f"          Top SHAP driver: {p2_data['top_contributing_features'][0]['feature']} (shap={p2_data['top_contributing_features'][0]['shap_value']:.4f})")

    # 6. Verify /transactions contains the predictions
    print("\n6. Verifying /transactions ...")
    tx_resp = requests.get(f"{BASE_URL}/transactions?limit=10", timeout=5)
    assert tx_resp.status_code == 200
    tx_data = tx_resp.json()
    assert tx_data["count"] >= 2
    tx_ids = [t["transaction_id"] for t in tx_data["transactions"]]
    assert "tx-live-legit-001" in tx_ids
    assert "tx-live-fraud-002" in tx_ids
    print(f"   [PASS] /transactions returned {tx_data['count']} items (total in DB: {tx_data['total']})")
    print(f"          Newest transaction ID: {tx_data['transactions'][0]['transaction_id']}")

    # 7. Verify /transactions/{transaction_id}
    print("\n7. Verifying /transactions/{transaction_id} ...")
    single_resp = requests.get(f"{BASE_URL}/transactions/tx-live-legit-001", timeout=5)
    assert single_resp.status_code == 200
    single_data = single_resp.json()
    assert single_data["transaction_id"] == "tx-live-legit-001"
    assert single_data["amount"] == 67.88
    assert single_data["decision"] == p1_data["decision"]
    assert len(single_data["top_contributing_features"]) == 5
    print(f"   [PASS] /transactions/tx-live-legit-001 returned matching record with 5 SHAP features preserved")

    # 8. Verify /transactions/{transaction_id} 404
    print("\n8. Verifying 404 on nonexistent transaction ...")
    nf_resp = requests.get(f"{BASE_URL}/transactions/tx-nonexistent-9999", timeout=5)
    assert nf_resp.status_code == 404
    print(f"   [PASS] Nonexistent ID returned HTTP 404: {nf_resp.json()['detail']}")

    # 9. Verify duplicate transaction_id behavior (returns latest)
    print("\n9. Testing duplicate transaction_id behavior ...")
    dup_payload = dict(LEGIT_PAYLOAD)
    dup_payload["Amount"] = 888.88
    dup_resp = requests.post(f"{BASE_URL}/predict", json=dup_payload, timeout=10)
    assert dup_resp.status_code == 200

    dup_lookup = requests.get(f"{BASE_URL}/transactions/tx-live-legit-001", timeout=5)
    assert dup_lookup.status_code == 200
    dup_data = dup_lookup.json()
    assert dup_data["amount"] == 888.88
    print(f"   [PASS] Duplicate transaction_id returned latest submission with amount: €{dup_data['amount']:.2f}")

    # 10. Verify /stats
    print("\n10. Verifying /stats calculations ...")
    final_stats_resp = requests.get(f"{BASE_URL}/stats", timeout=5)
    assert final_stats_resp.status_code == 200
    f_stats = final_stats_resp.json()
    assert f_stats["total_transactions"] == f_stats["fraud_count"] + f_stats["legitimate_count"]
    expected_rate = round(f_stats["fraud_count"] / f_stats["total_transactions"] * 100.0, 2)
    assert f_stats["fraud_rate"] == expected_rate
    print(f"   [PASS] /stats accurate: total={f_stats['total_transactions']}, fraud={f_stats['fraud_count']}, legit={f_stats['legitimate_count']}, rate={f_stats['fraud_rate']}%")

    print("\n" + "=" * 60)
    print("ALL API VERIFICATION STEPS PASSED SUCCESSFULLY")
    print("=" * 60)

if __name__ == "__main__":
    run_checks()
