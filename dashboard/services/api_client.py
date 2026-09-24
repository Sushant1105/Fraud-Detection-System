"""
HTTP API client for communicating with the FastAPI Fraud Detection Service.
Strictly decoupled: Streamlit never imports ML models or preprocessors.
"""
import os
from typing import Dict, Any, Tuple, Optional
import requests

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000").rstrip("/")


class FraudApiClient:
    """Client for synchronous communication with FastAPI backend."""

    def __init__(self, base_url: str = API_BASE_URL):
        self.base_url = base_url

    def check_health(self, timeout: float = 3.0) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
        """
        Check health of FastAPI microservice.
        Returns: (is_online, health_data_dict, error_message)
        """
        url = f"{self.base_url}/health"
        try:
            resp = requests.get(url, timeout=timeout)
            if resp.status_code == 200:
                return True, resp.json(), None
            return False, None, f"Health check returned HTTP {resp.status_code}: {resp.text}"
        except requests.exceptions.ConnectionError:
            return False, None, f"Cannot connect to FastAPI at {self.base_url}. Service appears to be offline."
        except requests.exceptions.Timeout:
            return False, None, f"Health check timed out after {timeout} seconds."
        except Exception as exc:
            return False, None, f"Unexpected error during health check: {str(exc)}"

    def predict_transaction(
        self,
        payload: Dict[str, Any],
        timeout: float = 6.0
    ) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
        """
        Send raw transaction to POST /predict.
        Returns: (success, prediction_dict, error_message)
        """
        url = f"{self.base_url}/predict"
        try:
            resp = requests.post(url, json=payload, timeout=timeout)
            if resp.status_code == 200:
                return True, resp.json(), None
            elif resp.status_code == 422:
                return False, None, f"Input Validation Error (HTTP 422): {resp.text}"
            elif resp.status_code == 503:
                return False, None, "FastAPI Service Unavailable (HTTP 503): Model still initializing."
            else:
                return False, None, f"Prediction failed with HTTP {resp.status_code}: {resp.text}"
        except requests.exceptions.ConnectionError:
            return False, None, f"Connection Error: Unable to reach FastAPI backend at {self.base_url}."
        except requests.exceptions.Timeout:
            return False, None, f"Request Timeout: FastAPI did not respond within {timeout} seconds."
        except Exception as exc:
            return False, None, f"Unexpected error communicating with API: {str(exc)}"

    def get_transactions(
        self,
        limit: int = 50,
        offset: int = 0,
        timeout: float = 5.0,
    ) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
        """
        Fetch paginated transaction history from GET /transactions.
        Returns: (success, response_dict, error_message)
        """
        url = f"{self.base_url}/transactions"
        params = {"limit": limit, "offset": offset}
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            if resp.status_code == 200:
                return True, resp.json(), None
            return False, None, f"Failed to retrieve transactions (HTTP {resp.status_code}): {resp.text}"
        except requests.exceptions.ConnectionError:
            return False, None, f"Connection Error: Unable to reach FastAPI at {self.base_url}."
        except requests.exceptions.Timeout:
            return False, None, f"Request Timeout: FastAPI did not respond within {timeout} seconds."
        except Exception as exc:
            return False, None, f"Unexpected error fetching transactions: {str(exc)}"

    def get_transaction(
        self,
        transaction_id: str,
        timeout: float = 5.0,
    ) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
        """
        Fetch single transaction prediction record by transaction_id from GET /transactions/{transaction_id}.
        Returns: (success, response_dict, error_message)
        """
        url = f"{self.base_url}/transactions/{transaction_id}"
        try:
            resp = requests.get(url, timeout=timeout)
            if resp.status_code == 200:
                return True, resp.json(), None
            elif resp.status_code == 404:
                return False, None, f"Transaction '{transaction_id}' not found."
            return False, None, f"Failed to retrieve transaction (HTTP {resp.status_code}): {resp.text}"
        except requests.exceptions.ConnectionError:
            return False, None, f"Connection Error: Unable to reach FastAPI at {self.base_url}."
        except requests.exceptions.Timeout:
            return False, None, f"Request Timeout: FastAPI did not respond within {timeout} seconds."
        except Exception as exc:
            return False, None, f"Unexpected error fetching transaction '{transaction_id}': {str(exc)}"

    def get_stats(
        self,
        timeout: float = 5.0,
    ) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
        """
        Fetch aggregate scoring metrics from GET /stats.
        Returns: (success, response_dict, error_message)
        """
        url = f"{self.base_url}/stats"
        try:
            resp = requests.get(url, timeout=timeout)
            if resp.status_code == 200:
                return True, resp.json(), None
            return False, None, f"Failed to retrieve statistics (HTTP {resp.status_code}): {resp.text}"
        except requests.exceptions.ConnectionError:
            return False, None, f"Connection Error: Unable to reach FastAPI at {self.base_url}."
        except requests.exceptions.Timeout:
            return False, None, f"Request Timeout: FastAPI did not respond within {timeout} seconds."
        except Exception as exc:
            return False, None, f"Unexpected error fetching statistics: {str(exc)}"

