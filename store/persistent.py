"""
Lightweight JSON-backed persistence layer for caching tool lookups,
filing metadata, and query history.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


class PersistentStore:
    """Simple file-based persistence for application data and caches."""

    def __init__(self, base_dir: str = "app_data"):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

        self._locks: Dict[str, threading.Lock] = {}

        # File paths
        self._paths = {
            "state": self._path("state.json"),
            "filings": self._path("filings.json"),
            "director_snippets": self._path("director_snippets.json"),
            "tables": self._path("tables.json"),
            "query_history": self._path("query_history.json"),
            "serp_cache": self._path("serp_cache.json"),
            "linkedin_cache": self._path("linkedin_cache.json"),
        }

        # Ensure files exist with sensible defaults
        self._initialize_file("state", {"next_filing_id": 1})
        self._initialize_file("filings", [])
        self._initialize_file("director_snippets", {})
        self._initialize_file("tables", {})
        self._initialize_file("query_history", [])
        self._initialize_file("serp_cache", {})
        self._initialize_file("linkedin_cache", {})

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------
    def _path(self, filename: str) -> str:
        return os.path.join(self.base_dir, filename)

    def _get_lock(self, key: str) -> threading.Lock:
        if key not in self._locks:
            self._locks[key] = threading.Lock()
        return self._locks[key]

    def _initialize_file(self, key: str, default: Any) -> None:
        path = self._paths[key]
        if not os.path.exists(path):
            self._write_json(path, default)

    def _read_json(self, path: str) -> Any:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _write_json(self, path: str, data: Any) -> None:
        tmp_path = f"{path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, path)

    # ------------------------------------------------------------------
    # Filing management
    # ------------------------------------------------------------------
    def _get_next_filing_id(self) -> int:
        with self._get_lock("state"):
            state = self._read_json(self._paths["state"])
            next_id = state.get("next_filing_id", 1)
            state["next_filing_id"] = next_id + 1
            self._write_json(self._paths["state"], state)
        return next_id

    def add_filing(self, name: str, urls: List[str]) -> Dict[str, Any]:
        filing_id = self._get_next_filing_id()
        entry = {
            "id": filing_id,
            "name": name,
            "urls": urls,
            "created_at": datetime.utcnow().isoformat() + "Z",
        }

        with self._get_lock("filings"):
            filings = self._read_json(self._paths["filings"])
            filings.append(entry)
            self._write_json(self._paths["filings"], filings)

        return entry

    def list_filings(self) -> List[Dict[str, Any]]:
        with self._get_lock("filings"):
            return self._read_json(self._paths["filings"])

    def get_name_id_map(self) -> Dict[str, int]:
        return {filing["name"]: filing["id"] for filing in self.list_filings()}

    def get_filing(self, filing_id: int) -> Optional[Dict[str, Any]]:
        filings = self.list_filings()
        for filing in filings:
            if filing["id"] == filing_id:
                return filing
        return None

    def resolve_filing_id(self, filing_name: str) -> Optional[int]:
        filings = self.list_filings()
        for filing in filings:
            if filing["name"] == filing_name:
                return filing["id"]
        return None

    # ------------------------------------------------------------------
    # Director snippets
    # ------------------------------------------------------------------
    def save_director_snippets(self, filing_id: int, snippets: Dict[str, str]) -> None:
        with self._get_lock("director_snippets"):
            data = self._read_json(self._paths["director_snippets"])
            data[str(filing_id)] = snippets
            self._write_json(self._paths["director_snippets"], data)

    def get_director_snippets(self, filing_id: Optional[int] = None) -> Dict[str, str]:
        with self._get_lock("director_snippets"):
            data = self._read_json(self._paths["director_snippets"])

        if filing_id is None:
            merged: Dict[str, str] = {}
            for snippets in data.values():
                merged.update(snippets)
            return merged

        return data.get(str(filing_id), {})

    # ------------------------------------------------------------------
    # Table storage
    # ------------------------------------------------------------------
    def save_tables(self, tables: List[Tuple[int, str, str]]) -> None:
        """Persist table HTML by table_id."""
        if not tables:
            return

        with self._get_lock("tables"):
            stored_tables = self._read_json(self._paths["tables"])
            for filing_id, table_id, html in tables:
                stored_tables[table_id] = {
                    "filling_id": filing_id,
                    "content": html,
                    "saved_at": datetime.utcnow().isoformat() + "Z",
                }
            self._write_json(self._paths["tables"], stored_tables)

    def get_table_by_id(self, table_id: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        with self._get_lock("tables"):
            tables = self._read_json(self._paths["tables"])
            table_entry = tables.get(table_id)

        if not table_entry:
            return None

        metadata = {
            "filling_id": table_entry.get("filling_id"),
            "saved_at": table_entry.get("saved_at"),
        }
        return table_entry.get("content", ""), metadata

    # ------------------------------------------------------------------
    # Query history
    # ------------------------------------------------------------------
    def add_query_history(self, query: str, answer: str, mode: str, filing_ids: List[int]) -> None:
        entry = {
            "query": query,
            "answer": answer,
            "mode": mode,
            "filing_ids": filing_ids,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
        with self._get_lock("query_history"):
            history = self._read_json(self._paths["query_history"])
            history.insert(0, entry)
            self._write_json(self._paths["query_history"], history[:100])

    def get_query_history(self) -> List[Dict[str, Any]]:
        with self._get_lock("query_history"):
            return self._read_json(self._paths["query_history"])

    # ------------------------------------------------------------------
    # SerpAPI caches
    # ------------------------------------------------------------------
    def get_director_linkedin_cache(self, key: str) -> Optional[str]:
        with self._get_lock("serp_cache"):
            cache = self._read_json(self._paths["serp_cache"])
            return cache.get(key)

    def set_director_linkedin_cache(self, key: str, value: str) -> None:
        with self._get_lock("serp_cache"):
            cache = self._read_json(self._paths["serp_cache"])
            cache[key] = value
            self._write_json(self._paths["serp_cache"], cache)

    # ------------------------------------------------------------------
    # LinkedIn scraper cache
    # ------------------------------------------------------------------
    def get_linkedin_scraper_cache(self, url: str) -> Optional[str]:
        with self._get_lock("linkedin_cache"):
            cache = self._read_json(self._paths["linkedin_cache"])
            return cache.get(url)

    def set_linkedin_scraper_cache(self, url: str, value: str) -> None:
        with self._get_lock("linkedin_cache"):
            cache = self._read_json(self._paths["linkedin_cache"])
            cache[url] = value
            self._write_json(self._paths["linkedin_cache"], cache)
