"""
export_issues.py
----------------
Connects to a MySQL database, identifies qualifying projects using the
canonical filtering query, then exports each project's issues to a
separate CSV file under data_to_train_on/.

Filtering rules (kept consistent in both stages):
  - Story_Point IN (1, 2, 3, 5, 8)
  - Per-project: COUNT(*) >= 500 AND COUNT(DISTINCT Story_Point) >= 4

pip install:
    mysql-connector-python pandas python-dotenv

Usage:
    python export_issues.py
"""

import os
import re
import sys

import mysql.connector
from mysql.connector import Error
import pandas as pd
from dotenv import load_dotenv

# Load credentials from .env (file must sit in the same directory as this script)
load_dotenv()


# ---------------------------------------------------------------------------
# Configuration — set values in .env, not here
# ---------------------------------------------------------------------------
DB_CONFIG = {
    "host":            os.getenv("DB_HOST", "localhost"),
    "port":            int(os.getenv("DB_PORT", 3306)),
    "user":            os.getenv("DB_USER"),
    "password":        os.getenv("DB_PASSWORD"),
    "database":        os.getenv("DB_NAME"),
    "connect_timeout": 10,
}

# Fail fast if required credentials are missing from .env
_missing = [k for k, v in DB_CONFIG.items() if k != "connect_timeout" and v is None]
if _missing:
    print(f"[ERROR] Missing required .env variable(s): {', '.join(f'DB_{k.upper()}' for k in _missing)}")
    sys.exit(1)

OUTPUT_DIR = "data_to_train_on"

# The exact Fibonacci values treated as valid story points
VALID_STORY_POINTS = (1, 2, 3, 5, 8)

# Project-level thresholds (must match the canonical query)
MIN_ISSUES = 500
MIN_CLASSES = 4


# ---------------------------------------------------------------------------
# SQL queries
# ---------------------------------------------------------------------------

# Step 1: Identify qualifying projects (canonical query — do not change)
PROJECT_SUMMARY_QUERY = """
SELECT
    p.Name AS Project_Name,
    COUNT(*) AS Total_Issues,
    COUNT(DISTINCT i.Story_Point) AS Num_Classes
FROM Issue i
JOIN Project p ON i.Project_ID = p.ID
WHERE i.Story_Point IN (1, 2, 3, 5, 8)
GROUP BY p.Name
HAVING COUNT(*) >= 500
  AND COUNT(DISTINCT i.Story_Point) >= 4
ORDER BY Total_Issues DESC;
"""

# Step 2: Fetch all issues for a single qualifying project.
# Uses the same WHERE conditions as the canonical query.
ISSUE_QUERY = """
SELECT
    i.Issue_Key,
    i.Title,
    i.Description,
    i.Story_Point,
    i.Type,
    i.Priority
FROM Issue i
JOIN Project p ON i.Project_ID = p.ID
WHERE p.Name = %s
  AND i.Story_Point IN (1, 2, 3, 5, 8);
"""


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def connect(config: dict) -> mysql.connector.MySQLConnection:
    """Open and return a MySQL connection; exit on failure."""
    try:
        conn = mysql.connector.connect(**config)
        if conn.is_connected():
            print(f"[DB] Connected - MySQL server version {conn.server_info}")
            return conn
    except Error as exc:
        print(f"[ERROR] Could not connect to MySQL: {exc}")
        sys.exit(1)


def fetch_qualifying_projects(conn) -> list[dict]:
    """
    Run the canonical project-summary query and return a list of
    {Project_Name, Total_Issues, Num_Classes} dicts.
    """
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute(PROJECT_SUMMARY_QUERY)
        rows = cursor.fetchall()
        print(f"[DB] Found {len(rows)} qualifying project(s).")
        return rows
    except Error as exc:
        print(f"[ERROR] Failed to fetch project summary: {exc}")
        sys.exit(1)
    finally:
        cursor.close()


def fetch_project_issues(conn, project_name: str) -> pd.DataFrame:
    """
    Fetch all qualifying issues for *project_name* and return a DataFrame.
    Applies the same Story_Point filter as the canonical query.
    """
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute(ISSUE_QUERY, (project_name,))
        rows = cursor.fetchall()
        return pd.DataFrame(rows)
    except Error as exc:
        print(f"[ERROR] Failed to fetch issues for '{project_name}': {exc}")
        return pd.DataFrame()
    finally:
        cursor.close()


# ---------------------------------------------------------------------------
# File-system helpers
# ---------------------------------------------------------------------------

def sanitize_filename(name: str) -> str:
    """
    Replace characters that are illegal in file names on Windows/Linux/macOS
    with an underscore, then strip leading/trailing whitespace and dots.
    """
    safe = re.sub(r'[\\/:*?"<>|]', "_", name)
    safe = re.sub(r"[\s_]+", "_", safe).strip("_. ")
    return safe or "project"


def ensure_output_dir(path: str) -> None:
    """Create the output directory if it does not already exist."""
    os.makedirs(path, exist_ok=True)
    print(f"[IO] Output directory: {os.path.abspath(path)}")


# ---------------------------------------------------------------------------
# Export logic
# ---------------------------------------------------------------------------

def export_project(conn, project_info: dict, output_dir: str) -> None:
    """
    Fetch issues for one project and write them to a UTF-8 CSV file.

    Args:
        conn:         Active MySQL connection.
        project_info: Row from the project-summary query.
        output_dir:   Destination folder for CSV files.
    """
    project_name = project_info["Project_Name"]
    total_expected = project_info["Total_Issues"]
    num_classes = project_info["Num_Classes"]

    print(
        f"\n[EXPORT] Project: '{project_name}' "
        f"| Expected issues: {total_expected} | Classes: {num_classes}"
    )

    df = fetch_project_issues(conn, project_name)

    if df.empty:
        print(f"  [WARN] No issues returned for '{project_name}' - skipping.")
        return

    # Defensive check: verify Story_Point values are within the valid set.
    # Guards against NULL coerced to 0 by the MySQL driver.
    invalid_mask = ~df["Story_Point"].isin(VALID_STORY_POINTS)
    if invalid_mask.any():
        dropped = invalid_mask.sum()
        print(f"  [WARN] Dropping {dropped} row(s) with unexpected Story_Point values.")
        df = df[~invalid_mask].reset_index(drop=True)

    safe_name = sanitize_filename(project_name)
    file_path = os.path.join(output_dir, f"{safe_name}.csv")

    # utf-8-sig writes a BOM so the file opens correctly in Excel if needed
    df.to_csv(file_path, index=False, encoding="utf-8-sig")

    dist = df["Story_Point"].value_counts().sort_index().to_dict()
    print(f"  [OK] Wrote {len(df)} issues -> {file_path}")
    print(f"       Story_Point distribution: {dist}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    ensure_output_dir(OUTPUT_DIR)

    conn = connect(DB_CONFIG)
    try:
        projects = fetch_qualifying_projects(conn)

        if not projects:
            print("[WARN] No projects satisfy the filtering criteria. Nothing exported.")
            return

        print("\nQualifying projects:")
        for i, p in enumerate(projects, 1):
            print(
                f"  {i:>2}. {p['Project_Name']}"
                f" - {p['Total_Issues']} issues, {p['Num_Classes']} classes"
            )

        for project_info in projects:
            export_project(conn, project_info, OUTPUT_DIR)

    finally:
        if conn.is_connected():
            conn.close()
            print("\n[DB] Connection closed.")

    print(f"\n[DONE] All projects exported to '{OUTPUT_DIR}/'.")


if __name__ == "__main__":
    main()
