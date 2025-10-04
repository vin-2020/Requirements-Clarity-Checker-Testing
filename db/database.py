import sqlite3
import os
from datetime import datetime
from typing import List, Tuple, Optional

# Use ONE consistent DB file
DB_PATH = os.path.join(os.path.dirname(__file__), "reqcheck.db")


def get_conn() -> sqlite3.Connection:
    """
    Open a DB connection with foreign key support enabled.
    SQLite requires PRAGMA per-connection.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db():
    """Initializes the database and creates tables if they don't exist."""
    conn = get_conn()
    cursor = conn.cursor()

    # Projects
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS projects (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            created_at TEXT NOT NULL
        )
    """)

    # Documents (cascade on project delete)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id INTEGER NOT NULL,
            file_name TEXT NOT NULL,
            version INTEGER NOT NULL,
            uploaded_at TEXT NOT NULL,
            clarity_score INTEGER,
            FOREIGN KEY (project_id) REFERENCES projects (id) ON DELETE CASCADE
        )
    """)

    # Requirements (cascade on document delete)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS requirements (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id INTEGER NOT NULL,
            req_id_string TEXT NOT NULL,
            req_text TEXT NOT NULL,
            FOREIGN KEY (document_id) REFERENCES documents (id) ON DELETE CASCADE
        )
    """)

    conn.commit()
    conn.close()


def add_project(project_name: str) -> str:
    """Adds a new project to the database."""
    conn = get_conn()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO projects (name, created_at) VALUES (?, ?)",
            (project_name, datetime.now().isoformat()),
        )
        conn.commit()
        return "Project added successfully."
    except sqlite3.IntegrityError:
        return "Project name already exists."
    finally:
        conn.close()


def get_all_projects() -> List[Tuple[int, str]]:
    """Retrieves (id, name) for all projects."""
    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT id, name FROM projects ORDER BY name")
    projects = cursor.fetchall()
    conn.close()
    return projects


def delete_project(project_id: int) -> str:
    """
    Deletes a project. Because FKs have ON DELETE CASCADE and PRAGMA is ON,
    related documents and requirements are removed automatically.
    """
    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM projects WHERE id = ?", (project_id,))
    conn.commit()
    conn.close()
    return "Project deleted successfully."


# -------------------
# NEW CORE FUNCTIONS
# -------------------

def add_document(project_id: int, file_name: str, version: int, score: Optional[int]) -> int:
    """Adds a document record and returns its ID."""
    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO documents (project_id, file_name, version, uploaded_at, clarity_score)
        VALUES (?, ?, ?, ?, ?)
        """,
        (project_id, file_name, version, datetime.now().isoformat(), score),
    )
    doc_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return doc_id


def add_requirements(doc_id: int, requirements: List[Tuple[str, str]]) -> None:
    """Adds a list of requirements to a document."""
    if not requirements:
        return
    conn = get_conn()
    cursor = conn.cursor()
    data = [(doc_id, rid, rtext) for rid, rtext in requirements]
    cursor.executemany(
        "INSERT INTO requirements (document_id, req_id_string, req_text) VALUES (?, ?, ?)",
        data,
    )
    conn.commit()
    conn.close()


def get_documents_for_project(project_id: int) -> List[Tuple[int, str, int, str, Optional[int]]]:
    """Returns (id, file_name, version, uploaded_at, clarity_score) for all docs in a project."""
    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT id, file_name, version, uploaded_at, clarity_score
        FROM documents
        WHERE project_id = ?
        ORDER BY uploaded_at DESC
        """,
        (project_id,),
    )
    docs = cursor.fetchall()
    conn.close()
    return docs


# ------------------------------------------
# Backward-compatible wrappers
# ------------------------------------------

def add_document_to_project(project_id: int, file_name: str, clarity_score: Optional[int]) -> int:
    """Adds a new document to a project, auto-incrementing its version for that file_name."""
    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT MAX(version) FROM documents WHERE project_id = ? AND file_name = ?",
        (project_id, file_name),
    )
    max_version = cursor.fetchone()[0]
    conn.close()

    new_version = 1 if max_version is None else max_version + 1
    return add_document(project_id, file_name, new_version, clarity_score)


def add_requirements_to_document(document_id: int, requirements_list: List[Tuple[str, str]]) -> None:
    """Wrapper kept for compatibility with existing UI code."""
    add_requirements(document_id, requirements_list)
