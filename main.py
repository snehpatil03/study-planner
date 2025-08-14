import datetime
import hashlib
import sqlite3
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Database utility functions
DB_PATH = "/home/oai/share/studyplannerapp/studyplanner.db"


def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db_connection()
    cur = conn.cursor()
    # Create tables if they do not exist
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS profiles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            target_exam_date TEXT,
            available_hours_per_week INTEGER,
            timezone TEXT,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS subjects (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            description TEXT
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS user_subject_progress (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            subject_id INTEGER NOT NULL,
            current_proficiency INTEGER,
            target_proficiency INTEGER,
            FOREIGN KEY(user_id) REFERENCES users(id),
            FOREIGN KEY(subject_id) REFERENCES subjects(id)
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS resources (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            subject_id INTEGER NOT NULL,
            title TEXT NOT NULL,
            url TEXT,
            type TEXT,
            difficulty INTEGER,
            description TEXT,
            FOREIGN KEY(subject_id) REFERENCES subjects(id)
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS study_tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            resource_id INTEGER,
            scheduled_date TEXT NOT NULL,
            completed INTEGER DEFAULT 0,
            difficulty_rating INTEGER,
            e_factor REAL DEFAULT 2.5,
            repetition_number INTEGER DEFAULT 0,
            last_review_date TEXT,
            FOREIGN KEY(user_id) REFERENCES users(id),
            FOREIGN KEY(resource_id) REFERENCES resources(id)
        )
        """
    )
    conn.commit()
    conn.close()


# Initialize DB when module is imported
init_db()


# Models for API
class RegisterModel(BaseModel):
    username: str
    password: str


class LoginModel(BaseModel):
    username: str
    password: str


class SubjectProgress(BaseModel):
    subject: str
    current_proficiency: int
    target_proficiency: int


class SetupModel(BaseModel):
    user_id: int
    target_exam_date: str  # ISO date string
    available_hours_per_week: int
    timezone: str
    subjects: List[SubjectProgress]


class DifficultyFeedback(BaseModel):
    user_id: int
    rating: int = Field(..., ge=0, le=5)


# Helper functions

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


# Weighted scheduling example

def weighted_schedule(user_id: int, profile: dict):
    """Generate simple weighted tasks for the user for the next 7 days."""
    conn = get_db_connection()
    cur = conn.cursor()
    # compute priorities based on progress
    progress_rows = cur.execute(
        "SELECT subject_id, current_proficiency, target_proficiency FROM user_subject_progress WHERE user_id = ?",
        (user_id,),
    ).fetchall()
    priorities = {}
    for row in progress_rows:
        gap = row[2] - row[1]  # target - current
        priorities[row[0]] = max(gap, 1)
    # generate tasks for next 7 days
    today = datetime.date.today()
    tasks = []
    for i in range(7):
        if not priorities:
            break
        date = today + datetime.timedelta(days=i)
        # choose subject with highest priority
        subject_id = max(priorities, key=priorities.get)
        # pick a random resource from that subject
        res_row = cur.execute(
            "SELECT id FROM resources WHERE subject_id = ? ORDER BY RANDOM() LIMIT 1",
            (subject_id,),
        ).fetchone()
        resource_id = res_row[0] if res_row else None
        cur.execute(
            "INSERT INTO study_tasks (user_id, resource_id, scheduled_date) VALUES (?, ?, ?)",
            (user_id, resource_id, date.isoformat()),
        )
        # decrease priority to distribute tasks
        priorities[subject_id] -= 1
        if priorities[subject_id] <= 0:
            del priorities[subject_id]
    conn.commit()
    conn.close()


# SM2 Scheduler

def sm2_schedule(task_id: int, quality: int):
    conn = get_db_connection()
    cur = conn.cursor()
    # fetch task
    row = cur.execute(
        "SELECT user_id, resource_id, scheduled_date, e_factor, repetition_number, last_review_date FROM study_tasks WHERE id = ?",
        (task_id,),
    ).fetchone()
    if not row:
        conn.close()
        return
    user_id, resource_id, scheduled_date, e_factor, repetition_number, last_review_date = row
    if e_factor is None:
        e_factor = 2.5
    if repetition_number is None:
        repetition_number = 0
    # update EF
    ef = e_factor + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))
    if ef < 1.3:
        ef = 1.3
    # compute interval
    if quality < 3:
        repetition_number = 0
        interval = 1
    else:
        if repetition_number == 0:
            interval = 1
        elif repetition_number == 1:
            interval = 6
        else:
            # compute previous interval based on last_review_date and scheduled_date
            if last_review_date:
                prev_date = datetime.date.fromisoformat(last_review_date)
            else:
                prev_date = datetime.date.fromisoformat(scheduled_date)
            curr_date = datetime.date.fromisoformat(scheduled_date)
            prev_interval = (curr_date - prev_date).days
            interval = int(round(prev_interval * ef))
        repetition_number += 1
    next_date = datetime.date.today() + datetime.timedelta(days=interval)
    # create new task for the next review
    cur.execute(
        "INSERT INTO study_tasks (user_id, resource_id, scheduled_date, e_factor, repetition_number, last_review_date) VALUES (?, ?, ?, ?, ?, ?)",
        (user_id, resource_id, next_date.isoformat(), ef, repetition_number, datetime.date.today().isoformat()),
    )
    conn.commit()
    conn.close()


app = FastAPI(title="Personalized Study Planner")


@app.post("/register")
def register_user(data: RegisterModel):
    conn = get_db_connection()
    cur = conn.cursor()
    password_hash = hash_password(data.password)
    try:
        cur.execute(
            "INSERT INTO users (username, password_hash) VALUES (?, ?)",
            (data.username, password_hash),
        )
        user_id = cur.lastrowid
        conn.commit()
    except sqlite3.IntegrityError:
        conn.close()
        raise HTTPException(status_code=400, detail="Username already exists")
    conn.close()
    return {"user_id": user_id}


@app.post("/login")
def login_user(data: LoginModel):
    conn = get_db_connection()
    cur = conn.cursor()
    row = cur.execute(
        "SELECT id, password_hash FROM users WHERE username = ?",
        (data.username,),
    ).fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=401, detail="Invalid credentials")
    user_id, password_hash = row
    if hash_password(data.password) != password_hash:
        conn.close()
        raise HTTPException(status_code=401, detail="Invalid credentials")
    conn.close()
    return {"user_id": user_id}


@app.post("/setup")
def setup_profile(data: SetupModel):
    """Create profile and subject progress for a user. Schedules initial tasks."""
    # Validate date
    try:
        exam_date = datetime.date.fromisoformat(data.target_exam_date)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid exam date format")
    conn = get_db_connection()
    cur = conn.cursor()
    # insert or update profile
    cur.execute(
        "SELECT id FROM profiles WHERE user_id = ?",
        (data.user_id,),
    )
    existing = cur.fetchone()
    if existing:
        cur.execute(
            "UPDATE profiles SET target_exam_date = ?, available_hours_per_week = ?, timezone = ? WHERE user_id = ?",
            (exam_date.isoformat(), data.available_hours_per_week, data.timezone, data.user_id),
        )
    else:
        cur.execute(
            "INSERT INTO profiles (user_id, target_exam_date, available_hours_per_week, timezone) VALUES (?, ?, ?, ?)",
            (data.user_id, exam_date.isoformat(), data.available_hours_per_week, data.timezone),
        )
    # create subjects if not exist and progress records
    for subj in data.subjects:
        # ensure subject exists
        cur.execute("INSERT OR IGNORE INTO subjects (name) VALUES (?)", (subj.subject,))
        # get subject_id
        row = cur.execute("SELECT id FROM subjects WHERE name = ?", (subj.subject,)).fetchone()
        subject_id = row[0]
        # insert or update progress
        row2 = cur.execute(
            "SELECT id FROM user_subject_progress WHERE user_id = ? AND subject_id = ?",
            (data.user_id, subject_id),
        ).fetchone()
        if row2:
            cur.execute(
                "UPDATE user_subject_progress SET current_proficiency = ?, target_proficiency = ? WHERE id = ?",
                (subj.current_proficiency, subj.target_proficiency, row2[0]),
            )
        else:
            cur.execute(
                "INSERT INTO user_subject_progress (user_id, subject_id, current_proficiency, target_proficiency) VALUES (?, ?, ?, ?)",
                (data.user_id, subject_id, subj.current_proficiency, subj.target_proficiency),
            )
    conn.commit()
    conn.close()
    # generate initial schedule using weighted scheduling
    weighted_schedule(data.user_id, data.dict())
    return {"status": "Profile setup complete and tasks scheduled"}


@app.get("/tasks")
def list_tasks(user_id: int, upcoming_only: bool = False):
    conn = get_db_connection()
    cur = conn.cursor()
    today = datetime.date.today()
    if upcoming_only:
        rows = cur.execute(
            "SELECT id, scheduled_date, completed, resource_id FROM study_tasks WHERE user_id = ? AND scheduled_date >= ? ORDER BY scheduled_date",
            (user_id, today.isoformat()),
        ).fetchall()
    else:
        rows = cur.execute(
            "SELECT id, scheduled_date, completed, resource_id FROM study_tasks WHERE user_id = ? ORDER BY scheduled_date",
            (user_id,),
        ).fetchall()
    tasks = []
    for r in rows:
        resource_title = None
        if r[3]:
            res_row = cur.execute("SELECT title FROM resources WHERE id = ?", (r[3],)).fetchone()
            resource_title = res_row[0] if res_row else None
        tasks.append(
            {
                "task_id": r[0],
                "scheduled_date": r[1],
                "completed": bool(r[2]),
                "resource_title": resource_title,
            }
        )
    conn.close()
    return {"tasks": tasks}


@app.post("/tasks/{task_id}/complete")
def complete_task(task_id: int, feedback: DifficultyFeedback):
    conn = get_db_connection()
    cur = conn.cursor()
    # verify task exists and belongs to user
    row = cur.execute(
        "SELECT user_id, completed FROM study_tasks WHERE id = ?",
        (task_id,),
    ).fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="Task not found")
    if row[0] != feedback.user_id:
        conn.close()
        raise HTTPException(status_code=403, detail="Not authorized to complete this task")
    if row[1]:
        conn.close()
        raise HTTPException(status_code=400, detail="Task already completed")
    # update task as completed
    cur.execute(
        "UPDATE study_tasks SET completed = 1, difficulty_rating = ? WHERE id = ?",
        (feedback.rating, task_id),
    )
    conn.commit()
    conn.close()
    # schedule next review using SM2
    sm2_schedule(task_id, feedback.rating)
    return {"status": "Task completed and next review scheduled"}


# Example endpoint to add resource manually
class ResourceModel(BaseModel):
    subject: str
    title: str
    url: Optional[str] = None
    type: Optional[str] = None
    difficulty: int
    description: Optional[str] = None


@app.post("/resources")
def add_resource(data: ResourceModel):
    conn = get_db_connection()
    cur = conn.cursor()
    # ensure subject exists
    cur.execute("INSERT OR IGNORE INTO subjects (name) VALUES (?)", (data.subject,))
    row = cur.execute("SELECT id FROM subjects WHERE name = ?", (data.subject,)).fetchone()
    subject_id = row[0]
    cur.execute(
        "INSERT INTO resources (subject_id, title, url, type, difficulty, description) VALUES (?, ?, ?, ?, ?, ?)",
        (subject_id, data.title, data.url, data.type, data.difficulty, data.description),
    )
    resource_id = cur.lastrowid
    conn.commit()
    conn.close()
    return {"resource_id": resource_id}


# Running instructions: use `uvicorn main:app --reload --port 8000` in this directory
