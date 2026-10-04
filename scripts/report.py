"""Print the unique visitor count and recent events straight from the database.
Usage: python scripts/report.py [path/to/face_tracker.db]"""
import sqlite3
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "data/face_tracker.db"
con = sqlite3.connect(path)
print("Unique visitors :", con.execute("SELECT COUNT(*) FROM faces").fetchone()[0])
print("Events by type  :", dict(con.execute("SELECT event_type, COUNT(*) FROM events GROUP BY event_type")))
print("\nLast 15 events:")
for row in con.execute("SELECT event_id, face_id, event_type, timestamp, image_path FROM events "
                       "ORDER BY event_id DESC LIMIT 15"):
    print(" ", row)
