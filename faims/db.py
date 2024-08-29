import sqlite3
import click
from flask import current_app, g
import os
import uuid


def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(current_app.config["DATABASE"],
                               detect_types=sqlite3.PARSE_DECLTYPES)
        g.db.row_factory = sqlite3.Row
        g.db.execute("PRAGMA foreign_keys = ON;")
    return g.db


def close_db(e):
    db = g.pop("db", None)
    if db is not None:
        db.close()
    return


def init_db():
    if os.path.exists(current_app.config["DATABASE"]):
        raise ValueError(f"Database already exists at {current_app.config['DATABASE']}")
    db = get_db()
    with current_app.open_resource("schema.sql") as f:
        db.executescript(f.read().decode("utf8"))
    db.commit()
    return


@click.command("init-db")
def init_db_command():
    init_db()
    click.echo(f"Database initialized at {current_app.config['DATABASE']}")
    return


def init_app(app):
    """Registers init db command"""
    app.teardown_appcontext(close_db)  # Call this function when shutting down
    app.cli.add_command(init_db_command)  # Add command to flask
    return


def get_unique_value_for_field(db, field: str, table: str) -> str:
    """
    Gets a unique uuid for the given field in the given table. This is intended for use with DBMs that don't have
    this as an built-in feature.
    Parameters
    ----------
    db
        Database object from which to pull, obtained via get_db
    field : str
        Field for which the unique value should be obtained
    table : str
        Table from which the field should be pulled

    Returns
    -------
    str
        Unique value for the field
    """
    row = 'temp'
    while len(row) != 0:
        tentative_id = str(uuid.uuid4())
        row = db.execute(f'SELECT {field} FROM {table} WHERE {field} = ?', (tentative_id,)).fetchall()
    return tentative_id
