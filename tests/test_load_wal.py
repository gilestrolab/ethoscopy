"""
Tests for opening ethoscope databases across SQLite journal/WAL states.

Ethoscopes record in WAL mode and the results tree is normally mounted
read-only (``:ro`` in Docker), a combination that has broken loading more than
once. Every test here builds a *real* SQLite file in a specific on-disk state
rather than mocking sqlite3, because the failures being guarded against come
from SQLite itself and only appear when a statement actually runs.

The states covered mirror what has been seen on the lab mounts:

===========================  ====================================================
state                        why it matters
===========================  ====================================================
delete mode                  the state databases are converted to as a fix
WAL, -wal empty + -shm       what an ethoscope leaves after checkpointing
WAL, data in -wal + -shm     a database copied while still being written
WAL, no sidecars             -wal/-shm dropped by rsync/cleanup; needs immutable
hot rollback journal         a crashed writer
===========================  ====================================================
"""

import os
import shutil
import sqlite3
import stat
import sys
import warnings

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ethoscopy.load import _READ_STRATEGIES, _connect_db  # noqa: E402

BASE_ROWS = 10
WAL_ROWS = 200


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def make_db(tmp_path):
    """
    Factory building a SQLite database in a chosen on-disk state.

    Returns:
        callable: (name, journal_mode, **state) -> Path to the database file
    """

    def _make(
        name,
        journal_mode="delete",
        leave_wal=False,
        drop_sidecars=False,
        corrupt_shm=False,
        hot_journal=False,
    ):
        directory = tmp_path / name
        directory.mkdir()
        db = directory / "fixture.db"

        conn = sqlite3.connect(str(db))
        conn.execute(f"PRAGMA journal_mode={journal_mode};")
        conn.execute("PRAGMA wal_autocheckpoint=0;")
        conn.execute("CREATE TABLE ROI_MAP (roi_idx INT, roi_value INT);")
        conn.executemany(
            "INSERT INTO ROI_MAP VALUES (?,?)",
            [(i, i * 2) for i in range(BASE_ROWS)],
        )
        conn.commit()

        if hot_journal:
            # Snapshot the directory mid-transaction, then roll back: the copy
            # is left with a hot rollback journal, as after a crashed writer.
            conn.execute("BEGIN IMMEDIATE")
            conn.executemany(
                "INSERT INTO ROI_MAP VALUES (?,?)",
                [(i, i * 2) for i in range(BASE_ROWS, WAL_ROWS)],
            )
            _snapshot(directory, db, ("", "-journal"))
            conn.rollback()
            conn.close()
            return db

        if leave_wal:
            # Commit more rows with checkpointing disabled, then snapshot while
            # the writer still holds them: the copy keeps them only in the -wal.
            conn.executemany(
                "INSERT INTO ROI_MAP VALUES (?,?)",
                [(i, i * 2) for i in range(BASE_ROWS, WAL_ROWS)],
            )
            conn.commit()
            _snapshot(directory, db, ("", "-wal", "-shm"))
            conn.close()
        else:
            if journal_mode == "wal":
                conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
            conn.close()

        if drop_sidecars:
            for suffix in ("-wal", "-shm"):
                sidecar = db.with_name(db.name + suffix)
                if sidecar.exists():
                    sidecar.unlink()
        if corrupt_shm:
            shm = db.with_name(db.name + "-shm")
            if shm.exists():
                with open(shm, "r+b") as fh:
                    fh.write(b"\xde\xad\xbe\xef" * 8)
        return db

    return _make


def _snapshot(directory, db, suffixes):
    """Replace `directory` with a copy of `db` + `suffixes` taken right now."""
    snap = directory.with_name(directory.name + "_snap")
    shutil.rmtree(snap, ignore_errors=True)
    snap.mkdir()
    for suffix in suffixes:
        src = db.with_name(db.name + suffix)
        if src.exists():
            shutil.copy(src, snap / src.name)
    shutil.rmtree(directory)
    snap.rename(directory)


@pytest.fixture
def readonly_dir():
    """
    Make a directory read-only for the duration of a test, then restore it.

    Returns:
        callable: (path) -> None
    """
    touched = []

    def _make_readonly(path):
        if os.geteuid() == 0:
            pytest.skip("running as root: directory permissions are not enforced")
        os.chmod(path, stat.S_IRUSR | stat.S_IXUSR)
        touched.append(path)

    yield _make_readonly

    for path in touched:
        os.chmod(path, stat.S_IRWXU)


def _rows(conn):
    return conn.execute("SELECT count(*) FROM ROI_MAP").fetchone()[0]


# --------------------------------------------------------------------------- #
# the core guarantee: every recoverable state opens AND answers a query
# --------------------------------------------------------------------------- #


READABLE_STATES = [
    # (id, build kwargs, expected rows)
    ("delete_clean", dict(journal_mode="delete"), BASE_ROWS),
    ("wal_checkpointed_no_sidecars", dict(journal_mode="wal"), BASE_ROWS),
    ("wal_data_in_wal", dict(journal_mode="wal", leave_wal=True), WAL_ROWS),
    (
        "wal_corrupt_shm",
        dict(journal_mode="wal", leave_wal=True, corrupt_shm=True),
        WAL_ROWS,
    ),
    ("hot_rollback_journal", dict(journal_mode="delete", hot_journal=True), BASE_ROWS),
]


@pytest.mark.parametrize("state_id,kwargs,expected", READABLE_STATES)
@pytest.mark.parametrize("read_only", [True, False], ids=["ro_dir", "rw_dir"])
def test_every_state_opens_and_reads(
    make_db, readonly_dir, state_id, kwargs, expected, read_only
):
    """
    A database in any recoverable state can be opened and queried.

    This is the regression guard for the ``mode=ro&nolock=1`` bug: that URI is
    rejected by SQLite for every WAL database, so any strategy that reaches for
    it fails the WAL rows here.
    """
    db = make_db(f"{state_id}_{read_only}", **kwargs)
    if read_only:
        readonly_dir(db.parent)

    conn = _connect_db(db)
    try:
        assert _rows(conn) == expected
    finally:
        conn.close()


@pytest.mark.parametrize("state_id,kwargs,expected", READABLE_STATES)
def test_uncheckpointed_wal_data_is_not_silently_dropped(
    make_db, readonly_dir, state_id, kwargs, expected
):
    """
    Rows committed to the -wal are still returned on a read-only mount.

    The immutable fallback cannot see them, so this fails if immutable is
    reached for before a faithful open mode has been tried.
    """
    db = make_db(f"fidelity_{state_id}", **kwargs)
    readonly_dir(db.parent)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        conn = _connect_db(db)
    try:
        assert _rows(conn) == expected
    finally:
        conn.close()


# --------------------------------------------------------------------------- #
# why the probe exists
# --------------------------------------------------------------------------- #


def test_sqlite_connect_alone_does_not_detect_a_broken_open_mode(make_db, readonly_dir):
    """
    Document the trap: sqlite3.connect() succeeds on a URI SQLite cannot use.

    The original bug survived a try/except around the connect call for exactly
    this reason - the failure landed later, inside pandas. _connect_db must
    therefore probe with a statement, which the next test asserts.
    """
    db = make_db("lazy_connect", journal_mode="wal", leave_wal=True)
    readonly_dir(db.parent)

    conn = sqlite3.connect(f"file:{db}?mode=ro&nolock=1", uri=True)
    try:
        with pytest.raises(sqlite3.OperationalError, match="unable to open database"):
            conn.execute("SELECT count(*) FROM ROI_MAP").fetchone()
    finally:
        conn.close()


def test_returned_connection_has_already_answered_a_statement(make_db, readonly_dir):
    """A connection handed back by _connect_db is usable, not merely created."""
    db = make_db("probed", journal_mode="wal")
    readonly_dir(db.parent)

    conn = _connect_db(db)
    try:
        # would raise here rather than at the caller's first real query
        assert conn.execute("SELECT count(*) FROM sqlite_master").fetchone()[0] > 0
    finally:
        conn.close()


def test_nolock_is_not_in_the_strategy_ladder():
    """nolock=1 is unusable for WAL databases; it must never come back."""
    assert not any("nolock" in strategy for strategy in _READ_STRATEGIES)


# --------------------------------------------------------------------------- #
# the degraded fallback must be audible
# --------------------------------------------------------------------------- #


def test_warns_when_immutable_fallback_hides_wal_data(make_db):
    """Falling back to immutable with a non-empty -wal warns about lost rows."""
    db = make_db("warns", journal_mode="wal", leave_wal=True)

    with pytest.warns(RuntimeWarning, match="ignores its .* -wal sidecar"):
        conn = _connect_db(db, degraded=True)
    conn.close()


def test_no_warning_for_the_ordinary_checkpointed_case(make_db, readonly_dir):
    """The common ethoscope state must load silently - no warning fatigue."""
    db = make_db("quiet", journal_mode="wal")
    readonly_dir(db.parent)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        conn = _connect_db(db)
        conn.close()

    assert [str(w.message) for w in caught] == []


def test_degraded_skips_the_faithful_strategies(make_db):
    """
    degraded=True is what makes a retry a different attempt.

    The warning only fires from the immutable rung, so its presence proves the
    faithful rung was skipped rather than retried identically.
    """
    db = make_db("degraded", journal_mode="wal", leave_wal=True)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        conn = _connect_db(db, degraded=True)
        conn.close()
    assert any(isinstance(w.message, RuntimeWarning) for w in caught)

    # without it, the faithful rung is used and stays silent
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        conn = _connect_db(db)
        conn.close()
    assert not [w for w in caught if isinstance(w.message, RuntimeWarning)]


# --------------------------------------------------------------------------- #
# ethoscopy is a reader: it must not touch the raw data
# --------------------------------------------------------------------------- #


def test_loading_does_not_mutate_the_database(make_db):
    """
    Opening a WAL database must not checkpoint or truncate it.

    A read-write connection checkpoints the -wal when the last connection
    closes, silently rewriting raw experimental data on a writable mount.
    """
    db = make_db("no_mutation", journal_mode="wal", leave_wal=True)
    before = {
        p.name: (p.stat().st_size, p.stat().st_mtime_ns) for p in db.parent.iterdir()
    }

    conn = _connect_db(db)
    _rows(conn)
    conn.close()

    after = {
        p.name: (p.stat().st_size, p.stat().st_mtime_ns) for p in db.parent.iterdir()
    }
    assert after[db.name] == before[db.name], "main database file was modified"
    wal = db.name + "-wal"
    assert after.get(wal) == before.get(wal), "-wal sidecar was checkpointed away"


# --------------------------------------------------------------------------- #
# failure reporting
# --------------------------------------------------------------------------- #


def test_missing_file_raises_a_clear_error(tmp_path):
    """
    A missing path is reported as missing.

    Left to SQLite, ``immutable=1`` "opens" a path that is not there and the
    caller gets ``no such table: ROI_MAP`` instead - which reads like a corrupt
    database rather than a wrong path or an unmounted share.
    """
    missing = tmp_path / "nope.db"

    with pytest.raises(FileNotFoundError) as excinfo:
        _connect_db(missing)

    assert str(missing) in str(excinfo.value)


def test_missing_file_is_not_created(tmp_path):
    """Probing a missing path must not litter an empty database behind it."""
    missing = tmp_path / "nope.db"

    with pytest.raises(FileNotFoundError):
        _connect_db(missing)

    assert not missing.exists()
    assert list(tmp_path.iterdir()) == []


def test_directory_instead_of_file_raises(tmp_path):
    """A path that is not a database is reported, not silently accepted."""
    with pytest.raises((FileNotFoundError, sqlite3.Error)):
        _connect_db(tmp_path)


# --------------------------------------------------------------------------- #
# end to end, against the real ethoscope database
# --------------------------------------------------------------------------- #


@pytest.fixture
def wal_ethoscope_db(real_ethoscope_db, tmp_path):
    """
    Copy of the real ethoscope database converted to WAL mode.

    Returns:
        Path: Path to the copied database, in its own directory
    """
    directory = tmp_path / "results"
    directory.mkdir()
    db = directory / real_ethoscope_db.name
    shutil.copy(real_ethoscope_db, db)

    conn = sqlite3.connect(str(db))
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.close()
    return db


@pytest.mark.integration
def test_read_single_roi_on_wal_database_on_readonly_mount(
    wal_ethoscope_db, readonly_dir
):
    """
    The exact production failure: a WAL ethoscope database on a :ro mount.

    Before the fix this raised
    ``DatabaseError: Execution failed on sql 'SELECT * FROM ROI_MAP': unable to
    open database file`` for every ROI.
    """
    from ethoscopy.load import read_single_roi

    readonly_dir(wal_ethoscope_db.parent)

    file_info = pd.Series(
        {
            "path": str(wal_ethoscope_db),
            "region_id": 1,
            "machine_id": "ETHOSCOPE_070",
            "date": "2025-07-10",
        }
    )

    result = read_single_roi(file_info)

    assert result is not None
    assert len(result) > 0
    assert {"t", "x", "y"} <= set(result.columns)


@pytest.mark.integration
def test_one_unreadable_database_does_not_sink_the_whole_load(
    real_ethoscope_db, wal_ethoscope_db, tmp_path
):
    """
    A broken database costs its own ROIs, not everybody else's.

    Opening a database and reading its shared ROI_MAP/VAR_MAP/METADATA tables
    happens once per file, outside the per-ROI error handling, so a single
    unreadable file used to abort the load and discard every ROI already read.
    """
    from ethoscopy.load import load_ethoscope

    metadata = pd.DataFrame(
        {
            "path": [str(wal_ethoscope_db), str(tmp_path / "gone.db")],
            "region_id": [1, 1],
            "machine_id": ["ETHOSCOPE_070", "ETHOSCOPE_999"],
            "machine_name": ["ETHOSCOPE_070", "ETHOSCOPE_999"],
            "date": ["2025-07-10", "2025-07-10"],
            "id": ["good|01", "missing|01"],
        }
    )

    data = load_ethoscope(metadata, progress=False, verbose=False)

    assert len(data) > 0, "the readable database should still have loaded"
    assert set(data["id"].unique()) == {"good|01"}


@pytest.mark.integration
def test_wal_and_delete_mode_return_identical_data(
    real_ethoscope_db, wal_ethoscope_db, readonly_dir
):
    """
    Journal mode must not change the science.

    Guards the immutable fallback: if it were reached for a checkpointed WAL
    database it would still pass, but any row loss shows up here.
    """
    from ethoscopy.load import read_single_roi

    file_info = pd.Series(
        {
            "path": str(real_ethoscope_db),
            "region_id": 1,
            "machine_id": "ETHOSCOPE_070",
            "date": "2025-07-10",
        }
    )
    baseline = read_single_roi(file_info)

    readonly_dir(wal_ethoscope_db.parent)
    file_info["path"] = str(wal_ethoscope_db)
    from_wal = read_single_roi(file_info)

    pd.testing.assert_frame_equal(baseline, from_wal)
