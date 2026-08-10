"""Isolated tests for db_store.reconcile — the natural-key app-side-diff helper
(WS15) that replaces delete-all + insert-all rebuilds with surgical
INSERT/UPDATE/DELETE. Exercised against throwaway tables on in-memory SQLite so
no real store or the live Azure database is touched.

Two table shapes are covered: a surrogate-``id`` table with a scoped natural key
(the gamelog/statcast/id-map/cache shape) and a composite natural-PK table with
no surrogate id (the recalibration_params shape)."""

import unittest

from sqlalchemy import (Column, Float, Integer, MetaData,
                        PrimaryKeyConstraint, String, Table, UniqueConstraint,
                        insert, select)

import db_store


def _rows(conn, table, scope=None):
    stmt = select(table).order_by(table.c[table.primary_key.columns.keys()[0]])
    if scope:
        for col, val in scope.items():
            stmt = stmt.where(table.c[col] == val)
    return [dict(r._mapping) for r in conn.execute(stmt)]


class _Backend:
    def setUp(self):
        db_store.configure_engine("sqlite://")
        self.md = MetaData()
        # Surrogate-id table: natural key (s, k) where s is the scope column.
        self.t = Table(
            "wtest", self.md,
            Column("id", Integer, primary_key=True, autoincrement=True),
            Column("s", String(8), nullable=False),   # scope
            Column("k", String(8), nullable=False),   # natural key within scope
            Column("v", Integer),
            Column("note", String(16)),
            UniqueConstraint("s", "k", name="uq_wtest"),
        )
        # Composite natural-PK table (no surrogate id) — recalibration_params shape.
        self.p = Table(
            "ptest", self.md,
            Column("sport", String(8), nullable=False),
            Column("prop", String(8), nullable=False),
            Column("a", Float),
            PrimaryKeyConstraint("sport", "prop", name="pk_ptest"),
        )
        self.md.create_all(db_store.get_engine())

    def tearDown(self):
        db_store.configure_engine(None)

    def _seed(self, table, *param_dicts):
        with db_store.get_engine().begin() as conn:
            if param_dicts:
                conn.execute(insert(table), list(param_dicts))


class SurrogateKeyReconcileTests(_Backend, unittest.TestCase):

    def test_insert_new_rows(self):
        desired = [{"s": "A", "k": "x", "v": 1, "note": "n1"},
                   {"s": "A", "k": "y", "v": 2, "note": "n2"}]
        with db_store.get_engine().begin() as conn:
            n = db_store.reconcile(conn, self.t, desired, ("k",),
                                   scope={"s": "A"})
        self.assertEqual(n, (2, 0, 0))
        with db_store.get_engine().connect() as conn:
            got = _rows(conn, self.t, {"s": "A"})
        self.assertEqual([(r["k"], r["v"]) for r in got], [("x", 1), ("y", 2)])

    def test_update_changed_row_keeps_surrogate_id(self):
        self._seed(self.t, {"s": "A", "k": "x", "v": 1, "note": "n1"},
                   {"s": "A", "k": "y", "v": 2, "note": "n2"})
        with db_store.get_engine().connect() as conn:
            id_x = next(r["id"] for r in _rows(conn, self.t, {"s": "A"})
                        if r["k"] == "x")
        desired = [{"s": "A", "k": "x", "v": 99, "note": "n1"},   # v changed
                   {"s": "A", "k": "y", "v": 2, "note": "n2"}]    # unchanged
        with db_store.get_engine().begin() as conn:
            n = db_store.reconcile(conn, self.t, desired, ("k",),
                                   scope={"s": "A"})
        self.assertEqual(n, (0, 1, 0))   # exactly one UPDATE, no churn on 'y'
        with db_store.get_engine().connect() as conn:
            got = {r["k"]: r for r in _rows(conn, self.t, {"s": "A"})}
        self.assertEqual(got["x"]["v"], 99)
        self.assertEqual(got["x"]["id"], id_x)   # surrogate id stable across update

    def test_delete_removed_row(self):
        self._seed(self.t, {"s": "A", "k": "x", "v": 1},
                   {"s": "A", "k": "y", "v": 2})
        desired = [{"s": "A", "k": "x", "v": 1}]   # y dropped
        with db_store.get_engine().begin() as conn:
            n = db_store.reconcile(conn, self.t, desired, ("k",),
                                   scope={"s": "A"})
        self.assertEqual(n, (0, 0, 1))
        with db_store.get_engine().connect() as conn:
            self.assertEqual([r["k"] for r in _rows(conn, self.t, {"s": "A"})],
                             ["x"])

    def test_identical_desired_is_a_noop(self):
        self._seed(self.t, {"s": "A", "k": "x", "v": 1, "note": "n"})
        desired = [{"s": "A", "k": "x", "v": 1, "note": "n"}]
        with db_store.get_engine().begin() as conn:
            n = db_store.reconcile(conn, self.t, desired, ("k",),
                                   scope={"s": "A"})
        self.assertEqual(n, (0, 0, 0))

    def test_scope_isolates_other_partitions(self):
        self._seed(self.t, {"s": "A", "k": "x", "v": 1},
                   {"s": "B", "k": "x", "v": 100})   # different scope, same key
        desired = [{"s": "A", "k": "z", "v": 2}]     # replace scope A entirely
        with db_store.get_engine().begin() as conn:
            n = db_store.reconcile(conn, self.t, desired, ("k",),
                                   scope={"s": "A"})
        self.assertEqual(n, (1, 0, 1))               # +z, -x(A); B untouched
        with db_store.get_engine().connect() as conn:
            self.assertEqual([r["k"] for r in _rows(conn, self.t, {"s": "A"})],
                             ["z"])
            b = _rows(conn, self.t, {"s": "B"})
        self.assertEqual([(r["k"], r["v"]) for r in b], [("x", 100)])

    def test_empty_desired_clears_scope(self):
        self._seed(self.t, {"s": "A", "k": "x", "v": 1},
                   {"s": "A", "k": "y", "v": 2})
        with db_store.get_engine().begin() as conn:
            n = db_store.reconcile(conn, self.t, [], ("k",), scope={"s": "A"})
        self.assertEqual(n, (0, 0, 2))
        with db_store.get_engine().connect() as conn:
            self.assertEqual(_rows(conn, self.t, {"s": "A"}), [])

    def test_duplicate_desired_identity_raises(self):
        desired = [{"s": "A", "k": "x", "v": 1},
                   {"s": "A", "k": "x", "v": 2}]
        with db_store.get_engine().begin() as conn:
            with self.assertRaises(ValueError):
                db_store.reconcile(conn, self.t, desired, ("k",), scope={"s": "A"})

    def test_duplicate_existing_identity_raises(self):
        # An unconstrained table could hold two rows sharing the natural key; the
        # helper must refuse to diff rather than silently orphan one. Force it by
        # inserting the collision directly (bypassing the UNIQUE via distinct ids
        # is impossible here, so use a scope where the key repeats across a column
        # not in identity_cols).
        self._seed(self.t, {"s": "A", "k": "x", "v": 1, "note": "a"})
        # Insert a second row with the same (s) scope but rely on identity_cols
        # being a strict subset that collides: identity ("v",) → both rows v=1.
        self._seed(self.t, {"s": "A", "k": "y", "v": 1, "note": "b"})
        with db_store.get_engine().begin() as conn:
            with self.assertRaises(ValueError):
                db_store.reconcile(conn, self.t, [{"s": "A", "k": "x", "v": 1}],
                                   ("v",), scope={"s": "A"})

    def test_returns_counts_across_mixed_diff(self):
        self._seed(self.t, {"s": "A", "k": "keep", "v": 1},
                   {"s": "A", "k": "change", "v": 1},
                   {"s": "A", "k": "drop", "v": 1})
        desired = [{"s": "A", "k": "keep", "v": 1},      # noop
                   {"s": "A", "k": "change", "v": 2},    # update
                   {"s": "A", "k": "add", "v": 9}]       # insert; drop deleted
        with db_store.get_engine().begin() as conn:
            n = db_store.reconcile(conn, self.t, desired, ("k",), scope={"s": "A"})
        self.assertEqual(n, (1, 1, 1))


class CompositeKeyReconcileTests(_Backend, unittest.TestCase):

    def test_composite_natural_pk_no_surrogate(self):
        self._seed(self.p, {"sport": "mlb", "prop": "hits", "a": 0.1},
                   {"sport": "mlb", "prop": "outs", "a": 0.2})
        desired = [{"sport": "mlb", "prop": "hits", "a": 0.9},   # update
                   {"sport": "mlb", "prop": "ks", "a": 0.3}]     # insert; outs gone
        with db_store.get_engine().begin() as conn:
            n = db_store.reconcile(conn, self.p, desired, ("sport", "prop"),
                                   scope={"sport": "mlb"})
        self.assertEqual(n, (1, 1, 1))
        with db_store.get_engine().connect() as conn:
            got = {r["prop"]: r["a"] for r in _rows(conn, self.p, {"sport": "mlb"})}
        self.assertEqual(got, {"hits": 0.9, "ks": 0.3})

    def test_ignores_surrogate_id_key_in_desired(self):
        # A desired dict that happens to carry an 'id' key is written without it
        # (the DB assigns/keeps the surrogate) — proves the id-strip.
        desired = [{"id": 12345, "s": "A", "k": "x", "v": 1}]
        with db_store.get_engine().begin() as conn:
            db_store.reconcile(conn, self.t, desired, ("k",), scope={"s": "A"})
        with db_store.get_engine().connect() as conn:
            got = _rows(conn, self.t, {"s": "A"})
        self.assertEqual(len(got), 1)
        self.assertNotEqual(got[0]["id"], 12345)   # DB-assigned, not the passed id


if __name__ == "__main__":
    unittest.main()
