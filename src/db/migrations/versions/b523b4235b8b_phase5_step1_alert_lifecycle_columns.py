"""phase5 step1 alert lifecycle columns

Revision ID: b523b4235b8b
Revises: b4e1f8a2c903
Create Date: 2026-08-22 14:20:10.390623

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = 'b523b4235b8b'
down_revision: Union[str, None] = 'b4e1f8a2c903'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── identity / grouping ───────────────────────────────────────────
    op.add_column("alerts", sa.Column("fingerprint", sa.String(200), nullable=False))
    op.create_index("ix_alerts_fingerprint", "alerts", ["fingerprint"])

    op.add_column("alerts", sa.Column("incident_id", postgresql.UUID(as_uuid=True), nullable=True))
    op.create_index("ix_alerts_incident_id", "alerts", ["incident_id"])

    # ── lifecycle — the columns that make dedup possible ─────────────
    op.add_column("alerts", sa.Column("last_seen_at", sa.DateTime(timezone=True), nullable=False))
    op.add_column("alerts", sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("alerts", sa.Column("acknowledged_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("alerts", sa.Column("acknowledged_by", sa.String(100), nullable=True))
    op.add_column("alerts", sa.Column("occurrence_count", sa.Integer(), nullable=False,
                                      server_default="1"))

    op.alter_column("alerts", "status", existing_type=sa.String(20),
                    nullable=False, server_default="ACTIVE")

    # ── detection facts ───────────────────────────────────────────────
    # NULL now allowed = "this was a multivariate verdict"
    op.alter_column("alerts", "metric_name", existing_type=sa.String(100), nullable=True)
    op.add_column("alerts", sa.Column("detected_by", sa.JSON(), nullable=True))
    op.add_column("alerts", sa.Column("triggering_metrics", sa.JSON(), nullable=True))
    op.create_index("ix_alerts_service_name", "alerts", ["service_name"])

    # ── prediction ────────────────────────────────────────────────────
    op.add_column("alerts", sa.Column("escalation_probability", sa.Float(), nullable=True))

    # ── notifications (Step 7) ────────────────────────────────────────
    op.add_column("alerts", sa.Column("last_notified_at", sa.DateTime(timezone=True), nullable=True))

    # NOTE: the partial unique index (one ACTIVE alert per fingerprint)
    #       is deliberately NOT here — it lands in Step 3's migration.
    #       Step 1 has no dedup yet and would be blocked by it.


def downgrade() -> None:
    op.drop_column("alerts", "last_notified_at")
    op.drop_column("alerts", "escalation_probability")
    op.drop_index("ix_alerts_service_name", table_name="alerts")
    op.drop_column("alerts", "triggering_metrics")
    op.drop_column("alerts", "detected_by")
    op.alter_column("alerts", "metric_name", existing_type=sa.String(100), nullable=False)
    op.alter_column("alerts", "status", existing_type=sa.String(20),
                    nullable=True, server_default=None)
    op.drop_column("alerts", "occurrence_count")
    op.drop_column("alerts", "acknowledged_by")
    op.drop_column("alerts", "acknowledged_at")
    op.drop_column("alerts", "resolved_at")
    op.drop_column("alerts", "last_seen_at")
    op.drop_index("ix_alerts_incident_id", table_name="alerts")
    op.drop_column("alerts", "incident_id")
    op.drop_index("ix_alerts_fingerprint", table_name="alerts")
    op.drop_column("alerts", "fingerprint")