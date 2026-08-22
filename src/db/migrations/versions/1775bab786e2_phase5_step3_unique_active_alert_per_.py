"""phase5 step3 unique active alert per fingerprint

Revision ID: 1775bab786e2
Revises: b523b4235b8b
Create Date: 2026-08-22 21:48:47.344798

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '1775bab786e2'
down_revision: Union[str, None] = 'b523b4235b8b'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_index(
        "ux_alerts_fingerprint_active",
        "alerts",
        ["fingerprint"],
        unique=True,
        postgresql_where=sa.text("status = 'ACTIVE'"),
    )


def downgrade() -> None:
    op.drop_index("ux_alerts_fingerprint_active", table_name="alerts")