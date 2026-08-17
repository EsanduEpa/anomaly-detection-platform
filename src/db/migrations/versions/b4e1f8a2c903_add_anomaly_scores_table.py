"""add anomaly scores table

Revision ID: b4e1f8a2c903
Revises: 9695e773ac65
Create Date: 2026-08-17
"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa


revision: str = 'b4e1f8a2c903'
down_revision: Union[str, None] = '9695e773ac65'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'anomaly_scores',
        sa.Column('id',            sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column('timestamp',     sa.DateTime(timezone=True), nullable=False),
        sa.Column('service_name',  sa.String(100),  nullable=False),
        sa.Column('host',          sa.String(100),  nullable=False),
        sa.Column('metric_name',   sa.String(100),  nullable=True),
        sa.Column('zscore_value',  sa.Float(),       nullable=True),
        sa.Column('zscore_flag',   sa.Boolean(),     nullable=False, server_default='false'),
        sa.Column('iforest_score', sa.Float(),       nullable=True),
        sa.Column('iforest_flag',  sa.Boolean(),     nullable=False, server_default='false'),
        sa.Column('lstm_error',    sa.Float(),       nullable=True),
        sa.Column('lstm_flag',     sa.Boolean(),     nullable=False, server_default='false'),
        sa.Column('votes',         sa.Integer(),     nullable=False, server_default='0'),
        sa.Column('ensemble_score',sa.Float(),       nullable=False, server_default='0.0'),
        sa.Column('is_anomaly',    sa.Boolean(),     nullable=False, server_default='false'),
        sa.Column('model_version', sa.String(50),    nullable=True),
        sa.Column('created_at',    sa.DateTime(timezone=True), server_default=sa.text('now()')),
    )
    # Indexes for the queries Phase 3 will run most often:
    # "give me all anomalies in the last hour" → timestamp index
    # "how many anomalies did payment-service have today?" → service_name index
    # "show me only the flagged rows" → is_anomaly index
    op.create_index('ix_anomaly_scores_timestamp',    'anomaly_scores', ['timestamp'])
    op.create_index('ix_anomaly_scores_service_name', 'anomaly_scores', ['service_name'])
    op.create_index('ix_anomaly_scores_is_anomaly',   'anomaly_scores', ['is_anomaly'])


def downgrade() -> None:
    op.drop_index('ix_anomaly_scores_is_anomaly',   table_name='anomaly_scores')
    op.drop_index('ix_anomaly_scores_service_name', table_name='anomaly_scores')
    op.drop_index('ix_anomaly_scores_timestamp',    table_name='anomaly_scores')
    op.drop_table('anomaly_scores')