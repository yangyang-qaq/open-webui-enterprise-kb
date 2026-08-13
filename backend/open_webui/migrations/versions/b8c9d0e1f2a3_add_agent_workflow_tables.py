"""add agent workflow tables

Revision ID: b8c9d0e1f2a3
Revises: c9d03e6dd01f
Create Date: 2026-07-22 00:00:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = 'b8c9d0e1f2a3'
down_revision: Union[str, None] = 'c9d03e6dd01f'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    existing = set(inspector.get_table_names())

    if 'agent_workflow' not in existing:
        op.create_table('agent_workflow',
            sa.Column('id', sa.Text(), nullable=False),
            sa.Column('user_id', sa.Text(), nullable=False),
            sa.Column('name', sa.Text(), nullable=False),
            sa.Column('description', sa.Text(), nullable=True),
            sa.Column('created_at', sa.BigInteger(), nullable=False),
            sa.Column('updated_at', sa.BigInteger(), nullable=False),
            sa.PrimaryKeyConstraint('id'),
        )

    if 'agent_workflow_step' not in existing:
        op.create_table('agent_workflow_step',
            sa.Column('id', sa.Text(), nullable=False),
            sa.Column('workflow_id', sa.Text(), nullable=False),
            sa.Column('order_index', sa.BigInteger(), nullable=False),
            sa.Column('agent_role', sa.Text(), nullable=False),
            sa.Column('knowledge_id', sa.Text(), nullable=True),
            sa.Column('prompt_template', sa.Text(), nullable=True),
            sa.Column('input_var', sa.Text(), nullable=True),
            sa.Column('output_var', sa.Text(), nullable=True),
            sa.Column('created_at', sa.BigInteger(), nullable=False),
            sa.ForeignKeyConstraint(['workflow_id'], ['agent_workflow.id'], ondelete='CASCADE'),
            sa.PrimaryKeyConstraint('id'),
        )


def downgrade() -> None:
    op.drop_table('agent_workflow_step')
    op.drop_table('agent_workflow')
