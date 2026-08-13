"""merge enterprise-kb and main heads

Revision ID: c9d03e6dd01f
Revises: 856c5b02fb54, a7b8c9d0e1f2
Create Date: 2026-07-22 06:49:53.819330

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import open_webui.internal.db


# revision identifiers, used by Alembic.
revision: str = 'c9d03e6dd01f'
down_revision: Union[str, None] = ('856c5b02fb54', 'a7b8c9d0e1f2')
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
