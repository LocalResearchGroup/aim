"""empty message

Revision ID: 5ae8371b7481
Revises: 11672b13f92c
Create Date: 2021-06-10 14:13:40.552387

"""

import sqlalchemy as sa

from alembic import op


# revision identifiers, used by Alembic.
revision = '5ae8371b7481'
down_revision = '11672b13f92c'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table(
        'dashboards',
        sa.Column('uuid', sa.Text(), nullable=False),
        sa.Column('name', sa.Text(), nullable=True),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.Column('is_archived', sa.Boolean(), nullable=True),
        sa.PrimaryKeyConstraint('uuid'),
    )
    op.create_table(
        'explore_states',
        sa.Column('uuid', sa.Text(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.Column('is_archived', sa.Boolean(), nullable=True),
        sa.Column('chart_focused_step', sa.Integer(), nullable=True),
        sa.Column('chart_focused_metric_run_hash', sa.Text(), nullable=True),
        sa.Column('chart_focused_metric_metric_name', sa.Text(), nullable=True),
        sa.Column('chart_focused_metric_trace_context', sa.Text(), nullable=True),
        sa.Column('chart_focused_circle_active', sa.Boolean(), nullable=True),
        sa.Column('chart_focused_circle_run_hash', sa.Text(), nullable=True),
        sa.Column('chart_focused_circle_metric_name', sa.Text(), nullable=True),
        sa.Column('chart_focused_circle_step', sa.Integer(), nullable=True),
        sa.Column('chart_focused_circle_trace_context', sa.Text(), nullable=True),
        sa.Column('chart_focused_circle_param', sa.Text(), nullable=True),
        sa.Column('chart_focused_circle_content_type', sa.Text(), nullable=True),
        sa.Column('chart_settings_zoom_mode', sa.Boolean(), nullable=True),
        sa.Column('chart_settings_single_zoom_mode', sa.Boolean(), nullable=True),
        sa.Column('chart_settings_zoom_history', sa.Text(), nullable=True),
        sa.Column('chart_settings_highlight_mode', sa.Text(), nullable=True),
        sa.Column('chart_settings_persistent_display_outliers', sa.Boolean(), nullable=True),
        sa.Column('chart_settings_persistent_zoom', sa.Text(), nullable=True),
        sa.Column('chart_settings_persistent_interpolate', sa.Boolean(), nullable=True),
        sa.Column('chart_settings_persistent_indicator', sa.Boolean(), nullable=True),
        sa.Column('chart_settings_persistent_x_alignment', sa.Text(), nullable=True),
        sa.Column('chart_settings_persistent_x_scale', sa.Integer(), nullable=True),
        sa.Column('chart_settings_persistent_y_scale', sa.Integer(), nullable=True),
        sa.Column('chart_settings_persistent_points_count', sa.Integer(), nullable=True),
        sa.Column('chart_settings_persistent_smoothing_algorithm', sa.Text(), nullable=True),
        sa.Column('chart_settings_persistent_smooth_factor', sa.Float(), nullable=True),
        sa.Column('chart_settings_persistent_aggregated', sa.Boolean(), nullable=True),
        sa.Column('chart_hidden_metrics', sa.Text(), nullable=True),
        sa.Column('chart_tooltip_options_display', sa.Boolean(), nullable=True),
        sa.Column('chart_tooltip_options_fields', sa.Text(), nullable=True),
        sa.Column('search_query', sa.Text(), nullable=True),
        sa.Column('search_v', sa.Integer(), nullable=True),
        sa.Column('search_input_value', sa.Text(), nullable=True),
        sa.Column('search_input_select_input', sa.Text(), nullable=True),
        sa.Column('search_input_select_condition_input', sa.Text(), nullable=True),
        sa.Column('context_filter_group_by_color', sa.Text(), nullable=True),
        sa.Column('context_filter_group_by_style', sa.Text(), nullable=True),
        sa.Column('context_filter_group_by_chart', sa.Text(), nullable=True),
        sa.Column('context_filter_group_against_color', sa.Boolean(), nullable=True),
        sa.Column('context_filter_group_against_style', sa.Boolean(), nullable=True),
        sa.Column('context_filter_group_against_chart', sa.Boolean(), nullable=True),
        sa.Column('context_filter_aggregated_area', sa.Text(), nullable=True),
        sa.Column('context_filter_aggregated_line', sa.Text(), nullable=True),
        sa.Column('context_filter_seed_color', sa.Integer(), nullable=True),
        sa.Column('context_filter_seed_style', sa.Integer(), nullable=True),
        sa.Column('context_filter_persist_color', sa.Boolean(), nullable=True),
        sa.Column('context_filter_persist_style', sa.Boolean(), nullable=True),
        sa.Column('color_palette', sa.Integer(), nullable=True),
        sa.Column('sort_fields', sa.Text(), nullable=True),
        sa.Column('table_row_height_mode', sa.Text(), nullable=True),
        sa.Column('table_columns_order_left', sa.Text(), nullable=True),
        sa.Column('table_columns_order_middle', sa.Text(), nullable=True),
        sa.Column('table_columns_order_right', sa.Text(), nullable=True),
        sa.Column('table_columns_widths', sa.Text(), nullable=True),
        sa.Column('table_excluded_fields', sa.Text(), nullable=True),
        sa.Column('screen_view_mode', sa.Text(), nullable=True),
        sa.Column('screen_panel_flex', sa.Float(), nullable=True),
        sa.Column('dashboard_id', sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(
            ['dashboard_id'],
            ['dashboards.uuid'],
        ),
        sa.PrimaryKeyConstraint('uuid'),
    )
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_table('explore_states')
    op.drop_table('dashboards')
    # ### end Alembic commands ###
