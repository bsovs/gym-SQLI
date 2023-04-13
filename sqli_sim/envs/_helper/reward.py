from sqli_sim.envs._helper.error_message import ESCAPE_CHAR, TABLE_ACTION, TABLES, FLAGS


class Reward:
    def __init__(self, capture=10, escape=-1, rows=-1, error=1, wrong=-1):
        self.capture = capture
        self.error = error
        self.escape = escape
        self.rows = rows
        self.wrong = wrong


class ErrorReward:
    def __init__(self, nothing=-1, db_type=2, flag=100, escape=1, table=2):
        self.nothing = nothing
        self.db_type = db_type
        self.flag = flag
        self.escape = escape
        self.table = table

        escape_values = {self.escape_type(escape): self.escape for escape in ESCAPE_CHAR}
        table_values = {self.action_type(action): self.escape for action in TABLE_ACTION}
        attack_values = {self.action_type(action): self.flag for action in FLAGS}
        tables = {self.table_type(table): self.table for table in TABLES}

        escape_values[self.escape_type("")] = -1

        self.error_values = {}
        self.error_values.update(escape_values)
        self.error_values.update(table_values)
        self.error_values.update(attack_values)
        self.error_values.update(tables)

    @classmethod
    def table_type(cls, table):
        return 'table', table

    @classmethod
    def escape_type(cls, escape):
        return 'escape', escape

    @classmethod
    def action_type(cls, action):
        return 'action', action
