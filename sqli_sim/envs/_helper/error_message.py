# Define the error messages that can be returned by the database
import random
from collections import defaultdict

DATABASES = ["MySQL", "PostgreSQL", "SQL Server"]
ESCAPE_CHAR = [
    '\'',
    '"',
    '%20',
    ';',
    '['
]
TOTAL_ESCAPE_CHAR = [''] + ESCAPE_CHAR
FLAGS = ["flag"]
LOCATIONS = ["before the parameter value", "after the parameter value"]
TABLES = ["users", "products", "orders", "admin"]
# a list of common actions to take
TABLE_ACTION = [
    'SELECT',
    'INSERT',
    'UPDATE',
    'DELETE'
]
"""
    'CREATE',
    'DROP',
    'ALTER',
    'TRUNCATE',
    'EXECUTE',
    'UNION SELECT',
"""
ATTACK_ACTION = [
    '1; DROP TABLE users',
    '1); DROP TABLE users',
    '1" OR 1=1;--',
    '1" OR 1=1;DROP TABLE users;--',
    '1\' OR 1=1;--',
    '1\' OR 1=1;DROP TABLE users;--',
    '1 or sleep(10)--',
    '1;SELECT%20SLEEP(10)--',
    '1 AND (SELECT COUNT(*) FROM tabname) = 1',
    '1" AND (SELECT * FROM users WHERE name = "admin") = "admin";--',
    '1\' AND (SELECT * FROM users WHERE name = \'admin\') = \'admin\';--',
    '1\' UNION SELECT 1,2,3,4,5,6,7,8,9,10;--',
    '1" UNION SELECT 1,2,3,4,5,6,7,8,9,10;--'
]


class Attack:
    OR_PROB = 0.2
    DROP_PROB = 0.1
    SLEEP_PROB = 0.1
    ONE_PROB = 0.5

    def __init__(self, attack_actions):
        self.ATTACK_ACTION = [
            self.generate_attack_string(escape, table, table_action) for
            _ in range(attack_actions // (len(ESCAPE_CHAR) + len(TABLES) + len(TABLE_ACTION))) for
            escape in ESCAPE_CHAR for
            table in TABLES for
            table_action in TABLE_ACTION
        ]
        self.num_actions = attack_actions // (len(ESCAPE_CHAR) + len(TABLES) + len(TABLE_ACTION))
        self.attack_map = {
            (escape, table, table_action, i): self.generate_attack_string(escape, table, table_action) for
            i in range(self.num_actions) for
            escape in ESCAPE_CHAR for
            table in TABLES for
            table_action in TABLE_ACTION
        }
        self.ATTACK_ACTION = list(self.attack_map.values())
        self.TABLE_ACTION = TABLE_ACTION
        self.ACTIONS = self.TABLE_ACTION + self.ATTACK_ACTION

        self.escape_map = {
            escape: (escape, self.attack_map[(escape, table, table_action, i)])
            for i in range(self.num_actions) for
            escape in ESCAPE_CHAR for
            table in TABLES for
            table_action in TABLE_ACTION
        }

        self.escape_map = defaultdict(list)
        for i in range(self.num_actions):
            for escape in ESCAPE_CHAR:
                for table in TABLES:
                    for table_action in TABLE_ACTION:
                        self.escape_map[escape].append((escape, self.attack_map[(escape, table, table_action, i)]))

        self.escape_list = []
        for escape in self.escape_map:
            self.escape_list += [(escape, None)]
            for table_action in TABLE_ACTION:
                self.escape_list.append((escape, table_action))
            for table_action in TABLE_ACTION:
                for table in TABLES:
                    self.escape_list.append((escape, f"{table_action} IN '{table}'"))
            for table_action in TABLE_ACTION:
                for table in TABLES:
                    self.escape_list.append((escape, self.generate_attack_string(escape, table, table_action)))
            # self.escape_list += self.escape_map[escape]

        self.FLAG_ATTACK = []
        self.flag_map = {}
        for action, (escape, injection) in enumerate(self.escape_list):
            if 21 > action % 37 > 0:
                f = action % 37
                e = action // 37
                if 5 > f > 0:
                    pass
                else:
                    self.flag_map[(f, e)] = action + 16
                    self.FLAG_ATTACK.append(self.escape_list[action + 16][1])

    def generate_attack_string(self, escape_char, table, table_action):
        where_clause = 'WHERE ' + table + '.id = 1'
        or_clause = 'OR ' + table + '.id = 2'
        drop_clause = 'DROP TABLE ' + table
        sleep_clause = 'sleep(10)'
        one_clause = '1'
        end_clause = random.choice([";--", "--", ""])

        attack_string = ""
        if random.random() < self.ONE_PROB:
            attack_string += f"{one_clause}"

        attack_string += f"{escape_char} {table_action} FROM {table} {where_clause}"
        if random.random() < self.OR_PROB:
            attack_string += f" {or_clause}"
        if random.random() < self.DROP_PROB:
            attack_string += f" {drop_clause}"
        if random.random() < self.SLEEP_PROB:
            attack_string += f" {sleep_clause}"

        attack_string += end_clause
        return attack_string


class ERROR_MESSAGES:
    error_messages = {
        "SELECT": {
            "MySQL": [
                "You have an error in your SQL syntax; check the manual that corresponds to your MySQL server version for the right syntax to use near 'X' at line 1",
                "Unknown column 'X' in 'field list'",
                "Table 'X' doesn't exist",
                "Access denied for user 'X'@'localhost' (using password: YES)"
            ],
            "PostgreSQL": [
                "syntax error at or near 'X'",
                "column \"X\" does not exist",
                "relation \"X\" does not exist",
                "password authentication failed for user \"X\""
            ],
            "SQL Server": [
                "Incorrect syntax near 'X'.",
                "Invalid column name 'X'.",
                "Invalid object name 'X'.",
                "Login failed for user 'X'."
            ]
        },
        "INSERT": {
            "MySQL": [
                "You have an error in your SQL syntax; check the manual that corresponds to your MySQL server version for the right syntax to use near 'X' at line 1",
                "Duplicate entry 'X' for key 'PRIMARY'",
                "Table 'X' doesn't exist",
                "Access denied for user 'X'@'localhost' (using password: YES)"
            ],
            "PostgreSQL": [
                "syntax error at or near 'X'",
                "duplicate key value violates unique constraint 'X_pkey'",
                "relation \"X\" does not exist",
                "password authentication failed for user \"X\""
            ],
            "SQL Server": [
                "Incorrect syntax near 'X'.",
                "Violation of PRIMARY KEY constraint 'X_PK'. Cannot insert duplicate key in object 'dbo.X'.",
                "Invalid object name 'X'.",
                "Login failed for user 'X'."
            ]
        },
        "UPDATE": {
            "MySQL": [
                "You have an error in your SQL syntax; check the manual that corresponds to your MySQL server version for the right syntax to use near 'X' at line 1",
                "Table 'X' doesn't exist",
                "Access denied for user 'X'@'localhost' (using password: YES)"
            ],
            "PostgreSQL": [
                "syntax error at or near 'X'",
                "relation \"X\" does not exist",
                "password authentication failed for user \"X\""
            ],
            "SQL Server": [
                "Incorrect syntax near 'X'.",
                "Invalid object name 'X'.",
                "Login failed for user 'X'."
            ]
        },
        "DELETE": {
            "MySQL": [
                "You have an error in your SQL syntax; check the manual that corresponds to your MySQL server version for the right syntax to use near 'X' at line 1",
                "Table 'X' doesn't exist",
                "Access denied for user 'X'@'localhost' (using password: YES)"
            ],
            "PostgreSQL": [
                "syntax error at or near 'X'",
                "relation \"X\" does not exist",
                "password authentication failed for user \"X\""
            ],
            "SQL Server": [
                "Incorrect syntax near 'X'.",
                "Invalid object name 'X'.",
                "Login failed for user 'X'."
            ]
        }
    }

    def get(self):
        error_messages = self.error_messages.copy()
        for operation in self.error_messages:
            for db in self.error_messages[operation]:
                for i in range(len(self.error_messages[operation][db])):
                    for table in TABLES:
                        if "X" in self.error_messages[operation][db][i]:
                            error_msg = self.error_messages[operation][db][i].replace("X", table)
                            error_messages[operation][db].append(error_msg)

        return error_messages


def error_messages():
    msgs = {}
    for operation in TABLE_ACTION:
        msgs[operation] = {}
        for db in DATABASES:
            # msgs[operation][db] = []
            errors = []
            for table in TABLES:
                for loc in LOCATIONS:
                    error_msg = f"{db} ERROR: FOR {operation} IN '{table}' {loc.upper()}"
                    errors.append(error_msg)
                    error_msg = f"{db} ERROR: NEAR {operation}"
                    errors.append(error_msg)
                    error_msg = f"ERROR: NEAR {operation}"
                    errors.append(error_msg)

            msgs[operation][db] = random.choice(errors)

    return msgs


def table_error_messages():
    msgs = {}
    for operation in TABLE_ACTION:
        msgs[operation] = {}
        for db in DATABASES:
            msgs[operation][db] = {}
            for table in TABLES:
                msgs[operation][db][table] = f"{db} ERROR: FOR {operation} IN '{table}'"

            msgs[operation][db][None] = f"{db} ERROR: NEAR {operation}"

    return msgs
