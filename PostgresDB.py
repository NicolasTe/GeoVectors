import psycopg2

"""
This is class is an interface to a PostgresDB
"""

class PostgresDB:
    def __init__(self, options):
        self._options = options
        self.pool = None

    def get_connection(self):
        return psycopg2.connect(host=self._options.dbHost,
                                    user=self._options.dbUser,
                                    password=self._options.dbPassword,
                                    database=self._options.dbName)

    def create_connection_pool(self):
        self.pool = []
        for i in range(1):
            self.pool.append(self.get_connection())

    def close_connection_pool(self):
        for c in self.pool:
            c.close()

    def get_pool_connection(self):
        return self.pool.pop()

    def free_pool_connection(self, conn):
        self.pool.append(conn)

