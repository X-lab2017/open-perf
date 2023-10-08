import sqlite3

class DataStorage:

    def __init__(self, db_name=":memory:"):
        self.conn = sqlite3.connect(db_name)
        self.cursor = self.conn.cursor()

    def store_data(self, table_name, data):
        # 将DataFrame转化为SQL
        data.to_sql(table_name, self.conn, if_exists='replace', index=False)

    def retrieve_data(self, table_name):
        query = f"SELECT * FROM {table_name}"
        return pd.read_sql(query, self.conn)

    def close_connection(self):
        self.conn.close()
