import os
import pandas as pd
from pandas.io.sql import SQLTable
import datetime
import sqlalchemy
import csv
from io import StringIO
from typing import Union, List
from . import Types
from . import Constants
from dotenv import load_dotenv
load_dotenv()


cnx = sqlalchemy.create_engine('postgresql://%s:%s@%s/%s' % (os.environ.get('POSTGRES-USER'),
                               os.environ.get('POSTGRES-PASS'), os.environ.get('POSTGRES-URL'), os.environ.get('POSTGRES-DB')))


def load_env_file(path: str):
    load_dotenv(path)
    globals()['cnx'] = sqlalchemy.create_engine('postgresql://%s:%s@%s/%s' % (os.environ.get('POSTGRES-USER'),
                                                                              os.environ.get('POSTGRES-PASS'), os.environ.get('POSTGRES-URL'), os.environ.get('POSTGRES-DB')))


def create_historical_table() -> None:
    cmd = '''
        CREATE TABLE symbols (
        symbol TEXT PRIMARY KEY,
        name TEXT
        );

        CREATE TABLE historical (
        time TIMESTAMP WITHOUT TIME ZONE NOT NULL,
        symbol TEXT,
        open NUMERIC,
        high NUMERIC,
        low NUMERIC,
        close NUMERIC,
        volume NUMERIC,
        FOREIGN KEY (symbol) REFERENCES symbols (symbol),
        UNIQUE (time, symbol)
        );

        SELECT create_hypertable('historical', 'time');

        create index on historical (symbol, time desc);
        '''
    cnx.execute(cmd)


def write_symbols(symbols: List[Types.Symbol]) -> None:
    formated_symbols = ',\n'.join("('%s')" % i.upper() for i in symbols)
    cmd = 'INSERT INTO symbols (symbol) VALUES\n%s\n' % formated_symbols
    cmd += 'ON CONFLICT DO NOTHING;'
    cnx.execute(cmd)


def get_symbols() -> List[Types.Symbol]:
    query = 'SELECT * FROM symbols;'
    symbols = pd.read_sql(query, con=cnx).symbol.to_list()
    return sorted(symbols)


def write_historical(sf: Types.StockFrame) -> None:
    def historical_method(table: SQLTable, conn: sqlalchemy.engine.Engine, keys: list, dataIter):
        with conn.connection.cursor() as cur:
            sBuf = StringIO()
            writer = csv.writer(sBuf)
            writer.writerows(dataIter)
            sBuf.seek(0)

            columns = ', '.join('"%s"' % k for k in keys)
            if table.schema:
                table_name = '%s.%s' % (table.schema, table.name)
            else:
                table_name = table.name
            temp_table_name = 'temp_%s' % table_name
            update_set = ", ".join(
                ['%s=EXCLUDED.%s' % (v, v) for v in keys if v not in ['time', 'symbol']])

            cmd = 'CREATE TEMPORARY TABLE %s (LIKE %s) ON COMMIT DROP;' % (
                temp_table_name, table_name)
            cur.execute(cmd)

            cmd = 'COPY %s (%s) FROM STDIN WITH CSV' % (
                temp_table_name, columns)
            cur.copy_expert(sql=cmd, file=sBuf)

            cmd = 'INSERT INTO %s(%s)\nSELECT %s FROM %s\n' % (
                table_name, columns, columns, temp_table_name)
            cmd += 'ON CONFLICT (time, symbol) DO UPDATE SET %s;' % update_set
            cur.execute(cmd)

            cur.execute('DROP TABLE %s' % temp_table_name)

    sf.index.names = ['time']
    start_timer = datetime.datetime.now()
    write_symbols(sf['symbol'].unique())
    sf.to_sql('historical', con=cnx, if_exists='append',
              index=True, method=historical_method)
    print(sf.symbol[0], len(sf.index), datetime.datetime.now() - start_timer)
    sf.index.names = ['timestamp']


def get_historical(symbols: Union[List[Types.Symbol], None] = None, start: Union[Types.Datetime, None] = None, end: Union[Types.Datetime, None] = None, time_list: Union[List[datetime.datetime], None] = None) -> Types.StockFrame:
    def get_query_date(date: datetime.datetime, end: bool = False):
        if end:
            return date.strftime('%Y-%m-%d 23:59:00')
        else:
            return date.strftime('%Y-%m-%d 00:00:00')

    binder = 'WHERE'
    query_filter = ''
    if symbols is not None and len(symbols) > 0:
        query_filter += 'WHERE symbol in (%s)\n' % (
            ','.join("'%s'" % i.upper() for i in symbols))
        binder = 'AND'
    if start is not None:
        if end is not None:
            query_filter += "%s time BETWEEN '%s' AND '%s'\n" % (
                binder, get_query_date(start), get_query_date(end, True))
        else:
            query_filter += "%s time >= '%s'\n" % (binder,
                                                   get_query_date(start))
        binder = 'AND'
    elif end is not None:
        query_filter += "%s time <= '%s'\n" % (
            binder, get_query_date(end, True))
        binder = 'AND'
    if time_list is not None:
        query_filter += '%s time::time in (%s)' % (binder,
                                                   ', '.join(i.strftime("'%H:%M:00'") for i in time_list))

    query = 'SELECT %s FROM historical\n%s' % (
        ','.join('%s' % i for i in Constants.stonkDb_colums), query_filter)

    start_timer = datetime.datetime.now()
    s_buf = StringIO()
    conn = cnx.raw_connection()
    cur = conn.cursor()
    cmd = "COPY (%s) TO STDOUT WITH (FORMAT CSV, DELIMITER ',')" % query
    cur.copy_expert(cmd, s_buf)
    s_buf.seek(0)

    sf = pd.read_csv(s_buf, names=Constants.stonkDb_colums, parse_dates=[
                     'time']).set_index('time').sort_index()
    sf.index.names = ['timestamp']
    # print(len(sf.index), datetime.datetime.now() - start_timer)
    return sf


def get_historical_date_range():
    start = cnx.execute('SELECT MIN(time) FROM historical').scalar()
    end = cnx.execute('SELECT MAX(time) FROM historical').scalar()
    return start, end


def get_historical_date_list():
    date_list = sorted([i[0] for i in cnx.execute(
        'SELECT DISTINCT time FROM historical')])
    return date_list
