sf_columns = ['symbol', 'timestamp', 'open', 'high', 'low', 'close', 'volume']
stonkDb_colums = list(sf_columns)
stonkDb_colums[1] = 'time'
sf_data_columns = stonkDb_colums[2:]
