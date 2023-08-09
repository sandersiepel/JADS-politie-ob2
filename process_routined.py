import sqlite3
import pandas as pd

df_original = pd.read_pickle("routined/routined_pickle.pkl")

# cnx = sqlite3.connect('routined/Cache.sqlite')
# df_sqlite = pd.read_sql_query("SELECT * FROM ZRTCLLOCATIONMO", cnx)
# df_sqlite.ZTIMESTAMP = pd.to_datetime(df_sqlite.ZTIMESTAMP + 978307200, unit='s') + pd.Timedelta(hours=2)

# df = pd.concat([df_original, df_sqlite])
# df = df.sort_values(by="ZTIMESTAMP").drop_duplicates()

# df.to_pickle('routined/routined_pickle.pkl')
# df.to_pickle('data/Routined.pkl')

import plotly.express as px

df_original['count'] = 1
df_resample = df_original.set_index('ZTIMESTAMP').resample('D').sum()

fig = px.line(df_resample, y="count", width=1000, height=600)
fig.show()