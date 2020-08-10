#==================================================================================================
#                       Create Test and Train Set for General Chatbot
#                       Author: Tim Kolb
#                       Created: 17.10.2019
#==================================================================================================

import sqlite3
import pandas as pd

timeframes=['2016-12']


for timeframe in timeframes:
    connection = sqlite3.connect('C:/Users/Tim/Desktop/Documents/02_Python_Projects/05_Trumpbot/Data/Reddit_Data/{}.db'.format(timeframe))
    c = connection.cursor()
    limit = 5000
    last_unix=0
    cur_length=limit
    counter=0
    test_done=False

    while cur_length == limit:
        df = pd.read_sql('SELECT * FROM PARENT_REPLY WHERE unix > {} and parent NOT NULL and score > 0 ORDER BY unix ASC LIMIT {}'.format(last_unix,limit),connection)
        last_unix=df.tail(1)['unix'].values[0]
        cur_length = len(df)

        if not test_done:
            with open('test.from','a',encoding='utf8') as f:
                for content in df['parent'].values:
                    f.write(content+'\n')
            with open('test.to','a',encoding='utf8') as f:
                for content in df['comment'].values:
                    f.write(content+'\n')
            test_done=True
        else:
            with open('train.from','a', encoding='utf8') as f:
                for content in df['parent'].values:
                    f.write(content+'\n')

            with open('train.to','a', encoding='utf8') as f:
                for content in df['comment'].values:
                    f.write(str(content)+'\n')
        counter += 1
        if counter % 20 == 0:
            print(counter*limit,'rows completed so far')