#==================================================================================================
#                       Main Script for General Chatbot
#                       Author: Tim Kolb
#                       Created: 16.10.2019
#==================================================================================================

import sqlite3
import json
from datetime import datetime

timeframe='2016-12'
sql_transaction=[]

connection=sqlite3.connect('{}.db'.format(timeframe))
c=connection.cursor()

### Create the SQLite-Table
def create_table():
    c.execute('''CREATE TABLE IF NOT EXISTS PARENT_REPLY
    (parent_id TEST PRIMARY KEY, comment_id TEXT UNIQUE, 
    parent TEXT, comment TEXT, subreddit TEXT, unix INT, 
    score INT)''')

### Remove certain elements from the posts
def format_data(data):
    data = data.replace("\n", "  newlinechar  ").replace("\r", "  newlinechar  ").replace('"', "'")
    return data


### This searches the parent ID of a post
def find_parent(pid):
    try:
        sql = "SELECT comment FROM PARENT_REPLY WHERE comment_id = '{}' LIMIT 1".format(pid)
        c.execute(sql)
        result = c.fetchone()
        if result != None:
            return result[0]
        else: 
            return False
    except Exception as e:
        #print("find_parent", e)
        return False


def find_existing_score(pid):
    try:
        sql = "SELECT score FROM PARENT_REPLY WHERE parent_id = '{}' LIMIT 1".format(pid)
        c.execute(sql)
        result = c.fetchone()
        if result != None:
            return result[0]
        else: 
            return False
    except Exception as e:
        #print("find_parent", e)
        return False


def acceptable(data):
    if len(data.split(' ')) > 50 or len(data) < 1:
        return False
    elif len(data) > 1000:
        return False
    elif data == '[deleted]' or data == '[removed]':
        return False
    else:
        return True


def transaction_bldr(sql):
    global sql_transaction
    sql_transaction.append(sql)
    if len(sql_transaction) > 1000:
        c.execute('BEGIN TRANSACTION')
        for s in sql_transaction:
            try:
                c.execute(s)
            except:
                pass
        connection.commit()
        sql_transaction = []





def sql_insert_replace_comment(commentid,parentid,parent,comment,subreddit,time,score):
    try:
        sql = """UPDATE parent_reply SET parent_id = ?, comment_id = ?, parent = ?, comment = ?, subreddit = ?, unix = ?, score = ? WHERE parent_id =?;""".format(parentid, commentid, parent, comment, subreddit, int(time), score, parentid)
        transaction_bldr(sql)
    except Exception as e:
        print('s-UPDATE insertion',str(e))

def sql_insert_has_parent(commentid,parentid,parent,comment,subreddit,time,score):
    try:
        sql = """INSERT INTO parent_reply (parent_id, comment_id, parent, comment, subreddit, unix, score) VALUES ("{}","{}","{}","{}","{}",{},{});""".format(parentid, commentid, parent, comment, subreddit, int(time), score)
        transaction_bldr(sql)
    except Exception as e:
        print('s-PARENT insertion',str(e))

def sql_insert_no_parent(commentid,parentid,comment,subreddit,time,score):
    try:
        sql = """INSERT INTO parent_reply (parent_id, comment_id, comment, subreddit, unix, score) VALUES ("{}","{}","{}","{}",{},{});""".format(parentid, commentid, comment, subreddit, int(time), score)
        transaction_bldr(sql)
    except Exception as e:
        print('s-NO_PARENT insertion',str(e))


if __name__=="__main__":
    create_table()
    row_counter = 0
    paired_rows = 0

    with open('C:/Users/Tim/Desktop/Documents/02_Python_Projects/05_Trumpbot/Data/Reddit_Data/RC_2016-12', buffering=1000) as f:
        for row in f:
            #print(row)
            row_counter += 1
            row = json.loads(row)
            #parent_id = row['parent_id']
            parent_id = row['parent_id'].split('_')[1]
            body = format_data(row['body'])
            created_utc = row['created_utc']
            score = row['score']
            subreddit = row['subreddit']
            comment_id = row['id']
            parent_data = find_parent(parent_id)

            if score >= 3:
                if acceptable(body): 
                    existing_comment_score = find_existing_score(parent_id)
                    #if existing_comment_score:
                    #    if score > existing_comment_score:
                    #        sql_insert_replace_comment(comment_id, parent_id, parent_data, body, subreddit, created_utc, score)
                    #else:
                    if parent_data:
                        sql_insert_has_parent(comment_id, parent_id, parent_data, body, subreddit, created_utc, score)
                        paired_rows += 1
                    # Insertion of the no parents comments is necessary to identify the parents
                    else:
                        sql_insert_no_parent(comment_id, parent_id, body, subreddit, created_utc, score)
        
        
            if row_counter % 100000 == 0:
                print("Total rows read: {}, Paired rows: {}, Time: {}".format(row_counter, paired_rows, str(datetime.now())))