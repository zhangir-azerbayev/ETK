import sqlite3
import pandas as pd
import json
from tqdm import tqdm

# This entire script is super inefficient. 
# anyone looking through my github who's considering hiring me, look away! 
# in my defense, I only need to run this once ever. 

def get_schema_string(db_name): 
    con = sqlite3.connect(f"database/{db_name}/{db_name}.sqlite")
    cur = con.cursor()
    cur1 = con.cursor()

    query = "SELECT name FROM sqlite_master WHERE type='table';"

    cur.execute(query)
    tables=[]
    for table_name in cur: 
        prompt = "# " + table_name[0].title() + "("
        df = pd.read_sql_query(f"SELECT * FROM {table_name[0]}", con)
        prompt += ", ".join(df.columns) + ")"
        tables.append(prompt)

    return "\n".join(tables)

with open("train_spider.json") as f: 
    data = json.load(f)

for i in tqdm(range(len(data))): 
    db_name = data[i]["db_id"]
    question = data[i]["question"]

    schema = get_schema_string(db_name)

    prompt = schema + "\n### " + question + "\nSELECT"

    data[i]["prompt"] = prompt
    data[i]["task_id"] = i 

with open("train_spider_with_prompts.json", "w") as f: 
    json.dump(data, f)
   
