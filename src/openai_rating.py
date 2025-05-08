import pandas as pd
import numpy as np
import os
import json
from ai_models import send_messages, send_prompt
from util import extract_single_numbers

def split(text):
    text = text.replace("• ", "")
    text = text.split("\n")
    
    res = ""
    for index, sub in enumerate(text):
        res += f"{index+1}. {sub} "

        
    return res



df_interest = pd.read_excel("../dataset/CABIN_info.xlsx")
df_interest["Illustrative Occupations"] = df_interest["Illustrative Occupations"].apply(split)
df_interest["Questionnaire Items"] = df_interest["Questionnaire Items"].apply(split)
df_interest.tail()



df_occup = pd.read_excel("../dataset/Occ_info.xlsx")
df_occup["Task"] = df_occup["Task"].apply(split)
df_occup.tail()



def get_prompt(basic_interest, title, description, task, definition, questionnaire_items, illustrative_occupations):
    return f"""Your job is to rate how descriptive the basic_interest “{basic_interest}” is of the occupation “{title}.”
DESCRIPTION ({title}): {description}
TASKS ({title}): {task}
DEFINITION ({basic_interest}): {definition}
INTEREST INVENTORY ITEMS ({basic_interest}): {questionnaire_items}
ILLUSTRATIVE OCCUPATIONS ({basic_interest}): {illustrative_occupations}
Use this scale: 
1 = not at all descriptive
2 = slightly descriptive 
3 = somewhat descriptive
4 = moderately descriptive
5 = quite descriptive
6 = highly descriptive
7 = extremely descriptive
Return only the number.
"""



import concurrent.futures
import os
import time

def process_item(ai_model, interest_row, occupation_row):
    try:
        log_path = f"./logs/{ai_model}_{interest_row}_{occupation_row}.txt"

        # skip if exists
        if os.path.exists(log_path):
            print("-", end="", flush=True)
            return ai_model, interest_row, occupation_row

        
        prompt = get_prompt(
            basic_interest = df_interest.loc[interest_row, "Basic Interest"], 
            title = df_occup.loc[occupation_row, "Title"],  
            description = df_occup.loc[occupation_row, "Description"],  
            task = df_occup.loc[occupation_row, "Task"],  
            definition = df_interest.loc[interest_row, "Definition"], 
            questionnaire_items = df_interest.loc[interest_row, "Illustrative Occupations"], 
            illustrative_occupations= df_interest.loc[interest_row, "Questionnaire Items"], 
        )
        resp = send_prompt(prompt, model=ai_model)
        
        os.makedirs("./logs", exist_ok=True)
        
        with open(log_path, "w") as file:
            file.write(resp)
        
        print(".", end="", flush=True)
        time.sleep(1) # for claud, wait for 2 seconds
        return ai_model, interest_row, occupation_row
    except Exception as e:
        with open(f"./logs/error.log", "a") as file:
            file.write(f"{ai_model}_{interest_row}_{occupation_row}\n")
        with open(f"./logs/{ai_model}_{interest_row}_{occupation_row}.log", "w") as file:
            file.write(str(e))
        print()
        print(e, flush=True)
        


if __name__ == "__main__":
    with open(f"./logs/error.log", "w") as file:
        file.write("")

    tasks = [] 
    for ai_model in [ "gpt-4o"]: # "gpt-4o", "deepseek-chat",
        for interest_row in range(len(df_interest)):
            for occupation_row in range(len(df_occup)):
                tasks.append((ai_model, interest_row, occupation_row))
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(process_item, *task) for task in tasks]
        
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
            except Exception as e:
                print(f"\nError: {e}")
                
    print("Finished!")