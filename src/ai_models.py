from openai import OpenAI 
import os
import anthropic

models = {
    "openai": ["gpt-3.5-turbo", "gpt-4o", "gpt-4o-mini"],
    "deepseek": ["deepseek-chat", "deepseek-reasoner"],
    "claude": ["claude-3-7-sonnet-latest", "claude-3-5-haiku-latest"],
}

 ##############################################################################################################################################
 ## AI clients
 ##############################################################################################################################################
 
 # need to add the api keys here before running the code

client_openai = OpenAI(
    # api_key = ""
)


client_deepseek = OpenAI(
    # api_key = "",
    base_url = "https://api.deepseek.com",
)

# client_claude = anthropic.Anthropic(api_key="")

 ##############################################################################################################################################
 ## functions for setting openai model parameters and sending to openai
 ##############################################################################################################################################

client_mappers = {"openai": client_openai, "deepseek": client_deepseek, "claude": client_claude}

model_mappers = {
    model: category for category in models for model in models[category]
}
model_client_mappers = {
    model: client_mappers[category] for category in models for model in models[category]
}



def send_messages(messages, model="gpt-4o-mini", temperature=0, max_tokens=300):
    client = model_client_mappers[model]

    if model_mappers[model] == "claude":
        message = client.messages.create(
            model=model,
            temperature=temperature,
            # system="You are a world-class poet. Respond only with short poems.",
            messages=messages,
            max_tokens=max_tokens,
        )
        return message.content[0].text
    else:
        response = client.chat.completions.create(
            model= model, 
            messages=messages,
            temperature=temperature
        )

        return response.choices[0].message.content



def send_prompt(prompt, model="gpt-4o-mini", temperature=0, max_tokens=300):
    client = model_client_mappers[model]

    if model_mappers[model] == "claude":
        message = client.messages.create(
            model=model,
            temperature=temperature,
            # system="You are a world-class poet. Respond only with short poems.",
            messages=[{"role":"user", "content": prompt}],
            max_tokens=max_tokens,
        )
        
        return message.content[0].text
    else:
    
        response = client.chat.completions.create(
            model= model, 
            messages=[{"role":"user", "content": prompt}],
            temperature=temperature
        )
        return response.choices[0].message.content
