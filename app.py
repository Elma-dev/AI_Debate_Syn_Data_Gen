import gradio as gr
from openai import OpenAI
import os
from tqdm import tqdm
import pandas as pd
from pathlib import Path

model_base_url={}

language="MOROCCAN Arabic"
SYSTEM_PROMPT = {
            "role": "system",
            "content":  f"""This is a context-based Q&A game where two AIs interact with a user-provided context. All interactions MUST be in {language}.

            QUESTIONER_AI:
            - Must only ask questions that can be answered from the provided context
            - Should identify key information gaps or unclear points
            - Must quote or reference specific parts of the context
            - Cannot ask questions about information not present in the context
            - Must communicate exclusively in {language}

            ANSWERER_AI:
            - Must only answer using information explicitly stated in the context
            - Cannot add external information or assumptions
            - Must indicate if a question cannot be answered from the context alone
            - Must communicate exclusively in {language}"""
        }

def add_model(model_name,base_url,api_key):
    model_base_url[model_name]=base_url
    model_quest.choices=list(model_base_url.keys())
    os.environ[model_name]=api_key
    return gr.Dropdown(label="Questioner Model",choices=list(model_base_url.keys())),gr.Dropdown(label="Answerer Model",choices=list(model_base_url.keys()))


def model_init(model):
    try:
        api_key=os.environ.get(model)
        base_url=model_base_url[model]
        client = OpenAI(api_key=api_key, base_url=base_url)
        return client
    except Exception as e:
         print(f"You should add api key of {model}")

# generate questions
def init_req_messages(sample_context):
  messages_quest=[
      SYSTEM_PROMPT,
      {
            "role":"user",
            "content":f"""Context for analysis:
            {sample_context}
            As QUESTIONER_AI, generate a question based on this context.
            """
      }
  ]
  return messages_quest
# generate Answers
def init_resp_messages(sample_context,question):
  messages_answ=[
      SYSTEM_PROMPT,
      {
          "role": "user",
          "content": f"""
          Context for analysis:
          {sample_context}
          Question: {question}
          As ANSWERER_AI, answer this question using only information from the context.
          """}

  ]
  return messages_answ

def chat_generation(client,model_name,messages):
  return client.chat.completions.create(
    model=model_name,
    messages=messages,
    temperature=0.5
    ).choices[0].message.content

def generate_question(client,model_name,messages_quest):
  question=chat_generation(client,model_name,messages_quest)
  messages_quest.append({"role":"assistant","content":question})
  return question

def generate_answer(client,model_name,messages_answ):
  answer=chat_generation(client,model_name,messages_answ)
  messages_answ.append({"role":"assistant","content":answer})
  return answer

def save_conversation(conversation):
    conv_flat={"user":[],"assistant":[]}
    for i in range(0,len(conversation)):
        conv_flat[conversation[i]["role"]].append(conversation[i]["content"])
    df=pd.DataFrame(conv_flat)
    df.to_csv("data.csv")
    return Path("data.csv").name

def user_input(context,model_a,model_b,num_rounds,conversation_history):
    conversation_history.clear()
    client_quest=model_init(model_a)
    client_ans=model_init(model_b)
    messages_quest=init_req_messages(context)
    for round_num in tqdm(range(num_rounds)):
            question = generate_question(client_quest,model_a,messages_quest)
            conversation_history.append(
                {"role":"user","content":question},
            )
            if round_num==0:
              messages_answ=init_resp_messages(context,question)
            else:
              messages_answ.append({"role":"user","content":question})
            answer = generate_answer(client_ans,model_b,messages_answ)
            messages_quest.append({"role":"user","content":answer})
            conversation_history.append(
                {"role":"assistant","content":answer},
            )
    file_path=save_conversation(conversation_history)
    
    return conversation_history,gr.DownloadButton(label="Save Conversation",value=file_path,visible=True)

with gr.Blocks() as demo:
    gr.Markdown("""
                <h1 style="text-align: center;">Mohadata: Debate Data Generator 🤖</h1>

                This tool generates a debate-style conversation between two AI models based on a given **context**. It simulates a question-answer dialogue, where one model acts as the questioner and the other as the answerer. The conversation is generated iteratively, with each model responding to the previous message from the other model.

                To use this tool:
                * First, add information about the models you want to use by clicking the "+" button and filling in the required details.
                * simply enter the **context** of the debate in the provided text box.
                * select the models you want to use for the **questioner** and **answerer**.
                * specify the **number of rounds** you want the conversation to last.
                * click the **"Submit"** button to generate the conversation.
                * download the conversation by clicking the **"Download Conversation"** button.

                The conversation will be displayed in the chatbot window, with the questioner's messages on the right and the answerer's messages on the left. This tool can be useful for generating debate-style conversations on a given topic, and can help in understanding different perspectives and arguments on a particular issue.
            """)
    with gr.Row("compact"):
        model_name=gr.Textbox(label="Model Name",placeholder="Enter Model Name")
        base_url=gr.Textbox(label="Base URL",placeholder="Enter Base URL")
        api_key=gr.Textbox(label="API Key",placeholder="Enter API Key",type="password")
    add=gr.Button("+",variant="huggingface")    
    with gr.Row(equal_height=True):
            context=gr.Textbox(label="context",lines=3)
    with gr.Row():
        model_quest=gr.Dropdown(label="Questioner Model",choices=list(model_base_url.keys()))
        model_answ=gr.Dropdown(label="Answerer Model",choices=list(model_base_url.keys()))
        num_rounds=gr.Number(label="Number Rounds",minimum=1)
    with gr.Row():
        submit=gr.Button("Submit",variant="primary")
    with gr.Row():
         chatbot=gr.Chatbot(
              type="messages",rtl=True
         )
    with gr.Row():
        save=gr.DownloadButton(label="Download Conversation",visible=False)
    add.click(
        add_model,
        inputs=[model_name,base_url,api_key],
        outputs=[model_quest,model_answ]
    )   
    submit.click(
         user_input,
         inputs=[context,model_quest,model_answ,num_rounds,chatbot],
         outputs=[chatbot,save])
    

    
demo.launch()
