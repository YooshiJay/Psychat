#!/usr/bin/env python
# coding: utf-8

# In[1]:


# ## import pandas
# # print(pandas.__version__)

# !pip install -q bitsandbytes datasets accelerate loralib
# !pip install -q git+https://github.com/huggingface/transformers.git@main git+https://github.com/huggingface/peft.git
# !pip install transformers_stream_generator einops
# !pip install openai
# !pip install cohere
# !pip install tiktoken
# !pip install gradio

# !pip uninstall transformers -y
# !pip install transformers==4.36.0


# In[2]:


class args:
    max_seq_length = 1024
    model_name = "/gemini/data-2"
    peft_model_id = "/gemini/pretrain"
    prompt = "你是一位心理咨询师，现在以温暖亲切的语气，与来访者进行对话，请主动多次询问来访者的内心诉求，\
              更注重共情和尊重来访者的感受。根据来访者的反馈调整回应，确保回应贴合来访者的情境和需求。"


# In[3]:


import torch
from peft import PeftModel, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import GenerationConfig
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4")

model = AutoModelForCausalLM.from_pretrained(
    args.model_name,
    device_map='auto',
    return_dict=True,
    quantization_config=quantization_config,
)

tokenizer = AutoTokenizer.from_pretrained(args.model_name)
tokenizer.im_start_id = tokenizer.encode('<|im_start|>')[0]
tokenizer.im_end_id = tokenizer.encode('<|im_end|>')[0]
tokenizer.eos_token = tokenizer.pad_token

model = prepare_model_for_kbit_training(model)

# Load the Lora model
model = PeftModel.from_pretrained(model, args.peft_model_id)
model.eval()


# In[4]:


import pandas as pd
import torch
from peft import PeftModel, PeftConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import GenerationConfig
from transformers import BitsAndBytesConfig


def generate_and_tokenize_prompt_for_interface(ask, history):
    system = args.prompt

    system_text = f'<|im_start|>system\n{system}<|im_end|>\n'
    input_ids = tokenizer.encode(system_text, add_special_tokens=False)

    # 拼接多轮对话
    for i, conv in enumerate(history):
        human = conv[0].strip()
        assistant = conv[1].strip()

        input_tokens = tokenizer.encode(f'<|im_start|>user\n{human}<|im_end|>\n', add_special_tokens=False)
        output_tokens = tokenizer.encode(f'<|im_start|>assistant\n{assistant}<|im_end|>' + tokenizer.eos_token + '\n', add_special_tokens=False)

        input_ids += input_tokens + output_tokens

    # 拼接当前提问
    input_ids += tokenizer.encode(f'<|im_start|>user\n{ask}<|im_end|>\n<|im_start|>assistant\n', add_special_tokens=False)

    # 对长度进行截断
    input_ids = input_ids[-args.max_seq_length:]
    input_ids = torch.tensor([input_ids]).cuda()
    return input_ids

###
# 输入:
# data:高三后的迷茫，高考前的恐惧，能给我一些建议么？
# history:[{"human": "你好呀", "assistant": "你好，我是xxx，很高兴为您服务"}, ...]

generation_config = GenerationConfig(
        temperature=0.35,
        top_p = 0.9,
        repetition_penalty=1.0,
        max_new_tokens=500,  # max_length=max_new_tokens+input_sequence
        # min_new_tokens = 1,
        do_sample = True,
#         eos_token_id=tokenizer.eos_token_id
)


# In[5]:


import gradio as gr
import time

def generate_answer(human, history):
    input_ids = generate_and_tokenize_prompt_for_interface(human, history)

    with torch.no_grad():
        s = model.generate(input_ids=input_ids, generation_config=generation_config)

    output = tokenizer.decode(s[0], skip_special_tokens=True)
    assistant = output[output.rfind("assistant")+10:]
    return assistant

def slow_echo(message, history):
    # 其中message来自于前端界面的输入信息，即用户的提问
    # history为历史问答记录，此处可以忽略
    # text也来自于前端界面的输入信息，为参考文本
    answer = generate_answer(message, history)
    for i in range(len(answer)):
        time.sleep(0.05)
        yield answer[: i+1]

# demo2 = gr.ChatInterface(
#     slow_echo,
# ).queue()
#   # 我们不强制要求你做成流输出的样式，只要能够得到回答即可。

# title = "💖 安心落意——专注心理咨询的大语言模型"

with gr.Blocks() as demo:
    # 创建一个Markdown组件，用来显示提示信息
    gr.Markdown('<h1 style="color: #333; font-size: 24px; margin-bottom: 20px; text-align: center;">💖 安心落意——专注心理咨询的大语言模型</h1>')
    gr.Markdown('<p style="font-family: Arial;">\
                     🔥 我们的心理对话模型旨在帮助那些面临情绪困扰、心理压力或人际交往困难的个人。</p>')
    gr.Markdown('<p style="font-family: Arial;">\
                     ⭐ 通过与我们的模型互动，他们可以获得情绪支持、心理指导和解决问题的方法，从而重拾内心的平衡与自信，实现更加健康、积极的生活方式。</p>')
    gr.Markdown('<p style="font-family: Arial;">\
                     🥰 请放心与我们的模型对话，我们郑重承诺不会获取任何个人隐私信息！</p>')
    gr.Markdown('<p style="font-family: Arial;">\
                     ⭕ 请注意：大模型只提供简单帮助，如有必要，请联系专业心理师！模型在训练时非常注意价值观引导，但我们不能保证其输出始终正确且无害。在内容方面，模型的作者和平台不承担相关责任。</p>')
    interface = gr.ChatInterface(slow_echo, 
                                 # title=title
                                ).queue()
    # blocks = gr.Blocks([demo2, md])

if __name__ == "__main__":
    model.eval()
    demo.launch(server_name='127.0.0.1', server_port=7860)


# In[ ]:




