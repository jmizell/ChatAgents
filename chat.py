import os
import json
import openai

response = openai.ChatCompletion.create(
  model="gpt-3.5-turbo-16k",
  messages=[
    {
      "role": "system",
      "content": "You're a helpful assistant. You provide concise answer unless prompted for more detail. You avoid providing lists, or advice unprompted. "
    },
    {
      "role": "user",
      "content": "hello?"
    }
  ],
  temperature=1,
  max_tokens=10000,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0
)

print(json.dumps(response))
print(response["choices"][0]["message"]["content"])