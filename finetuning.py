from openai import OpenAI

client = OpenAI()

response = client.files.create(
  file=open("data.jsonl", "rb"),
  purpose="fine-tune"
)

print(response)

client.fine_tuning.jobs.create(
  training_file=response.id,
  model="gpt-4o-2024-08-06",
)