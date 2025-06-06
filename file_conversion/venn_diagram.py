from openai import OpenAI
from dotenv import load_dotenv
import time
import os
import matplotlib.pyplot as plt
from matplotlib_venn import venn2, venn3
import json
from matplotlib_venn import venn2_unweighted

def generate_venn_diagram(file_path = None):
  
  load_dotenv(dotenv_path=".env", override=True)
  client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

  # 1. Define the two-person conversation
  if file_path:
    with open(file_path, 'r', encoding='utf-8') as file:
      conversation_text = file.read()
  else:
    conversation_text = """
    User: I think we should prioritize user experience above all.
    Speaker: Yes, user experience is crucial for success.
    User: I'm aiming for a product launch in September.
    Speaker: I believe October would be a safer target for launch.
    User: Also, I suggest we keep the design minimalistic.
    Speaker: I agree, minimalism helps clarity.
    """
    
  # conversation_text = """
  # User: I think we should prioritize user experience above all.
  # Speaker: Yes, user experience is crucial for success.
  # User: I'm aiming for a product launch in September.
  # Speaker: I believe October would be a safer target for launch.
  # User: Also, I suggest we keep the design minimalistic.
  # Speaker: I agree, minimalism helps clarity.
  # """

  # conversation_text = """
  # User: I think we should prioritize user experience above all. I'm aiming for a product launch in September.Also, I suggest we keep the design minimalistic.
  # Speaker: Yes, user experience is crucial for success. I believe October would be a safer target for launch. I agree, minimalism helps clarity.
  # """

  # 2. Define the system prompt to extract agreement and divergence
  prompt = f"""
  Analyze the following conversation between User and Speaker.

  Conversation:
  \"\"\"
  {conversation_text}
  \"\"\"

  Instructions:
  - Extract key points from each speaker very concisely, only include the basic information (using short phrases).
  - Group points into:
    - "agreement" (shared by both)
    - "User" (only User's opinions)
    - "Speaker" (only Speaker's opinions)
  - Output only valid JSON format as:

  {{
    "agreement": [
      "shared point 1",
      "shared point 2"
    ],
    "only_User": [
      "only User point 1",
      "only User point 2"
    ],
    "only_Speaker": [
      "only Speaker point 1",
      "only Speaker point 2"
    ]
  }}

  Important Note: 
  - If a statement by Person A is already reflected in the agreement points, do not repeat it in "only_A".
  - Similarly for Person B.
  - Keep points non-redundant and logically distinct across agreement/only_A/only_B.

  Output only raw JSON without triple backticks.
  Do not add any extra text or explanations outside the JSON.
  """

  # 3. Query the LLM
  response = client.chat.completions.create(
      model="gpt-4o",  # or gpt-4
      messages=[{"role": "user", "content": prompt}],
      temperature=0.2  # keep it deterministic
  )

  # 4. Parse the output
  output_text = response.choices[0].message.content
  print("\n--- LLM Analysis Output ---\n")
  print(output_text)

  # For simplicity, you manually split the output here
  # (can automate later if needed)
  # try:
  structured_data = json.loads(output_text)
  print("\n--- Parsed Structured Data ---\n")
  print(json.dumps(structured_data, indent=2))
  # except json.JSONDecodeError:
  #     print("Failed to parse LLM output. Check formatting.")

  # 5. Extract lists
  agreement = structured_data.get("agreement", [])
  only_a = structured_data.get("only_User", [])
  only_b = structured_data.get("only_Speaker", [])

  # 5. Draw the Venn diagram
  plt.figure(figsize=(10,8))
  venn = venn2_unweighted(
      subsets=(len(only_a), len(only_b), len(agreement)),
      set_labels=('User', 'Speaker')
  )

  # Optionally, set custom labels
  venn.get_label_by_id('10').set_text('\n'.join(only_a))
  venn.get_label_by_id('01').set_text('\n'.join(only_b))
  venn.get_label_by_id('11').set_text('\n'.join(agreement))

  plt.title("Agreement and Divergence Between User and Speaker")
  plt.show()
  
generate_venn_diagram(None)