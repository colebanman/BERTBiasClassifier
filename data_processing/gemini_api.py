# backend/gemini.py
import os
import time
from google.generativeai.types import HarmCategory, HarmBlockThreshold

import google.generativeai as genai
import json

# REDACTED API KEY
# genai.configure(api_key="")

def upload_to_gemini(path, mime_type=None):
  """Uploads the given file to Gemini.

  See https://ai.google.dev/gemini-api/docs/prompting_with_media
  """
  file = genai.upload_file(path, mime_type=mime_type)
  print(f"Uploaded file '{file.display_name}' as: {file.uri}")
  return file

def wait_for_files_active(files):
  """Waits for the given files to be active.

  Some files uploaded to the Gemini API need to be processed before they can be
  used as prompt inputs. The status can be seen by querying the file's "state"
  field.

  This implementation uses a simple blocking polling loop. Production code
  should probably employ a more sophisticated approach.
  """
  print("Waiting for file processing...")
  for name in (file.name for file in files):
    file = genai.get_file(name)
    while file.state.name == "PROCESSING":
      print(".", end="", flush=True)
      time.sleep(10)
      file = genai.get_file(name)
    if file.state.name != "ACTIVE":
      raise Exception(f"File {file.name} failed to process")
  print("...all files ready")
  print()

# Create the model
# See https://ai.google.dev/api/python/google/generativeai/GenerativeModel
generation_config = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 64,
  "max_output_tokens": 100000,
  "response_mime_type": "application/json",
}

model = genai.GenerativeModel(
  model_name="gemini-1.5-flash",
  generation_config=generation_config,
  safety_settings ={
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    },
  # See https://ai.google.dev/gemini-api/docs/safety-settings
  system_instruction="""You are a sophisticated AI trained to identify and categorize various forms of bias present within textual content. Your goal is to analyze articles and pinpoint sentences exhibiting potential bias, striving for a balanced representation of different bias categories in your output. Fix any major grammatical errors, but don't change any sentiment. Ensure to process every sentence of the article, skipping none, and decode any Unicode characters. Never return a raw Unicode character; replace them with the most appropriate character or string. Only output actual sentences; ignore any non-sentence data such as HTML tags, captions, or other anomalies. Every single sentence must be an actual, valid, human-reproducible sentence.

Here's your process:

1. **Thorough Understanding:** Carefully read the entire article, including the title. Develop a comprehensive understanding of the topic, the author's potential stance, and the overall tone.

2. **Sentence-Level Analysis:** Examine each sentence individually for potential bias, considering the language, framing, and any underlying assumptions or generalizations. Decode any Unicode characters during this process, replacing them with the most appropriate characters or strings. Only consider valid, human-reproducible sentences, ignoring any non-sentence data such as HTML tags, captions, or other anomalies.

3. **Bias Identification and Labeling:** For each sentence, determine if it exhibits any of the following biases and assign the corresponding label. Only use one label per sentence. The bias label is the bias being exerted by the author. Absolutely ensure that every entry includes both a "sentence" and a "label".

   * **Political Bias:**
     * **"left"**: Sentence leans towards a liberal or progressive political perspective.
     * **"right"**: Sentence leans towards a conservative or right-wing political perspective.

   * **Cognitive Bias:**
     * **"confirmation"**: Sentence presents information likely to reinforce pre-existing beliefs.

   * **Social Bias:**
     * **"cultural"**: Sentence reflects biases about a specific culture's traditions, beliefs, or practices.
     * **"gender"**: Sentence perpetuates stereotypes or inequalities based on gender.
     * **"racial"**: Sentence expresses prejudice against individuals or groups based on race or ethnicity.
     * **"age"**: Sentence displays bias towards a particular age group, often through stereotypes.
     * **"class"**: Sentence demonstrates prejudice based on socioeconomic status.

   * **"unbiased"**: If a sentence doesn't exhibit any of the above biases, label it as "unbiased."

4. **Balanced Representation:** Strive to identify examples of EACH bias category present within the text. If one category is significantly more prevalent, prioritize finding examples of less represented biases, but do not invent bias where it doesn't exist.

5. **Detailed Output:** Present your analysis in JSON format:
   ```json
   [
     {"sentence": "Sentence from the article.", "label": "left"},
     {"sentence": "Another sentence.", "label": "cultural"},
     ...
   ]
   ```""",
)

# TODO Make these files available on the local file system
# You may need to update the file paths
files = [
  upload_to_gemini(".txt", mime_type="text/plain"),
]

# Some files have a processing delay. Wait for them to be ready.
wait_for_files_active(files)

allSentences = json.loads(open(".json").read())
lastSentence = allSentences[-1]
history=[
    {
        "role": "user",
        "parts": [
        files[0],
        ],
    },
]

# Add chat history to the model. For every five sentences, add the user saying "Please respond with an analysis for the next 5 sentences." and the model responding with the analysis.
for i in range(0, len(allSentences), 10):
    part = allSentences[i:i+10]
    history.append(
        {
            "role": "user",
            "parts": [
                {
                    "text": "Please respond with an analysis for the next 10 sentences.",
                },
            ],
        },
    )
    history.append(
        {
            "role": "model",
            "parts": [
                {
                    "text": json.dumps(part),
                }
            ],
        },
    )

chat_session = model.start_chat(
    history=history,
)


# Prompt for 10 sentences at a time
firstRun = True
while True:
    try:
        if firstRun:
            response = chat_session.send_message(
                f"Please respond with an analysis for the next 10 sentences, starting after the sentence: '{lastSentence['sentence']}'",
            )
            firstRun = False
        else:
            response = chat_session.send_message(
                "Please respond with an analysis for the next 10 sentences.",
            )
        text = response.text
        jsonText = json.loads(text)
        allSentences.extend(jsonText)

        print(text)

        # Save allSentences to a file
        with open(".json", "w") as f:
            json.dump(allSentences, f, indent=4)
    except Exception as e:
        print(e)
        print("429 Error: Too many requests. Waiting for 60 seconds...")
        time.sleep(60)
        continue