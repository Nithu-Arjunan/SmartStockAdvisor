import os
import json
import itables.options as itable_opts
import pandas as pd
from IPython.display import HTML, Markdown, display
from google.genai.types import GenerateContentConfig, Part
from google import genai


PROJECT_ID = "fast-tensor-464714-p1"
LOCATION = os.environ.get("GOOGLE_CLOUD_REGION", "us-central1")
client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)


# Set Gemini Flash and Pro models to be used in this notebook
GEMINI_FLASH_MODEL_ID = "gemini-2.0-flash-001"
#GEMINI_PRO_MODEL_ID = "gemini-2.0-flash"

# Provide link to a public YouTube video to summarize
YOUTUBE_VIDEO_URL = (
    "https://www.youtube.com/watch?v=vFCcJo83JH4"  # @param {type:"string"}
)



# Call Gemini API with prompt to summarize video
video_summary_prompt = "Give a detailed summary of this video.As the video speaks about a stock, try to capture the sentiment of the video"

video_summary_response = client.models.generate_content(
    model=GEMINI_FLASH_MODEL_ID,
    contents=[
        Part.from_uri(
            file_uri=YOUTUBE_VIDEO_URL,
            mime_type="video/webm",
        ),
        video_summary_prompt,
    ],
)

summary_text = video_summary_response.candidates[0].content.parts[0].text
display(Markdown(summary_text))
print(summary_text)