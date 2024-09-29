from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from PIL import Image
import time
import os
from transformers import CLIPProcessor, CLIPModel
import google.generativeai as genai
import os
import json

app = FastAPI()

class URLInput(BaseModel):
    url: str

def take_multiple_screenshots(url: str, save_path: str):
    """Takes multiple screenshots by scrolling through the entire page and saves them individually."""
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")

    service = Service(executable_path="/Users/anjalitripathi/anaconda3/envs/LT/lib/python3.10/site-packages/chromedriver_py/chromedriver_mac-arm64")
    driver = webdriver.Chrome(service=service, options=options)

    try:
        driver.get(url)
        time.sleep(2)  # Allow the page to load

        total_height = driver.execute_script("return document.body.scrollHeight")
        viewport_height = driver.execute_script("return window.innerHeight")

        screenshot_paths = []
        scroll_position = 0
        screenshot_index = 0

        while scroll_position < total_height:
            driver.execute_script(f"window.scrollTo(0, {scroll_position});")
            time.sleep(0.5)  # Allow time for content to load after scrolling

            screenshot_path = os.path.join(save_path, f"screenshot_{screenshot_index}.png")
            driver.save_screenshot(screenshot_path)
            screenshot_paths.append(screenshot_path)

            scroll_position += viewport_height
            screenshot_index += 1

        return screenshot_paths

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error taking screenshots: {str(e)}")
    finally:
        driver.quit()

def generate_potential_actions_from_multiple_screenshots(image_paths: list) -> list:
    """Generates potential actions from a batch of multiple screenshots using the Gemini model (LLM)."""
    try:
        images = [genai.upload_file(image_path) for image_path in image_paths]

        # Craft a clear and concise prompt
        prompt = "Analyze the following screenshots of a website and suggest the actions a user can take on this website for example Click on login button, enter username in field, search for an blue bag, etc. Each image represents part of a webpage. Return the actions in a python list named actions.\n"

        # Use the GEMINI API to generate actions for all screenshots
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(images + [prompt])
   
        print(response)

        # Parse the response safely (avoiding eval)
        if response.candidates[0].content.parts[0].text:
            action_list=eval(response.candidates[0].content.parts[0].text.strip("```python\n").strip("``` \n").split("=")[1])
        print(action_list)
        return action_list
  
    except Exception as e:
        print(str(e))
        raise HTTPException(status_code=500, detail=f"Error generating potential actions: {str(e)}")

@app.post("/detect-actions/")
def detect_actions(url_input: URLInput):
    """API endpoint to detect actions from multiple webpage screenshots."""
    save_path = "screenshots"
    os.makedirs(save_path, exist_ok=True)

    try:
        # Step 1: Take multiple screenshots
        screenshot_paths = take_multiple_screenshots(url_input.url, save_path)

        # Step 2: Generate potential actions based on all the screenshots
        potential_actions = generate_potential_actions_from_multiple_screenshots(screenshot_paths)

        output = {
            "url": url_input.url,
            "actions": potential_actions
        }

        return output

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
