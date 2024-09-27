from transformers import CLIPModel, CLIPProcessor
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from PIL import Image
import time
import os
from transformers import CLIPProcessor, CLIPModel
from langchain.llms import OpenAI
import google.generativeai as genai
import os
app = FastAPI()


class URLInput(BaseModel):
    url: str

def take_fullpage_screenshot(url: str, save_path: str):
    """Takes a full-page screenshot by scrolling through the entire page."""
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")

    service = Service(executable_path="/Users/anjalitripathi/anaconda3/envs/LT/lib/python3.10/site-packages/chromedriver_py/chromedriver_mac-arm64")
    driver = webdriver.Chrome(service=service, options=options)

    try:
        driver.get(url)
        time.sleep(2)  # Allow the page to load

        # Get the total height of the page
        total_height = driver.execute_script("return document.body.scrollHeight")
        viewport_height = driver.execute_script("return window.innerHeight")

        stitched_image = Image.new("RGB", (driver.execute_script("return document.body.scrollWidth"), total_height))

        scroll_position = 0
        while scroll_position < total_height:
            driver.execute_script(f"window.scrollTo(0, {scroll_position});")
            time.sleep(0.5)  # Allow time for content to load after scrolling

            # Take a screenshot of the visible area
            screenshot_path = os.path.join(save_path, f"screenshot_{scroll_position}.png")
            driver.save_screenshot(screenshot_path)

            # Open the screenshot and paste it into the stitched image
            screenshot = Image.open(screenshot_path)
            stitched_image.paste(screenshot, (0, scroll_position))

            scroll_position += viewport_height

        # Save the final stitched image
        full_screenshot_path = os.path.join(save_path, "full_screenshot.png")
        stitched_image.save(full_screenshot_path)

        return full_screenshot_path
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error taking full-page screenshot: {str(e)}")
    finally:
        driver.quit()

def take_screenshot(url: str, save_path: str):
    """Uses Selenium to take a screenshot of the given URL."""
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")

    service = Service(
        executable_path="/Users/anjalitripathi/anaconda3/envs/LT/lib/python3.10/site-packages/chromedriver_py/chromedriver_mac-arm64")
    driver = webdriver.Chrome(service=service, options=options)

    try:
        driver.get(url)
        time.sleep(2)
        screenshot_path = os.path.join(save_path, "screenshot.png")
        driver.save_screenshot(screenshot_path)
        return screenshot_path
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error taking screenshot: {str(e)}")
    finally:
        driver.quit()


# can be used in future
def detect_genre_from_image(image_path: str) -> str:
    """Detects the genre of the website from the image using CLIP or a fine-tuned model."""
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Define potential website genres for classification
    genres = ["order food", "hotel booking", "clothes", "online_courses","e-commerce","education", "social media", "news", "blog", "corporate", "entertainment", "forum"]

    image = Image.open(image_path)

    # Prepare inputs for CLIP
    inputs = processor(text=genres, images=image, return_tensors="pt", padding=True)
    outputs = model(**inputs)

    logits_per_image = outputs.logits_per_image  # similarity scores
    probs = logits_per_image.softmax(dim=1)  # probabilities

    # Get the genre with the highest probability
    detected_genre_idx = probs.argmax().item()
    detected_genre = genres[detected_genre_idx]

    return detected_genre
# can be used in future
def generate_potential_actions(genre: str) -> list:
    """Generates potential actions based on the detected genre using GEMINI API."""
  
    prompt = f"Given the genre '{genre}', suggest a list of potential actions a user might take on a website of this genre. Return a python list"

    try:

        # use the commented code if we have enough access to chatgpt api key
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])

        model = genai.GenerativeModel(model_name="gemini-1.5-flash")
        response = model.generate_content(prompt)
        if response.candidates[0].content.parts[0].text:
            action_list=eval(response.candidates[0].content.parts[0].text.strip("```").strip("``` \n").split("=")[1])
        print(action_list)
        return action_list
    except Exception as e:
        print(f"Error generating actions: {str(e)}")
        raise HTTPException(
            status_code=500, detail="Error generating potential actions.")


def extract_features_from_image(image_path: str, actions: list):
    """Extracts features from the image using CLIP and suggests actions with confidence scores."""
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    image = Image.open(image_path)

    # Prepare inputs for CLIP
    inputs = processor(text=actions, images=image,
                       return_tensors="pt", padding=True)
    outputs = model(**inputs)

    logits_per_image = outputs.logits_per_image  # similarity scores
    probs = logits_per_image.softmax(dim=1)  # convert to probabilities

    # Create a dictionary of actions with their probabilities
    action_scores = {action: prob.item()
                     for action, prob in zip(actions, probs[0])}
    threshold = 0.02
    filtered_actions = {action: score for action,
                        score in action_scores.items() if score > threshold}

    # Optionally sort actions by confidence score in descending order
    sorted_actions = dict(sorted(filtered_actions.items(),
                          key=lambda item: item[1], reverse=True))

    return sorted_actions


@app.post("/detect-actions/")
def detect_actions(url_input: URLInput):
    """API endpoint to detect actions from a webpage."""
    save_path = "screenshots"
    os.makedirs(save_path, exist_ok=True)

    try:
        # Step 1: Take a full-page screenshot of the webpage
        full_screenshot_path = take_fullpage_screenshot(url_input.url, save_path)

        # Step 2: Detect the genre of the website
        genre = detect_genre_from_image(full_screenshot_path)
        # print(genre)

        # Step 3: Generate potential actions based on the detected genre
        potential_actions = generate_potential_actions(genre)
        # print(potential_actions)

        # Step 4: Extract features from the screenshot using CLIP
        features = extract_features_from_image(full_screenshot_path, potential_actions)

        output={
            "url": url_input.url,
            "genre":genre,
            "features": features
        }
        print(output)

        return output

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
