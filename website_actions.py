from transformers import CLIPModel, CLIPProcessor
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from PIL import Image
import time
import os
import torch
from transformers import CLIPProcessor, CLIPModel
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
import openai
app = FastAPI()


class URLInput(BaseModel):
    url: str


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
    """Detects the genre of the website from the image using a classification model."""
    # Placeholder: Load and use a pre-trained genre classification model
    # Here we use a mock function for demonstration purposes
    # Replace this with actual genre classification code
    genres = ["e-commerce", "social media", "news", "blog",
              "education", "corporate", "entertainment", "forum"]
    # Simulate genre detection
    return genres[0]  # Assume it always returns 'e-commerce' for demo


# can be used in future
def generate_potential_actions(genre: str) -> list:
    """Generates potential actions based on the detected genre using OpenAI API."""
  
    prompt = f"Given the genre '{genre}', suggest a list of potential actions a user might take on a website of this genre."

    try:
        
        # use the commented code if we have enough access to chatgpt api key
        # llm=OpenAI(temperature=0.7,model_name="gpt-3.5-turbo")
        # response=llm(prompt)

        # actions = response.choices[0].message['content']

        # return actions.split(',')

        #using a list of actions for now
        actions = ['Read next post', 'Add friend', 'Enter shipping address', 'Bookmark article', 'Add to cart', 'Choose payment method', 'Share on social media', 'View product details', 'Share article', 'Subscribe to newsletter', 'Download content', 'Like photo/video', 'Proceed to checkout', 'Follow user', 'View posts', 'Leave a comment', 'Apply discount code', 'Send a message', 'Confirm order', 'Subscribe to blog', 'Select size/color',
                   "Order Food",
                   "Login",
                   "Signup",
                   "Get Delivery",
                   "Find Restaurant",
                   "Write Review",
                   "View Menu",
                   "Reserve Table",
                   "Check Reviews",
                   "Search Dishes",
                   "Get Directions",
                   "View Offers",
                   "View Ratings",
                   "Update Profile",
                   "Manage Orders",
                   "Provide Feedback",
                   "Contact Support",
                   "Browse Categories",
                   "Save Favorite",
                   "Share Experience",
                   "Explore Nearby",
                   "Check Availability"]

        return actions
    except openai.error.OpenAIError as e:
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
        # Step 1: Take a screenshot of the webpage
        screenshot_path = take_screenshot(url_input.url, save_path)

        # Step 2: Detect the genre of the website
        genre = detect_genre_from_image(screenshot_path)

        # Step 3: Generate potential actions based on the detected genre using LangChain
        potential_actions = generate_potential_actions(genre)
        print(potential_actions)

        # Step 4: Extract features from the screenshot using CLIP
        features = extract_features_from_image(
            screenshot_path, potential_actions)

        return {
            "url": url_input.url,
            "features": features
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


