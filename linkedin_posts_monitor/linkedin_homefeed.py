"""
This script initiates a Selenium WebDriver (automated Chrome browser) to log into the fake
LinkedIn account and records the posts appearing on the home page feed. It imitates human-like
scrolling and skips promoted/suggested/reposted/liked posts. The collected posts are saved in the
form of a CSV file and a JSON file. 

LLM is only called to translate non-English posts to English, using simple English words to detect
the language of the post content, minimising API calls and costs.
"""


import time
import json
import csv
import re
import os
from datetime import datetime, timedelta
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import ElementClickInterceptedException
from dotenv import load_dotenv
from openai import AzureOpenAI

# Load environment variables
load_dotenv()

# CONFIGURATION — Adjust as needed
DAYS_LIMIT = 31  # Number of days to look back
SCROLL_PAUSE_SEC = 10
TRANSLATE = True  # Set False to skip translation
OUTPUT_DIR = "linkedin_data"  # Directory to save results

# SCROLL CONFIGURATION
SCROLL_METHOD = "smooth"  # Options: "to_bottom", "fixed_pixels", "by_viewport", "smooth"
SCROLL_PIXELS = 800  # For "fixed_pixels" method - pixels to scroll each time
SCROLL_SPEED = "slow"  # Options: "slow", "medium", "fast" - for smooth scrolling

# Azure OpenAI Configuration for Translation
AZURE_CLIENT = None
try:
    AZURE_CLIENT = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
    )
except Exception as e:
    print(f"⚠️  Warning: Could not initialize Azure OpenAI client: {e}")
    print("   Translation will be disabled.")

def is_english(text):
    """
    Detect if text is in English using multiple heuristics
    
    Args:
        text: Text to check
    
    Returns:
        True if text is likely English, False otherwise
    """
    if not text or len(text.strip()) < 10:
        return True  # Assume short text is English
    
    text_lower = text.lower()
    
    # Check for common English words (more comprehensive list)
    common_english_words = [
        'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i',
        'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at',
        'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she',
        'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their', 'what'
    ]
    
    # Count how many common words appear
    word_count = 0
    for word in common_english_words:
        # Use word boundaries to avoid partial matches
        if f' {word} ' in f' {text_lower} ' or text_lower.startswith(f'{word} ') or text_lower.endswith(f' {word}'):
            word_count += 1
    
    # Calculate percentage of text that is common English words
    words_in_text = len(text_lower.split())
    if words_in_text > 0:
        english_ratio = word_count / min(words_in_text, len(common_english_words))
        # If 30% or more of the words are common English words, consider it English
        if english_ratio >= 0.3:
            return True
    
    # Additional check: look for non-English characters
    # English uses mostly ASCII characters
    non_ascii_count = sum(1 for char in text if ord(char) > 127)
    if non_ascii_count > len(text) * 0.2:  # More than 20% non-ASCII
        return False
    
    # If we found at least 5 common English words, it's probably English
    return word_count >= 5

def translate_to_english(text):
    """
    Translate text to English using Azure OpenAI API
    Only calls the API if text is NOT in English.
    
    Args:
        text: Text to translate (any language)
    
    Returns:
        Translated English text, or original text if already English or translation fails
    """
    if not text or not text.strip():
        return text
    
    # If Azure client not initialized, return original text
    if not AZURE_CLIENT:
        return text
    
    # Check if text is already in English - SKIP API CALL if true
    if is_english(text):
        return text
    
    # Only reach here if text is NOT English
    try:
        response = AZURE_CLIENT.chat.completions.create(
            model="pmo-gpt-4.1-nano",  # Using your configured model
            messages=[
                {
                    "role": "system",
                    "content": "You are a professional translator. Translate the following text to English. Only return the translated text, nothing else."
                },
                {
                    "role": "user",
                    "content": text
                }
            ],
            temperature=0.3,
            max_tokens=2000
        )
        
        translated_text = response.choices[0].message.content.strip()
        print(f"      🌐 Translated non-English post")
        return translated_text
        
    except Exception as e:
        print(f"      ⚠️  Translation failed: {str(e)[:50]}... Keeping original text.")
        return text

def scroll_page(driver, method="smooth", pixels=800, speed="slow"):
    """
    Scroll the page using different methods
    
    Args:
        driver: Selenium WebDriver
        method: Scroll method - "to_bottom", "fixed_pixels", "by_viewport", "smooth"
        pixels: Number of pixels to scroll (for fixed_pixels method)
        speed: Scroll speed - "slow", "medium", "fast" (for smooth method)
    
    Returns:
        Current scroll height after scrolling
    """
    if method == "to_bottom":
        # Original method - jump to absolute bottom
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        
    elif method == "fixed_pixels":
        # Scroll by a fixed number of pixels from current position
        driver.execute_script(f"window.scrollBy(0, {pixels});")
        
    elif method == "by_viewport":
        # Scroll by one viewport height (screen height)
        driver.execute_script("window.scrollBy(0, window.innerHeight);")
        
    elif method == "smooth":
        # Smooth scroll animation - more human-like
        speed_map = {
            "slow": 50,    # Scroll 50px at a time
            "medium": 100, # Scroll 100px at a time
            "fast": 200    # Scroll 200px at a time
        }
        step = speed_map.get(speed, 100)
        
        # Get viewport height
        viewport_height = driver.execute_script("return window.innerHeight;")
        
        # Scroll in small increments
        for i in range(0, viewport_height, step):
            driver.execute_script(f"window.scrollBy(0, {step});")
            time.sleep(0.05)  # Small delay between increments
    
    # Return current scroll position
    return driver.execute_script("return window.pageYOffset + window.innerHeight;")

def parse_relative_date(relative_date_text):
    """
    Convert LinkedIn's relative date format (e.g., '2h', '3d', '1w', '2mo') 
    to actual datetime string
    """
    if not relative_date_text:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Clean the text
    text = relative_date_text.lower().strip()
    
    # Try to extract time information
    now = datetime.now()
    
    # Patterns: "2h ago", "3 days ago", "1 week ago", "2 months ago", etc.
    patterns = [
        (r'(\d+)\s*s(?:ec|econd)?s?\s*(?:ago)?', 'seconds'),
        (r'(\d+)\s*m(?:in|inute)?s?\s*(?:ago)?', 'minutes'),
        (r'(\d+)\s*h(?:r|our)?s?\s*(?:ago)?', 'hours'),
        (r'(\d+)\s*d(?:ay)?s?\s*(?:ago)?', 'days'),
        (r'(\d+)\s*w(?:eek)?s?\s*(?:ago)?', 'weeks'),
        (r'(\d+)\s*mo(?:nth)?s?\s*(?:ago)?', 'months'),
        (r'(\d+)\s*y(?:ear)?s?\s*(?:ago)?', 'years'),
    ]
    
    for pattern, unit in patterns:
        match = re.search(pattern, text)
        if match:
            value = int(match.group(1))
            
            if unit == 'seconds':
                post_time = now - timedelta(seconds=value)
            elif unit == 'minutes':
                post_time = now - timedelta(minutes=value)
            elif unit == 'hours':
                post_time = now - timedelta(hours=value)
            elif unit == 'days':
                post_time = now - timedelta(days=value)
            elif unit == 'weeks':
                post_time = now - timedelta(weeks=value)
            elif unit == 'months':
                post_time = now - timedelta(days=value*30)  # Approximate
            elif unit == 'years':
                post_time = now - timedelta(days=value*365)  # Approximate
            
            return post_time.strftime("%Y-%m-%d %H:%M:%S")
    
    # If no pattern matches, return current time
    return now.strftime("%Y-%m-%d %H:%M:%S")

def login_linkedin(driver, username, password):
    driver.get("https://www.linkedin.com/login")
    driver.find_element(By.ID, "username").send_keys(username)
    driver.find_element(By.ID, "password").send_keys(password)
    driver.find_element(By.XPATH, "//button[@type='submit']").click()
    time.sleep(5)

def parse_post(post_element, debug=False):
    try:
        # Skip promoted/suggested/reposted/liked posts
        ad_badges = post_element.find_elements(By.XPATH, ".//*[contains(text(),'Promoted') or contains(text(),'Suggested') or contains(text(),'Reposted') or contains(text(),'Liked')]")
        if ad_badges:
            if debug:
                print("      ⏭️  Skipped: Promoted/Suggested post")
            return None

        # Try multiple strategies to find author/company name
        author = ""
        
        # Strategy 1: Look for aria-label with person/company name
        try:
            author_links = post_element.find_elements(By.XPATH, ".//a[contains(@href, '/in/') or contains(@href, '/company/')]")
            for author_link in author_links:
                aria_label = author_link.get_attribute("aria-label")
                if aria_label and len(aria_label.strip()) > 0:
                    # Filter out non-name labels
                    if not any(x in aria_label.lower() for x in ['hashtag', 'like', 'comment', 'share', 'repost']):
                        # Clean up the aria-label to extract just the name
                        cleaned_name = aria_label.strip()
                        # Remove "View" prefix patterns
                        cleaned_name = re.sub(r'^View\s+', '', cleaned_name, flags=re.IGNORECASE)
                        # Remove "'s profile" and similar patterns
                        cleaned_name = re.sub(r"'s?\s+(profile|page|link)\s*$", '', cleaned_name, flags=re.IGNORECASE)
                        # Remove "profile" at the end
                        cleaned_name = re.sub(r'\s+profile\s*$', '', cleaned_name, flags=re.IGNORECASE)
                        # Remove "graphic" suffix
                        cleaned_name = re.sub(r',?\s*graphic\.?$', '', cleaned_name, flags=re.IGNORECASE)
                        cleaned_name = re.sub(r'\s+graphic\s+(link|icon)?\s*$', '', cleaned_name, flags=re.IGNORECASE)
                        cleaned_name = cleaned_name.strip()
                        
                        if cleaned_name and len(cleaned_name) > 1:
                            author = cleaned_name
                            if debug and author:
                                print(f"      📝 Found author via aria-label: {author}")
                            break
        except:
            pass
        
        # Strategy 2: Enhanced span selectors with more variations
        if not author:
            author_selectors = [
                ".//span[contains(@class, 'feed-shared-actor__name')]//span[@aria-hidden='true']",
                ".//span[contains(@class, 'update-components-actor__name')]//span[@aria-hidden='true']",
                ".//div[contains(@class, 'update-components-actor__name')]//span[@aria-hidden='true']",
                ".//div[contains(@class, 'update-components-actor')]//span[@dir='ltr']",
                ".//a[contains(@class, 'app-aware-link')]//span[@dir='ltr'][1]",
                ".//span[contains(@class, 'feed-shared-actor__name')]",
                ".//div[contains(@class, 'feed-shared-actor__container-link')]//span[1]",
                ".//a[contains(@class, 'feed-shared-actor__container-link')]//span[not(@aria-hidden='true')]",
                ".//div[contains(@class, 'feed-shared-actor')]//a//span[1]",
                ".//span[contains(@class, 'update-components-actor__title')]//span[1]"
            ]
            for selector in author_selectors:
                try:
                    elem = post_element.find_element(By.XPATH, selector)
                    author = elem.text.strip()
                    if author and len(author) > 0 and not author.startswith('•'):
                        if debug:
                            print(f"      📝 Found author via selector: {author}")
                        break
                except:
                    continue
        
        # Strategy 3: Look for links with profile/company URLs and extract visible text
        if not author:
            try:
                profile_links = post_element.find_elements(By.XPATH, ".//a[contains(@href, '/in/') or contains(@href, '/company/')]")
                for link in profile_links:
                    text = link.text.strip()
                    # Get visible text that's not empty and looks like a name
                    if text and len(text) > 2 and len(text) < 100:
                        # Avoid time/date related text
                        if not any(x in text.lower() for x in ['ago', 'edited', '•', 'follow', 'like', 'comment']):
                            # Clean up the text
                            cleaned_name = text
                            cleaned_name = re.sub(r'^View\s+', '', cleaned_name, flags=re.IGNORECASE)
                            cleaned_name = re.sub(r"'s?\s+(profile|page|link)\s*$", '', cleaned_name, flags=re.IGNORECASE)
                            cleaned_name = re.sub(r'\s+profile\s*$', '', cleaned_name, flags=re.IGNORECASE)
                            cleaned_name = re.sub(r',?\s*graphic\.?$', '', cleaned_name, flags=re.IGNORECASE)
                            cleaned_name = re.sub(r'\s+graphic\s+(link|icon)?\s*$', '', cleaned_name, flags=re.IGNORECASE)
                            cleaned_name = cleaned_name.strip()
                            
                            if cleaned_name and len(cleaned_name) > 1:
                                author = cleaned_name
                                if debug:
                                    print(f"      📝 Found author via profile link text: {author}")
                                break
            except:
                pass
        
        # Strategy 4: Get all text from actor container and extract first meaningful text
        if not author:
            try:
                actor_containers = post_element.find_elements(By.XPATH, ".//div[contains(@class, 'feed-shared-actor') or contains(@class, 'update-components-actor')]")
                for actor_container in actor_containers:
                    text_spans = actor_container.find_elements(By.TAG_NAME, "span")
                    for span in text_spans:
                        text = span.text.strip()
                        # Look for name-like text: not too short, not too long, doesn't contain time indicators
                        if text and 2 < len(text) < 100:
                            if not any(x in text.lower() for x in ['ago', 'edited', 'reposted', '•', 'follow', 'promoted', 'sponsored', 'h ', 'd ', 'w ', 'mo ']):
                                author = text
                                if debug:
                                    print(f"      📝 Found author via actor container text: {author}")
                                break
                    if author:
                        break
            except:
                pass
        
        # Strategy 5: Try to extract from data-id attribute pattern
        if not author:
            try:
                data_id = post_element.get_attribute("data-id")
                if data_id and 'urn:li:activity' in data_id:
                    # Sometimes the post structure has name in a specific location
                    first_link = post_element.find_element(By.XPATH, ".//a[@href][1]")
                    if first_link:
                        author = first_link.text.strip()
                        if author and len(author) > 2:
                            if debug:
                                print(f"      📝 Found author via first link: {author}")
            except:
                pass
        
        # Final validation and cleanup
        if author:
            # Remove any trailing bullets or extra symbols
            author = author.split('•')[0].strip()
            author = author.split('\n')[0].strip()  # Take only first line if multi-line
            
            # Apply comprehensive cleaning patterns to catch any remaining artifacts
            author = re.sub(r'^View\s+', '', author, flags=re.IGNORECASE)
            author = re.sub(r"'s?\s+(profile|page|link)\s*$", '', author, flags=re.IGNORECASE)
            author = re.sub(r'\s+profile\s*$', '', author, flags=re.IGNORECASE)
            author = re.sub(r',?\s*graphic\.?$', '', author, flags=re.IGNORECASE)
            author = re.sub(r'\s+graphic\s+(link|icon)?\s*$', '', author, flags=re.IGNORECASE)
            # Remove any remaining possessive forms at the end (e.g., "Name's")
            author = re.sub(r"'s?\s*$", '', author, flags=re.IGNORECASE)
            author = author.strip()
            
            # Remove trailing commas or periods
            author = author.rstrip('.,')
            
            # Ensure it's not too long (probably captured wrong element)
            if len(author) > 150:
                author = ""
        
        if not author or len(author) < 2:
            if debug:
                print("      ⏭️  Skipped: Could not find author name after all strategies")
            return None

        # Try multiple selectors for relative date
        relative_date = ""
        date_selectors = [
            ".//span[contains(@class, 'feed-shared-actor__sub-description')]",
            ".//span[contains(@class, 'update-components-actor__sub-description')]",
            ".//time",
            ".//*[contains(text(), 'ago') or contains(text(), 'h') or contains(text(), 'd') or contains(text(), 'w')]"
        ]
        
        for selector in date_selectors:
            try:
                elem = post_element.find_element(By.XPATH, selector)
                text = elem.text.strip()
                # Look for time indicators
                if any(indicator in text.lower() for indicator in ['ago', 'h', 'd', 'w', 'mo', 'yr', 'sec', 'min', 'hour', 'day', 'week', 'month', 'year']):
                    relative_date = text
                    break
            except:
                continue
        
        # Convert relative date to absolute datetime
        actual_datetime = parse_relative_date(relative_date) if relative_date else datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Try to click "...more" button to expand full content
        try:
            # Multiple selectors for the "see more" button
            see_more_selectors = [
                ".//button[contains(@aria-label, 'more')]",
                ".//button[contains(text(), '…more')]",
                ".//button[contains(text(), 'see more')]",
                ".//button[contains(@class, 'see-more')]",
                ".//span[contains(@class, 'see-more')]//button",
                ".//button[contains(@aria-label, 'See more')]"
            ]
            
            for selector in see_more_selectors:
                try:
                    see_more_button = post_element.find_element(By.XPATH, selector)
                    # Scroll to the button to make sure it's in view
                    post_element.parent.execute_script("arguments[0].scrollIntoView({block: 'center'});", see_more_button)
                    time.sleep(0.3)  # Short pause to ensure visibility
                    see_more_button.click()
                    time.sleep(0.5)  # Wait for content to expand
                    if debug:
                        print("      🔍 Clicked 'see more' to expand full content")
                    break
                except:
                    continue
        except Exception as e:
            # No "see more" button found or couldn't click - this is fine, post might already be fully visible
            pass

        # Try multiple selectors for content
        content = ""
        content_selectors = [
            ".//div[contains(@class, 'feed-shared-update-v2__description')]",
            ".//div[contains(@class, 'update-components-text')]",
            ".//div[contains(@class, 'feed-shared-text')]",
            ".//span[contains(@class, 'break-words')]"
        ]
        for selector in content_selectors:
            try:
                content_elem = post_element.find_element(By.XPATH, selector)
                content = content_elem.text.strip()
                if content:
                    break
            except:
                continue
        
        if TRANSLATE and content:
            content = translate_to_english(content)
        
        # Extract URLs
        urls = " | ".join([a.get_attribute("href") for a in post_element.find_elements(By.TAG_NAME, "a") 
                          if a.get_attribute("href") and ("http" in a.get_attribute("href"))])

        post_data = {
            "Person/Company name": author,
            "Date of post": actual_datetime,
            "Content of post": content if content else "No content",
            "URLs": urls
        }
        
        if debug:
            print(f"      ✅ Parsed: {author[:30]}... | Date: {actual_datetime} | Content: {len(content)} chars")
        
        return post_data
        
    except Exception as e:
        if debug:
            print(f"      ❌ Error parsing post: {str(e)}")
        return None

def save_results(results, output_dir="linkedin_data"):
    """Save results to CSV and JSON files with timestamp"""
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate timestamp for filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save to CSV
    csv_filename = f"{output_dir}/linkedin_posts_{timestamp}.csv"
    if results:
        with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ["Person/Company name", "Date of post", "Content of post", "URLs"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"✅ CSV saved: {csv_filename}")
    
    # Save to JSON
    json_filename = f"{output_dir}/linkedin_posts_{timestamp}.json"
    with open(json_filename, 'w', encoding='utf-8') as jsonfile:
        json.dump(results, jsonfile, indent=2, ensure_ascii=False)
    print(f"✅ JSON saved: {json_filename}")
    
    return csv_filename, json_filename

def collect_posts(username, password, screenshot_callback=None, status_callback=None):
    """
    Collect LinkedIn posts with optional callbacks for Streamlit display
    
    Args:
        username: LinkedIn username
        password: LinkedIn password
        screenshot_callback: Function to call with screenshot (for Streamlit display)
        status_callback: Function to call with status updates
    """
    chrome_options = Options()
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_argument("--start-maximized")
    driver = webdriver.Chrome(options=chrome_options)
    
    def update_status(message):
        """Helper to update status via callback or print"""
        print(message)
        if status_callback:
            status_callback(message)
    
    def capture_screenshot():
        """Helper to capture and send screenshot"""
        if screenshot_callback:
            try:
                screenshot = driver.get_screenshot_as_png()
                screenshot_callback(screenshot)
            except Exception as e:
                print(f"Screenshot capture failed: {e}")
    
    update_status("🌐 Opening LinkedIn login page...")
    capture_screenshot()
    login_linkedin(driver, username, password)
    update_status("✅ Login successful!")
    capture_screenshot()

    update_status("📰 Navigating to LinkedIn feed...")
    driver.get("https://www.linkedin.com/feed/")
    time.sleep(3)  # Wait for feed to load
    capture_screenshot()
    
    posts = set()
    results = []
    last_height = driver.execute_script("return document.body.scrollHeight")
    start_time = time.time()

    update_status(f"🔄 Starting to collect posts (max 10 minutes)...")
    update_status("   [Debug mode enabled - showing detailed parsing info]\n")
    scroll_count = 0
    
    while True:
        # Try multiple XPath patterns for post elements
        post_elements = []
        post_selectors = [
            "//div[contains(@class,'feed-shared-update-v2')]",
            "//div[contains(@class,'feed-shared-update')]",
            "//div[@data-id and contains(@class, 'feed-shared')]"
        ]
        
        for selector in post_selectors:
            post_elements = driver.find_elements(By.XPATH, selector)
            if post_elements:
                break
        
        update_status(f"   Scroll #{scroll_count + 1}: Found {len(post_elements)} post elements on page")
        
        new_posts = 0
        skipped = 0
        
        for i, post_element in enumerate(post_elements):
            if post_element in posts:
                continue
            
            data = parse_post(post_element, debug=True)
            if data:
                if data not in results:
                    results.append(data)
                    new_posts += 1
            else:
                skipped += 1
            posts.add(post_element)
        
        scroll_count += 1
        update_status(f"   ✅ New posts: {new_posts} | Skipped: {skipped} | Total collected: {len(results)}\n")
        capture_screenshot()

        # Scroll using configured method
        scroll_page(driver, method=SCROLL_METHOD, pixels=SCROLL_PIXELS, speed=SCROLL_SPEED)
        time.sleep(SCROLL_PAUSE_SEC)
        capture_screenshot()
        new_height = driver.execute_script("return document.body.scrollHeight")
        
        if new_height == last_height:
            update_status("   ⚠️  Reached end of feed (no more scrolling)")
            break
        last_height = new_height
        
        # Set a time or post count cutoff
        if (time.time() - start_time) > 60*10:  # 10 minute max
            update_status("   ⏱️  Time limit reached (10 minutes)")
            break

    update_status(f"\n✅ Collection complete! Total posts collected: {len(results)}")
    capture_screenshot()
    
    # Save results to CSV and JSON
    if results:
        csv_file, json_file = save_results(results, OUTPUT_DIR)
        update_status(f"\n📊 Results saved:")
        update_status(f"   - CSV: {csv_file}")
        update_status(f"   - JSON: {json_file}")
    else:
        update_status("⚠️  No posts collected")
    
    update_status("\n🔒 Closing browser...")
    capture_screenshot()
    driver.quit()
    
    return results

# Main execution
if __name__ == "__main__":
    # Get credentials from environment variables or use defaults
    linkedin_username = os.getenv("LINKEDIN_USERNAME", "shaistasharifah@gmail.com")  # fake account that follows VCs of interest
    linkedin_password = os.getenv("LINKEDIN_PASSWORD", "RU+eVC@UM6n/2U*")
    
    print("=" * 60)
    print("LinkedIn Home Page Post Scraper")
    print("=" * 60)
    print("\n📋 Configuration:")
    print(f"   User: {linkedin_username}")
    print(f"   Scroll method: {SCROLL_METHOD}")
    print(f"   Scroll pause: {SCROLL_PAUSE_SEC}s")
    print(f"   Translation: {'Enabled' if TRANSLATE else 'Disabled'}")
    print(f"   Output directory: {OUTPUT_DIR}")
    print(f"   Days limit: {DAYS_LIMIT}")
    print("\n" + "=" * 60 + "\n")
    
    # Start collecting posts from home page feed
    collect_posts(linkedin_username, linkedin_password)
