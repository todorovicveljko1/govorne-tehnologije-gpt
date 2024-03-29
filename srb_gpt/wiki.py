import re
import time
import random

import requests
from bs4 import BeautifulSoup

def get_link_and_parse(link):
    """
    Function to retrieve a webpage, parse it using BeautifulSoup,
    and return the BeautifulSoup object.

    Args:
        link (str): The URL of the webpage to retrieve and parse.

    Returns:
        BeautifulSoup: The parsed HTML content of the webpage.
    """
    response = requests.get(link)
    soup = BeautifulSoup(response.content, 'html.parser')
    return soup

def get_wiki_links(soup, base_url):
    """
    Function to extract Wikipedia links from a BeautifulSoup object.

    Args:
        soup (BeautifulSoup): The BeautifulSoup object representing the parsed HTML content.

    Returns:
        list: A list of Wikipedia link URLs found in the parsed content.
    """
    all_links = soup.find(id="bodyContent").find_all("a")
    wiki_links = []
    for link in all_links:
        if link.has_attr('href') and link['href'].find("/wiki/") != -1 and link.has_attr("title") and link['href'].find(":") == -1:
            wiki_links.append(base_url + link['href'])
    return wiki_links


def break_text(text, max_length=80):
    """
    Breaks the text after a certain number of characters or words.

    Args:
        text (str): The text to be broken.
        max_length (int): Maximum length of each line.
    
    Returns:
        str: The text with lines broken.
    """
    text = re.sub(r'(.{1,' + str(max_length) + r'}(?:\s|$))', r'\1\n', text, flags=re.DOTALL)
    text = re.sub(r' \n', r'\n ', text)
    return text


def clean_text(text):
    """ 
    clean text
    """
    pattern = r"\[.*?\]" # remove [x] - refrences and [uredi]
    
    text = re.sub(pattern, '', text)
    
    return text


def extract_text_content(soup, break_words = ["Види још", "Галерија", "Референце", "Напомене", "Извори", "Reference", "Spoljašnje veze"]):
    out = ""
    tags = soup.find(id="bodyContent").find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li'])
    break_words = set(break_words)
    for tag in tags:
        text = tag.get_text()
        text = clean_text(text)
        if text.strip() == "" or text.strip() == "\n":
            continue
        if text.strip() in break_words:
            break
        if tag.name == "p" or tag.name == "li": 
            out += break_text(text)
        else:
            out += (text+'\n')

    return out

def break_text_preserving_formulas(text, max_length=80):
    FORUMLA_ID = "FORMULAAAAID"
    # Regular expression pattern to match LaTeX formulas
    formula_pattern = r'{\\displaystyle .*?}'

    # Find all formulas in the text
    formulas = re.findall(formula_pattern, text)

    # Replace formulas with placeholders to temporarily remove them from the text
    text_with_placeholders = re.sub(formula_pattern, FORUMLA_ID, text)

    # break the text
    text_with_placeholders = break_text(text_with_placeholders, max_length)
    #print(formulas)
    # Restore formulas in text
    for formula in formulas:
        text_with_placeholders = text_with_placeholders.replace(FORUMLA_ID, formula, 1)
    return text_with_placeholders

def post_cleanup(text):
    # Clean LaTex forumulas and empty lines with small number of chars <= 3
    out = ""
    for line in text.split("\n"):
        if len(line.strip()) <= 3 and "}" not in line and "{" not in line:
            continue
        out += line
    out = re.sub(r'\n{', '\n {', out)

    out = re.sub(r'\n(\s|\.|\,|\?|\!|\:|\;)', r'\1', out)
    lines = out.split("\n")
    out = ''
    for l in lines:
        out += break_text_preserving_formulas(l)
    return out


def download_wiki_data_around_link(link, base_url, save_to = 'data.txt'):
    text = ""
    # Get root link
    soup = get_link_and_parse(link)
    text += extract_text_content(soup)
    # Get linkes to navigate to and take data
    links = get_wiki_links(soup, base_url)
    unique_links = list(set(links))
    # Get text from each link
    for l in unique_links:
        time.sleep(random.uniform(0.1, 0.3)) # Be nice to servers
        soup = get_link_and_parse(l)
        text += extract_text_content(soup)
    # Additional data cleanup
    text = post_cleanup(text)
    with open(save_to, "w", encoding='utf-8') as file:
        file.write(text)
    return text

    

