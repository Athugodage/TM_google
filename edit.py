from selenium import webdriver

from selenium.webdriver import Chrome
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait as Wait
from selenium.webdriver.support import expected_conditions as EC


from webdriver_manager.chrome import ChromeDriverManager

import re

driver = webdriver.Chrome(ChromeDriverManager().install())

def correct(label):
    query = re.sub(' ', '+', label)

    link = f'https://nova.rambler.ru/search?query={query}'
    driver.get(link)
    xpath = '/html/body/div[1]/div/div[3]/div/div[1]/div/div[1]/div/div[1]/div/b'
    element = Wait(driver, 0.15).until(EC.presence_of_element_located((By.XPATH, xpath)))
    return element.text

#print(correct('метод в картинка'))

topics = []
labels = []


with open('result.txt', 'r', encoding='utf8') as f:
    lines = f.read().split('\n')
    for line in lines:
        if 'TOPIC:' in line:
            topics.append(line)
        elif 'LABELS:' in line:
            labels.append(line.strip('LABELS: '))



with open('new_result.txt', 'w') as w:
    for n in range(len(topics)):
        w.write(topics[n])
        w.write('\n')
        new_labels = []
        for label in labels[n].split(', '):
            try:
                new_labels.append(correct(label))
            except:
                new_labels.append(label)
        print(new_labels)
        w.write(f'LABELS: {new_labels}')
        w.write('\n\n')
