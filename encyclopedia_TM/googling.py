from selenium import webdriver

from selenium.webdriver import Chrome
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait as Wait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains


from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup as bs


import re
import pymorphy2
from razdel import tokenize



import unicodedata
import os

## select interpreter - NewTM

ACCENT_MAPPING = {
    '́': '',
    '̀': '',
    'а́': 'а',
    'а̀': 'а',
    'е́': 'е',
    'ѐ': 'е',
    'и́': 'и',
    'ѝ': 'и',
    'о́': 'о',
    'о̀': 'о',
    'у́': 'у',
    'у̀': 'у',
    'ы́': 'ы',
    'ы̀': 'ы',
    'э́': 'э',
    'э̀': 'э',
    'ю́': 'ю',
    '̀ю': 'ю',
    'я́́': 'я',
    'я̀': 'я',
}
ACCENT_MAPPING = {unicodedata.normalize('NFKC', i): j for i, j in ACCENT_MAPPING.items()}


def unaccentify(s):
    ## чтобы не было проблем с надстрочными символами

    source = unicodedata.normalize('NFKC', s)
    for old, new in ACCENT_MAPPING.items():
        source = source.replace(old, new)
    return source


class googling():
    def __init__(self):

        self.driver = webdriver.Chrome(ChromeDriverManager().install())  ##  инициализация веб-драйвера
        self.morph = pymorphy2.MorphAnalyzer()

    def _parse(self):
        ##  внутренняя функция парсинга

        soup = bs(self.driver.page_source, 'html.parser')
        search = soup.find_all('div', class_="yuRUbf")
        for h in search:
            self.names.append(h.get_text())

    def find(self, word, pages=3):
        ##  ищет первые 30 (по умолчанию) выдач; кол-во страниц можно настраивать

        self.word = word
        self.names = []
        url = "http://www.google.com/search?q=" + self.word
        self.driver.get(url)

        self._parse()  ##  парсим

        for n in range(pages - 1):
            try:
                ##  поиск дополнительных 20
                ##  смотрим страницу 2 и 3 в поисковой выдаче
                ##  в будущем нужно выделить в отдельную функцию (если будем смотреть все 30 выдач)

                action = ActionChains(self.driver)
                action.key_down(Keys.END).perform()  ##  скролим страницу вниз

                xpath = '//*[@id="pnnext"]/span[2]'
                element = Wait(self.driver, 10).until(EC.presence_of_element_located((By.XPATH, xpath)))
                element.click()  ##  нажимаем на "следующий"

                self._parse()  ##  парсим

            except:
                ##  в случае если по этим топикам нет много данных
                pass

        self.clean()

    def quit(self):
        ##  функция выхода из веб-драйвера
        ##  по сути дублирует функцию самого драйвера, но в ООП иначе никак

        self.driver.quit()

    def _lemmatize(self, text):
        ## лемматизация

        doc = unaccentify(text)
        lemmas = [self.morph.parse(token.text)[0].normal_form for token in tokenize(doc)]

        return " ".join(lemmas)

    def clean(self):
        ##  внутренняя функция, призванная очистить от лишних знаков
        ##  надо доработать исходя из того, что требуется в итоге

        self.result = []

        for name in self.names:
            text = re.split('http', name)[0]
            text = re.sub('[-|a-z|.|,|:|0-9|A-z"]', '', text)
            text = re.sub('  ', ' ', text)

            self.result.append(self._lemmatize(text.strip('  ')))

    def show_topics(self):
        ##  для проверки результатов без сохранения в отдельный файл

        print('QUERY:  ', self.word)
        print('TITLES: \n\n')
        for res in self.result:
            print(res)

    def save(self, kuda):
        ##  сохраняем результаты поисковой выдачи (ОТ ОДНОГО ТОПИКА) в отдельный файл

        source = str(kuda) + '.txt'
        with open(source, 'w', encoding="utf-8") as f:
            f.write(f'QUERY: {self.word}\n')
            f.write('TITLES:\n')
            for res in self.result:
                f.write(res)
                f.write('\n')

#os.makedirs('topics_forcomments')


with open('topics_nmf.txt', 'r', encoding="utf-8") as f:
    f = f.read()
    topics = f.split('\n')

    topics = [topic.split(': ')[-1] for topic in topics if topic != '']

    yandex_part = googling()
    for n, topic in enumerate(topics):
        yandex_part.find(topic, pages=4)
        yandex_part.save(f'topics_forcomments/topic_{n}')

    yandex_part.quit()



