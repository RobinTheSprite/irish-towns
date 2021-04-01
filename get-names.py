from bs4 import BeautifulSoup

soup = BeautifulSoup(open("english-towns-1.html"), features="html5lib")

towns = list(map(BeautifulSoup.get_text, soup.find_all("a")))

towns = list(map(str.strip, towns))

towns = "\n".join(towns)

f = open("english-towns-1.txt", "w")
f.writelines(towns)