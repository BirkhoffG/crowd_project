import time
import requests
from tqdm import tqdm


for i in tqdm(range(4000)):
    response = requests.get("https://www.thispersondoesnotexist.com/image")
    file = open(f"img/img_{i+1}.png", "wb")
    file.write(response.content)
    file.close()
    time.sleep(1)
