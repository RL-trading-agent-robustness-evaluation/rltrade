from datetime import timedelta, datetime, timezone
import requests
from pathlib import Path
import time

dir=Path(__file__).resolve().parent
rl=dir.parent / "data"


columns = ["date", "open", "high", "low", "close", "volume", "divCash", "splitFactor"]
ticker="spy"

#time
offset=timedelta(hours=-4)
timezone=timezone(offset)
cur_time=datetime.now(timezone)
formatted_time = cur_time.strftime("%Y-%m-%d")

url=f"https://api.tiingo.com/tiingo/daily/{ticker}/prices?startDate=2019-01-02&endDate={formatted_time}&format=csv&resampleFreq=daily&columns={','.join(columns)}&token=bbd6787d039372ccb417d1de316de0d28fb586ef"
headers = {
        'Content-Type': 'application/json'
        }
requestResponse = requests.get(url, headers=headers)

#print("status:", requestResponse.status_code)
#print("content-type:", requestResponse.headers.get("Content-Type"))
#print("body:", requestResponse.text[:500])

if requestResponse.status_code == 200:

        target=rl/(f"{ticker}_price.csv")
        with open(target, "w") as f:
                f.write(requestResponse.text)

else:
        print("Error: status ", requestResponse.status_code, requestResponse.text)
