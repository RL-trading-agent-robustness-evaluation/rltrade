import requests

columns = ["date", "open", "high", "low", "close", "volume", "divCash", "splitFactor"]
ticker="spy"

url=f"https://api.tiingo.com/tiingo/daily/{ticker}/prices?startDate=2019-01-02&endDate=2025-12-31&format=csv&resampleFreq=daily&columns={','.join(columns)}&token=bbd6787d039372ccb417d1de316de0d28fb586ef"
headers = {
        'Content-Type': 'application/json'
        }
requestResponse = requests.get(url, headers=headers)

#print("status:", requestResponse.status_code)
#print("content-type:", requestResponse.headers.get("Content-Type"))
#print("body:", requestResponse.text[:500])

if requestResponse.status_code == 200:

        filename=f"C:/03_大學/大三專題/rltrade/data/{ticker}_price.csv"
        with open(filename, "w") as f:
                f.write(requestResponse.text)

else:
        print("Error: status ", requestResponse.status_code, requestResponse.text)
