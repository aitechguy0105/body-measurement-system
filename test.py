import requests
from datetime import datetime
from datetime import timedelta
datetime_cur = datetime.utcnow()
date_cur = datetime_cur.date()
date_delta = timedelta(days = 10)
date_start = date_cur - date_delta
date_delta = timedelta(days = 0)
date_end = date_cur - date_delta
gold_price_api_key = '7gRDFMlHumWeRX7R0ImtHwdkKDhvGRL1'

limit = 5000
url = f"https://api.polygon.io/v2/aggs/ticker/C:XAUUSD/range/1/day/{date_start}/{date_end}?adjusted=true&sort=asc&limit={limit}&apiKey={gold_price_api_key}"

response = requests.get(url)

data = response.json()
lt_result = []
highest_price = 0
lowest_price = 9999999999999
for item in data['results']:

    date_obj = datetime.fromtimestamp(item['t']/1000)
    day_of_week = date_obj.strftime("%A")
    date_str = date_obj.strftime("%Y-%m-%d")
    # print(date_str, day_of_week,  item['o'], item['h'], item['l'])
    if highest_price < item['h']:
        highest_price = item['h']
    if lowest_price > item['l']:
        lowest_price = item['l']
    item_result = {'date': date_str, 'open': item['o'], 'high': item['h'], 'low': item['l']}
    lt_result.append(item_result)

result = {'highest_price': highest_price, 'lowest_price': lowest_price, 'prices': lt_result}
print(result)