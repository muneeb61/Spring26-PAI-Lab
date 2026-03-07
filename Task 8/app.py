from flask import Flask, render_template, request
import requests
import urllib.parse

app = Flask(__name__)

def get_weather(city):
    city = city.strip().strip('"').strip("'").title()
    url = f"https://wttr.in/{urllib.parse.quote(city)}?format=j1"
    print(f"[DEBUG] Fetching URL: {url}")
    try:
        response = requests.get(url, timeout=10)
        print(f"[DEBUG] Status Code: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"[DEBUG] Data received successfully")
            current = data['current_condition'][0]
            area = data['nearest_area'][0]

            return {
                'city': city,
                'country': area['country'][0]['value'],
                'temp_c': current['temp_C'],
                'temp_f': current['temp_F'],
                'feels_like_c': current['FeelsLikeC'],
                'humidity': current['humidity'],
                'wind_speed': current['windspeedKmph'],
                'wind_dir': current['winddir16Point'],
                'description': current['weatherDesc'][0]['value'],
                'visibility': current['visibility'],
                'pressure': current['pressure'],
                'uv_index': current['uvIndex'],
                'weather_code': current['weatherCode'],
                'error': None
            }
        else:
            print(f"[DEBUG] Bad status code: {response.status_code}")
            return {'error': f'API returned status code {response.status_code}'}
    except requests.exceptions.Timeout:
        print("[DEBUG] Request timed out")
        return {'error': 'Request timed out. Please try again.'}
    except requests.exceptions.ConnectionError:
        print("[DEBUG] Connection error")
        return {'error': 'Connection error. Check your internet connection.'}
    except Exception as e:
        print(f"[DEBUG] Exception: {e}")
        return {'error': str(e)}


def get_weather_icon(description):
    desc = description.lower()
    if 'sunny' in desc or 'clear' in desc:
        return '☀️'
    elif 'partly cloudy' in desc:
        return '⛅'
    elif 'cloudy' in desc or 'overcast' in desc:
        return '☁️'
    elif 'rain' in desc or 'drizzle' in desc:
        return '🌧️'
    elif 'thunder' in desc or 'storm' in desc:
        return '⛈️'
    elif 'snow' in desc or 'blizzard' in desc:
        return '❄️'
    elif 'fog' in desc or 'mist' in desc:
        return '🌫️'
    elif 'wind' in desc:
        return '💨'
    else:
        return '🌤️'


@app.route('/', methods=['GET', 'POST'])
def index():
    weather = None
    icon = None
    city_input = ''
    if request.method == 'POST':
        city_input = request.form.get('city', '').strip().strip('"').strip("'")
        print(f"[DEBUG] City input received: '{city_input}'")
        if city_input:
            weather = get_weather(city_input)
            print(f"[DEBUG] Weather result: {weather}")
            if weather and not weather.get('error'):
                icon = get_weather_icon(weather['description'])
                print(f"[DEBUG] Icon selected: {icon}")
    return render_template('index.html', weather=weather, icon=icon, city_input=city_input)


if __name__ == '__main__':
    app.run(debug=True)