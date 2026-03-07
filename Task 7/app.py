from flask import Flask, render_template, request
import requests

app = Flask(__name__)

def get_weather(city):
    url = f"https://wttr.in/{city}?format=j1"
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            current = data['current_condition'][0]
            area = data['nearest_area'][0]

            return {
                'city': city.title(),
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
    except Exception as e:
        return {'error': str(e)}
    return {'error': 'Could not fetch weather data.'}


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
        city_input = request.form.get('city', '').strip()
        if city_input:
            weather = get_weather(city_input)
            if weather and not weather.get('error'):
                icon = get_weather_icon(weather['description'])
    return render_template('index.html', weather=weather, icon=icon, city_input=city_input)


if __name__ == '__main__':
    app.run(debug=True)
