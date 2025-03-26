<?php
require_once 'API/api.php';
require_once 'CRUD/read.php';
require_once 'CRUD/create.php';

$weather = null;
$forecast = null;

if (!empty($_POST['city'])) {
    $city = $_POST['city'];
    
    // Récupérer les données météo actuelles
    $weather = fetchWeatherData('current.json', ['q' => $city]);
    
    // Récupérer les prévisions sur 7 jours
    $forecastData = fetchWeeklyForecast($city);
    
    if (isset($weather['current'])) {
        // Traduire les conditions météorologiques
        $conditions = [
            'Sunny' => 'Soleil ça brille les yeux carrément !',
            'Partly Cloudy' => 'Vif du ptit nuage mais trql',
            'Cloudy' => 'Ptit temps nuageux il fait gris',
            'Overcast' => 'Couvert',
            'Mist' => 'Brume mystique',
            'Patchy rain possible' => 'Big pluie en perspective',
            'Patchy snow possible' => 'Neige éparse possible',
            'Patchy sleet possible' => 'Neige fondue éparse possible',
            'Patchy freezing drizzle possible' => 'Bruine verglaçante éparse possible',
            'Thundery outbreaks possible' => 'Orages possibles fais belek ca gronde',
            'Light rain' => 'Pluie mignonne',
            'Moderate rain' => 'Pluie tranquille',
            'Heavy rain' => 'Il pleut sa mère fais belek',
            'Light snow' => 'Ptite neige trql zarma Noël',
            'Moderate snow' => 'Neige modérée',
            'Heavy snow' => 'Big neige tah la Russie',
            'Blowing snow' => 'Neige de fou sa mère',
            'Clear' => 'Clair'
        ];
        
        // Emoji pour chaque condition météo
        $weatherEmojis = [
            'Sunny' => '☀️',
            'Partly Cloudy' => '⛅',
            'Cloudy' => '☁️',
            'Overcast' => '☁️',
            'Mist' => '🌫️',
            'Patchy rain possible' => '🌦️',
            'Patchy snow possible' => '🌨️',
            'Patchy sleet possible' => '🌨️',
            'Patchy freezing drizzle possible' => '🌧️',
            'Thundery outbreaks possible' => '⛈️',
            'Light rain' => '🌧️',
            'Moderate rain' => '🌧️',
            'Heavy rain' => '🌧️',
            'Blowing snow' => '🌨️',
            'Light snow' => '❄️',
            'Moderate snow' => '❄️',
            'Heavy snow' => '❄️',
            'Clear' => '🌟'
        ];
        
        if (isset($weather['current']['condition']['text'])) {
            $conditionText = $weather['current']['condition']['text'];
            $emoji = isset($weatherEmojis[$conditionText]) ? $weatherEmojis[$conditionText] : '';
            
            if (array_key_exists($conditionText, $conditions)) {
                $weather['current']['condition']['text'] = $conditions[$conditionText];
                $weather['current']['condition']['emoji'] = $emoji;
            }
        }
        
        // Enregistrer la recherche dans la base de données
        $result = saveSearch($city, $weather['current']['temp_c'], $weather['current']['condition']['text']);
        
        // Traiter et enregistrer les prévisions
        if (isset($forecastData['forecast']['forecastday'])) {
            $forecast = $forecastData['forecast']['forecastday'];
            
            // Ajouter des emoji à chaque prévision
            foreach ($forecast as $key => $day) {
                $conditionText = $day['day']['condition']['text'];
                $emoji = isset($weatherEmojis[$conditionText]) ? $weatherEmojis[$conditionText] : '';
                $forecast[$key]['day']['condition']['emoji'] = $emoji;
                
                // Enregistrer chaque jour de prévision dans la base de données
                $date = $day['date'];
                $temp = $day['day']['avgtemp_c'];
                
                // Traduire la condition si elle existe dans notre tableau
                $translatedCondition = isset($conditions[$conditionText]) ? $conditions[$conditionText] : $conditionText;
                
                $humidity = $day['day']['avghumidity'];
                $precipitation = $day['day']['totalprecip_mm'];
                $wind = $day['day']['maxwind_kph'];
                
                saveForecast($city, $date, $temp, $translatedCondition, $humidity, $precipitation, $wind);
            }
        }
    }
}
?>
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Y-FLOP Météo</title>
    <style>
        /* styles.css */
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: 'Arial', sans-serif;
        }

        body {
            background-color: #f0f4f8;
            color: #333;
            min-height: 100vh;
            padding: 20px;
            background-image: linear-gradient(135deg, #74ebd5, #9face6); /* Gradient background */
            font-size: 16px;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }

        header {
            text-align: center;
            margin-bottom: 40px;
        }

        h1 {
            font-size: 2.5rem;
            color: #2c3e50;
            margin-bottom: 10px;
            text-align: center;
        }

        h2 {
            font-size: 1.8rem;
            color: #34495e;
            margin-bottom: 10px;
        }

        h3 {
            font-size: 1.4rem;
            color: #34495e;
            margin-bottom: 10px;
        }

        p {
            font-size: 1rem;
            color: #7f8c8d;
            margin-bottom: 15px;
        }
        
        .emoji {
            font-size: 2rem;
            margin-bottom: 10px;
            display: block;
        }

        form {
            display: flex;
            flex-direction: column;
            align-items: center;
            background-color: #ffffff;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 10px 20px rgba(0, 0, 0, 0.1);
            margin-bottom: 30px;
            width: 100%;
            max-width: 400px;
            margin: 0 auto;
        }

        input[type="text"] {
            width: 100%;
            padding: 12px;
            margin-bottom: 15px;
            font-size: 1.1rem;
            border: 1px solid #ddd;
            border-radius: 8px;
            outline: none;
            transition: border-color 0.3s ease;
        }

        input[type="text"]:focus {
            border-color: #3498db;
        }

        button[type="submit"] {
            padding: 12px 20px;
            font-size: 1.1rem;
            color: white;
            background-color: #3498db;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            transition: background-color 0.3s ease;
            width: 100%;
        }

        button[type="submit"]:hover {
            background-color: #2980b9;
        }

        .weather-info {
            background-color: #ecf0f1;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 10px 20px rgba(0, 0, 0, 0.1);
            margin-top: 20px;
            width: 100%;
            max-width: 800px;
            text-align: center;
            font-size: 1.2rem;
            margin: 20px auto;
        }

        .weather-info p {
            margin-bottom: 15px;
            font-weight: bold;
            color: #34495e;
        }

        .weather-info h2 {
            color: #2980b9;
            margin-bottom: 20px;
        }

        .forecast-container {
            display: flex;
            flex-wrap: wrap;
            justify-content: space-between;
            margin-top: 40px;
        }

        .forecast-day {
            background-color: #ffffff;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
            padding: 15px;
            margin-bottom: 20px;
            width: calc(25% - 15px);
            text-align: center;
        }

        .forecast-day h3 {
            color: #2980b9;
            margin-bottom: 10px;
        }

        .forecast-day p {
            margin-bottom: 8px;
            color: #34495e;
        }

        @media (max-width: 900px) {
            .forecast-day {
                width: calc(33.33% - 15px);
            }
        }

        @media (max-width: 700px) {
            .forecast-day {
                width: calc(50% - 10px);
            }
        }

        @media (max-width: 600px) {
            h1 {
                font-size: 2rem;
            }

            form {
                width: 100%;
                padding: 20px;
            }

            input[type="text"] {
                width: 100%;
            }

            button[type="submit"] {
                width: 100%;
            }

            .weather-info {
                max-width: 100%;
                padding: 20px;
            }
            
            .forecast-day {
                width: 100%;
            }
        }

        .nav-links {
            display: flex;
            justify-content: center;
            margin-bottom: 30px;
        }

        .nav-link {
            padding: 10px 20px;
            text-decoration: none;
            color: #3498db;
            font-weight: bold;
            margin: 0 10px;
            border-radius: 5px;
            transition: background-color 0.3s;
        }

        .nav-link:hover {
            background-color: rgba(52, 152, 219, 0.1);
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Y-FLOP Météo</h1>
            <p>Prévisions météorologiques détaillées</p>
        </header>

        <div class="nav-links">
            <a href="index.php" class="nav-link">Météo d'aujourd'hui</a>
        </div>

        <form method="POST">
            <input type="text" name="city" placeholder="Entrez une ville" required>
            <button type="submit">Rechercher</button>
        </form>

        <?php if ($weather): ?>
            <div class="weather-info">
                <h2>Météo actuelle pour <?= htmlspecialchars($city) ?></h2>
                <span class="emoji"><?= isset($weather['current']['condition']['emoji']) ? $weather['current']['condition']['emoji'] : '' ?></span>
                <p>Température : <?= $weather['current']['temp_c'] ?> °C</p>
                <p>Condition : <?= $weather['current']['condition']['text'] ?></p>
                <p>Humidité : <?= $weather['current']['humidity'] ?>%</p>
                <p>Précipitations : <?= isset($weather['current']['precip_mm']) ? $weather['current']['precip_mm'] . " mm" : "Non disponible" ?></p>
                <p>Vent : <?= isset($weather['current']['wind_kph']) ? $weather['current']['wind_kph'] . " km/h" : "Non disponible" ?></p>
                <p>Pression : <?= isset($weather['current']['pressure_mb']) ? $weather['current']['pressure_mb'] . " mb" : "Non disponible" ?></p>
            </div>

            <?php if ($forecast): ?>
                <div class="weather-info">
                    <h2>Prévisions sur 3 jours pour <?= htmlspecialchars($city) ?></h2>
                    <div class="forecast-container">
                        <?php foreach ($forecast as $index => $day): ?>
                            <div class="forecast-day">
                                <h3><?= date('d/m', strtotime($day['date'])) ?></h3>
                                <p><?= date('l', strtotime($day['date'])) ?></p>
                                <span class="emoji"><?= $day['day']['condition']['emoji'] ?></span>
                                <p>Temp. Min: <?= $day['day']['mintemp_c'] ?> °C</p>
                                <p>Temp. Max: <?= $day['day']['maxtemp_c'] ?> °C</p>
                                <p>Condition: <?= isset($conditions[$day['day']['condition']['text']]) ? $conditions[$day['day']['condition']['text']] : $day['day']['condition']['text'] ?></p>
                                <p>Humidité: <?= $day['day']['avghumidity'] ?>%</p>
                                <p>Précip.: <?= $day['day']['totalprecip_mm'] ?> mm</p>
                                <p>Vent: <?= $day['day']['maxwind_kph'] ?> km/h</p>
                            </div>
                        <?php endforeach; ?>
                    </div>
                </div>
            <?php endif; ?>
        <?php endif; ?>
    </div>
</body>
</html>