<?php
require_once 'API/api.php';
require_once 'CRUD/read.php';
require_once 'CRUD/create.php';

$weather = null;

if (!empty($_POST['city'])) {
    $city = $_POST['city'];
    $weather = fetchWeatherData('current.json', ['q' => $city]);

    if (isset($weather['current'])) {
        // Remplacer "sunny" par "ensoleillé"
        if (isset($weather['current']['condition']['text']) && $weather['current']['condition']['text'] == 'Sunny') {
            $weather['current']['condition']['text'] = 'Ensoleillé';
        }

        // Remplacer "partly cloudy" par "nuageux"
        if (isset($weather['current']['condition']['text']) && $weather['current']['condition']['text'] == 'Partly cloudy') {
            $weather['current']['condition']['text'] = 'Nuageux';
        }

        // Remplacer "overcast" par "couvert"
        if (isset($weather['current']['condition']['text']) && $weather['current']['condition']['text'] == 'Overcast') {
            $weather['current']['condition']['text'] = 'Couvert';
        }

        saveSearch($city, $weather['current']['temp_c'], $weather['current']['condition']['text']);
    }
}
?>
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Site Météo</title>
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
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            height: 100vh;
            padding: 20px;
            background-image: linear-gradient(135deg, #74ebd5, #9face6); /* Gradient background */
            font-size: 16px;
        }

        h1 {
            font-size: 2.5rem;
            color: #2c3e50;
            margin-bottom: 30px;
            text-align: center;
        }

        h2 {
            font-size: 1.8rem;
            color: #34495e;
            margin-bottom: 10px;
        }

        p {
            font-size: 1rem;
            color: #7f8c8d;
            margin-bottom: 15px;
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
            max-width: 400px;
            text-align: center;
            font-size: 1.2rem;
        }

        .weather-info p {
            margin-bottom: 15px;
            font-weight: bold;
            color: #34495e;
        }

        .weather-info h2 {
            color: #2980b9;
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
        }
    </style>
</head>
<body>
    <h1>Recherchez la météo</h1>

    <form method="POST">
        <input type="text" name="city" placeholder="Entrez une ville" required>
        <button type="submit">Rechercher</button>
    </form>

    <?php if ($weather): ?>
        <div class="weather-info">
            <h2>Météo pour <?= htmlspecialchars($city) ?></h2>
            <p>Température : <?= $weather['current']['temp_c'] ?> °C</p>
            <p>Condition : <?= $weather['current']['condition']['text'] ?></p>
        </div>
    <?php endif; ?>
</body>
</html>
