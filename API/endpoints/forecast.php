<?php
require_once '../api.php';
require_once '../../CRUD/create.php';

if (!empty($_GET['city'])) {
    $city = $_GET['city'];
    $forecast = fetchWeatherData('forecast.json', ['q' => $city, 'days' => 7]);
    
    // Enregistrer les prévisions dans la base de données
    if (isset($forecast['forecast']['forecastday'])) {
        foreach ($forecast['forecast']['forecastday'] as $day) {
            $date = $day['date'];
            $temp = $day['day']['avgtemp_c'];
            $condition = $day['day']['condition']['text'];
            $humidity = $day['day']['avghumidity'];
            $precipitation = $day['day']['totalprecip_mm'];
            $wind = $day['day']['maxwind_kph'];
            
            saveForecast($city, $date, $temp, $condition, $humidity, $precipitation, $wind);
        }
    }
    
    echo json_encode($forecast);
} else {
    echo json_encode(['error' => 'Ville non spécifiée']);
}
?>