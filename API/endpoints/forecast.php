<?php
require_once '../api.php';

if (!empty($_GET['city'])) {
    $city = $_GET['city'];
    $forecast = fetchWeatherData('forecast.json', ['q' => $city, 'days' => 3]);
    echo json_encode($forecast);
} else {
    echo json_encode(['error' => 'Ville non spécifiée']);
}
