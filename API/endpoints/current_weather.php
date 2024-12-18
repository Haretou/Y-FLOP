<?php
require_once '../api.php';

if (!empty($_GET['city'])) {
    $city = $_GET['city'];
    $weather = fetchWeatherData('current.json', ['q' => $city]);
    echo json_encode($weather);
} else {
    echo json_encode(['error' => 'Ville non spécifiée']);
} 
?>