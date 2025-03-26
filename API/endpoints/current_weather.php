<?php
require_once '../api.php';
require_once '../../CRUD/create.php';

if (!empty($_GET['city'])) {
    $city = $_GET['city'];
    $weather = fetchWeatherData('current.json', ['q' => $city]);
    
    // Enregistrer la recherche actuelle dans la base de données
    if (isset($weather['current'])) {
        saveSearch($city, $weather['current']['temp_c'], $weather['current']['condition']['text']);
    }
    
    echo json_encode($weather);
} else {
    echo json_encode(['error' => 'Ville non spécifiée']);
} 
?>