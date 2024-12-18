<?php
// Clé API WeatherAPI
define('API_KEY', '6f5b88ad73d84dc585b132034241812');
define('BASE_URL', 'https://api.weatherapi.com/v1/');

// Configuration MySQL
define('DB_HOST', 'localhost');
define('DB_NAME', 'meteo_site');
define('DB_USER', 'root');
define('DB_PASS', 'root');

// Connexion à la base de données
function getDatabaseConnection() {
    try {
        return new PDO("mysql:host=" . DB_HOST . ";dbname=" . DB_NAME, DB_USER, DB_PASS, [
            PDO::ATTR_ERRMODE => PDO::ERRMODE_EXCEPTION
        ]);
    } catch (PDOException $e) {
        die('Erreur de connexion : ' . $e->getMessage());
    }
}
?>
