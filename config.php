<?php
// Clé API WeatherAPI
define('API_KEY', '6f5b88ad73d84dc585b132034241812');
define('BASE_URL', 'https://api.weatherapi.com/v1/');


$host = 'localhost';  // Hôte de la base de données
$dbname = 'meteo_db'; // Nom de la base de données
$username = 'root';   // Nom d'utilisateur
$password = '';       // Mot de passe (s'il n'y en a pas, laisse vide)

// Configuration de la connexion PDO
try {
    // Créer la connexion à la base de données avec PDO
    $pdo = new PDO("mysql:host=$host;dbname=$dbname", $username, $password);
    // Configurer le mode d'erreur de PDO pour afficher les erreurs
    $pdo->setAttribute(PDO::ATTR_ERRMODE, PDO::ERRMODE_EXCEPTION);
} catch (PDOException $e) {
    // En cas d'erreur, afficher un message d'erreur
    echo "Erreur de connexion à la base de données : " . $e->getMessage();
}
?>
?>
