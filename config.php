<?php

    define('BASE_URL', 'http://api.weatherapi.com/v1/');  // URL de base de l'API
    define('API_KEY','6f5b88ad73d84dc585b132034241812');  // Clé d'API pour accéder à l'API
function getDatabaseConnection() {
    $host = 'localhost';  // Hôte de la base de données
    $dbname = 'meteo_db'; // Nom de la base de données
    $username = 'root';   // Nom d'utilisateur (par défaut 'root' sur MAMP)
    $password = 'root';       // Mot de passe (vide par défaut sur MAMP)

    try {
        // Créer la connexion à la base de données avec PDO
        $pdo = new PDO("mysql:host=$host;dbname=$dbname", $username, $password);
        // Configurer le mode d'erreur de PDO pour afficher les erreurs
        $pdo->setAttribute(PDO::ATTR_ERRMODE, PDO::ERRMODE_EXCEPTION);
        return $pdo; // Retourner l'objet PDO
    } catch (PDOException $e) {
        echo "Erreur de connexion à la base de données : " . $e->getMessage();
        return null;
    }
}
?>
