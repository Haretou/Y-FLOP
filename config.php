<?php
    define('BASE_URL', 'http://api.weatherapi.com/v1/');  // URL de base de l'API
    define('API_KEY','6f5b88ad73d84dc585b132034241812');  // Clé d'API pour accéder à l'API
    function getDatabaseConnection() {
        $host = 'localhost';
        $dbname = 'meteo_db'; // Vérifiez que ce nom de base de données est correct
        $username = 'root';
        $password = 'root'; // Vérifiez que ce mot de passe est correct
    
        try {
            // Créer la connexion
            $pdo = new PDO("mysql:host=$host;dbname=$dbname;charset=utf8", $username, $password);
            // Configurer pour afficher les erreurs
            $pdo->setAttribute(PDO::ATTR_ERRMODE, PDO::ERRMODE_EXCEPTION);
            $pdo->setAttribute(PDO::ATTR_DEFAULT_FETCH_MODE, PDO::FETCH_ASSOC);
            return $pdo;
        } catch (PDOException $e) {
            error_log("Erreur de connexion PDO: " . $e->getMessage());
            return null;
        }
    }
?>