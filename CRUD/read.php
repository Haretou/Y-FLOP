<?php
require_once __DIR__ . '/../config.php';  // Inclure la connexion à la base de données

// Obtenir la connexion PDO
$pdo = getDatabaseConnection();

function getRecentSearches($pdo, $limit = 10) {
    if (!$pdo) {
        error_log("Erreur de connexion à la base de données dans getRecentSearches");
        return [];
    }

    try {
        // Préparer et exécuter la requête pour récupérer les recherches récentes
        // Modification pour inclure la condition météo
        $stmt = $pdo->prepare("SELECT city, temperature, `condition` FROM searches ORDER BY search_date DESC LIMIT ?");
        $stmt->execute([$limit]);
        return $stmt->fetchAll(PDO::FETCH_ASSOC);
    } catch (PDOException $e) {
        error_log("Erreur SQL dans getRecentSearches: " . $e->getMessage());
        return [];
    }
}

function getWeeklyForecast($pdo, $city) {
    if (!$pdo) {
        error_log("Erreur de connexion à la base de données dans getWeeklyForecast");
        return [];
    }

    try {
        // Récupérer les prévisions pour les 7 prochains jours pour la ville spécifiée
        $stmt = $pdo->prepare("SELECT forecast_date, temperature, `condition`, humidity, precipitation, wind 
                            FROM weather_forecasts 
                            WHERE city = :city 
                            ORDER BY forecast_date ASC 
                            LIMIT 7");
        $stmt->execute(['city' => $city]);
        return $stmt->fetchAll(PDO::FETCH_ASSOC);
    } catch (PDOException $e) {
        error_log("Erreur SQL dans getWeeklyForecast: " . $e->getMessage());
        return [];
    }
}

if ($pdo) {
    $query = "
        SELECT 
            Place.Name AS Place_Name, 
            Place.Latitude, 
            Place.Longitude, 
            Weather.State, 
            Weather.Town, 
            Weather.temperature, 
            Weather.precipitation, 
            Weather.wind, 
            Weather.humidity, 
            Time.day, 
            Time.hour
        FROM Place
        JOIN Time ON Place.id = Time.place_id
        JOIN Weather ON Time.weather_id = Weather.id
    ";

    try {
        $statement = $pdo->query($query);
        $results = $statement->fetchAll(PDO::FETCH_ASSOC);
        // Pour déboguer
        error_log("Résultats de la requête de base : " . json_encode($results));
    } catch (PDOException $e) {
        error_log("Erreur SQL dans le bloc principal: " . $e->getMessage());
    }
} else {
    error_log("Erreur de connexion à la base de données dans le bloc principal");
}
?>