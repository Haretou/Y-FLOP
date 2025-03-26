<?php
/**
 * Script pour mettre à jour automatiquement les données météorologiques.
 * À exécuter via un cron job, par exemple quotidiennement.
 * 
 * Exemple de configuration cron:
 * 0 0 * * * php /chemin/vers/cron_update.php
 */

require_once 'config.php';
require_once 'API/api.php';
require_once 'CRUD/create.php';

// Journalisation des actions
function logMessage($message) {
    $date = date('Y-m-d H:i:s');
    echo "[$date] $message\n";
    file_put_contents('cron_log.txt', "[$date] $message\n", FILE_APPEND);
}

// Fonction pour mettre à jour les prévisions d'une ville
function updateCityForecasts($city) {
    global $pdo;
    
    logMessage("Mise à jour des prévisions pour $city...");
    
    // Récupérer les prévisions sur 7 jours
    $forecastData = fetchWeeklyForecast($city);
    
    if (!isset($forecastData['forecast']['forecastday'])) {
        logMessage("Erreur: Impossible de récupérer les prévisions pour $city");
        return false;
    }
    
    // Définir les traductions pour les conditions météorologiques
    $conditions = [
        'Sunny' => 'Ensoleillé',
        'Partly cloudy' => 'Partiellement nuageux',
        'Cloudy' => 'Nuageux',
        'Overcast' => 'Couvert',
        'Mist' => 'Brume',
        'Patchy rain possible' => 'Pluie éparse possible',
        'Patchy snow possible' => 'Neige éparse possible',
        'Patchy sleet possible' => 'Neige fondue éparse possible',
        'Patchy freezing drizzle possible' => 'Bruine verglaçante éparse possible',
        'Thundery outbreaks possible' => 'Orages possibles',
        'Light rain' => 'Pluie légère',
        'Moderate rain' => 'Pluie modérée',
        'Heavy rain' => 'Forte pluie',
        'Light snow' => 'Neige légère',
        'Moderate snow' => 'Neige modérée',
        'Heavy snow' => 'Forte neige',
        'Clear' => 'Clair'
    ];
    
    $success = true;
    $forecastDays = $forecastData['forecast']['forecastday'];
    
    // Supprimer les anciennes prévisions pour cette ville
    try {
        $stmt = $pdo->prepare("DELETE FROM weather_forecasts WHERE city = :city");
        $stmt->execute(['city' => $city]);
        logMessage("Anciennes prévisions supprimées pour $city");
    } catch (PDOException $e) {
        logMessage("Erreur lors de la suppression des anciennes prévisions: " . $e->getMessage());
        $success = false;
    }
    
    // Enregistrer les nouvelles prévisions
    foreach ($forecastDays as $day) {
        $date = $day['date'];
        $temp = $day['day']['avgtemp_c'];
        $conditionText = $day['day']['condition']['text'];
        
        // Traduire la condition si elle existe dans notre tableau
        if (array_key_exists($conditionText, $conditions)) {
            $conditionText = $conditions[$conditionText];
        }
        
        $humidity = $day['day']['avghumidity'];
        $precipitation = $day['day']['totalprecip_mm'];
        $wind = $day['day']['maxwind_kph'];
        
        if (!saveForecast($city, $date, $temp, $conditionText, $humidity, $precipitation, $wind)) {
            logMessage("Erreur lors de l'enregistrement de la prévision pour $city le $date");
            $success = false;
        }
    }
    
    if ($success) {
        logMessage("Prévisions mises à jour avec succès pour $city");
    }
    
    return $success;
}

// Début du script principal
// Obtention de la connexion PDO
$pdo = getDatabaseConnection();
if (!$pdo) {
    logMessage("Erreur: Impossible de se connecter à la base de données");
    exit(1);
}

logMessage("Démarrage de la mise à jour automatique des prévisions météorologiques...");

try {
    // Récupérer les villes les plus recherchées depuis la base de données
    $stmt = $pdo->query(
        "SELECT city, COUNT(*) as search_count 
        FROM searches 
        GROUP BY city 
        ORDER BY search_count DESC 
        LIMIT 10"
    );
    $topCities = $stmt->fetchAll(PDO::FETCH_ASSOC);
    
    if (empty($topCities)) {
        logMessage("Aucune ville trouvée dans la base de données");
        // Utiliser quelques villes par défaut si la base est vide
        $defaultCities = ['Paris', 'Marseille', 'Lyon', 'Toulouse', 'Nice'];
        foreach ($defaultCities as $city) {
            updateCityForecasts($city);
        }
    } else {
        logMessage("Mise à jour des prévisions pour les " . count($topCities) . " villes les plus recherchées");
        foreach ($topCities as $cityData) {
            updateCityForecasts($cityData['city']);
        }
    }
    
    logMessage("Mise à jour terminée avec succès");
} catch (PDOException $e) {
    logMessage("Erreur lors de la récupération des villes: " . $e->getMessage());
    exit(1);
}