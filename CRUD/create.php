<?php
require_once __DIR__ . '/../config.php';  // Inclure la connexion à la base de données

// Obtenir la connexion PDO
$pdo = getDatabaseConnection();

// Vérifier si la connexion à la base de données a échoué
if ($pdo === null) {
    die("Erreur de connexion à la base de données!");
}

function saveSearch($city, $temperature, $condition) {
    global $pdo;

    // Enregistrer les détails dans le journal
    error_log("Tentative d'enregistrement: Ville=$city, Temp=$temperature, Condition=$condition");
    
    if (!$pdo) {
        error_log("Erreur: PDO est null dans saveSearch()");
        return false;
    }

    try {
        // Préparer la requête
        $query = "INSERT INTO searches (city, temperature, `condition`) VALUES (?, ?, ?)";
        $stmt = $pdo->prepare($query);
        
        // Exécuter avec des paramètres directs
        $result = $stmt->execute([$city, $temperature, $condition]);
        
        // Vérifier le résultat et l'enregistrer dans le journal
        if ($result) {
            error_log("Enregistrement réussi dans 'searches': $city, $temperature, $condition");
            return true;
        } else {
            error_log("Échec de l'exécution de l'enregistrement. Erreur info: " . implode(", ", $stmt->errorInfo()));
            return false;
        }
    } catch (PDOException $e) {
        error_log("Exception PDO dans saveSearch(): " . $e->getMessage());
        return false;
    }
}
// Fonction pour enregistrer les prévisions météo
function saveForecast($city, $date, $temperature, $condition, $humidity, $precipitation, $wind) {
    global $pdo;

    // Enregistrer les détails dans le journal
    error_log("Tentative d'enregistrement prévision: Ville=$city, Date=$date, Temp=$temperature");
    
    if (!$pdo) {
        error_log("Erreur: PDO est null dans saveForecast()");
        return false;
    }

    try {
        // Préparer la requête
        $query = "INSERT INTO weather_forecasts (city, forecast_date, temperature, `condition`, humidity, precipitation, wind) 
                VALUES (?, ?, ?, ?, ?, ?, ?)";
        $stmt = $pdo->prepare($query);
        
        // Exécuter avec des paramètres directs
        $result = $stmt->execute([$city, $date, $temperature, $condition, $humidity, $precipitation, $wind]);
        
        // Vérifier le résultat et l'enregistrer dans le journal
        if ($result) {
            error_log("Enregistrement prévision réussi: $city, $date");
            return true;
        } else {
            error_log("Échec de l'enregistrement prévision. Erreur info: " . implode(", ", $stmt->errorInfo()));
            return false;
        }
    } catch (PDOException $e) {
        error_log("Exception PDO dans saveForecast(): " . $e->getMessage());
        return false;
    }
}

if ($_SERVER['REQUEST_METHOD'] == 'POST' && isset($_POST['city'], $_POST['temperature'])) {
    $city = $_POST['city'];
    $temperature = $_POST['temperature'];
    $condition = $_POST['condition'] ?? '';  // Condition est optionnelle, peut être vide si non fournie

    saveSearch($city, $temperature, $condition);  // Appel de la fonction avec les bons paramètres
}
?>