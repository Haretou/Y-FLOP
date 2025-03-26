<?php
require_once '../machine_learning.php';
require_once '../../config.php';

// Obtenir la connexion à la base de données
$pdo = getDatabaseConnection();

// Vérifier le type de requête demandé
if (!empty($_GET['action']) && !empty($_GET['city'])) {
    $action = $_GET['action'];
    $city = $_GET['city'];
    $response = [];
    
    switch ($action) {
        case 'trend':
            // Obtenir les tendances de température
            $response = getTemperatureTrend($pdo, $city);
            break;
            
        case 'predict':
            // Obtenir des prédictions météo ajustées
            $response = predictWeather($pdo, $city);
            break;
            
        case 'anomalies':
            // Détecter les anomalies météo
            $response = detectWeatherAnomalies($pdo, $city);
            break;
            
        default:
            $response = ['error' => 'Action non reconnue'];
    }
    
    // Renvoyer la réponse au format JSON
    header('Content-Type: application/json');
    echo json_encode($response);
} else {
    header('Content-Type: application/json');
    echo json_encode(['error' => 'Paramètres manquants (action et city requis)']);
}
?>