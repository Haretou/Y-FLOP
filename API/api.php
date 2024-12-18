<?php

// Charger les paramètres de configuration à partir du fichier config.php
$config = require_once __DIR__ . '/../config.php';

// Définir les en-têtes pour retourner des réponses en JSON
header('Content-Type: application/json');

/**
 * Fonction pour gérer les requêtes API vers l'API Météo.
 * @param string $endpoint L'endpoint de l'API à appeler (par exemple, current.json, forecast.json).
 * @param array $params Paramètres de la requête à inclure dans la demande.
 * @return array Réponse JSON décodée de l'API.
 */
function fetch_weather_data($endpoint, $params = []) {
    global $config;

    // Construire l'URL complète de l'API avec les paramètres
    $url = $config['api_base_url'] . $endpoint . '?' . http_build_query(array_merge($params, [
        'key' => $config['api_key'],
    ]));

    $curl = curl_init(); 

    curl_setopt_array($curl, [
        CURLOPT_URL => $url, 
        CURLOPT_RETURNTRANSFER => true, 
        CURLOPT_TIMEOUT => $config['timeout'], 
    ]);

    $response = curl_exec($curl); 

    if (curl_errno($curl)) {
        http_response_code(500); 
        echo json_encode([ 'error' => 'Échec de la connexion à l\'API météo.' ]);
        exit;
    }

    curl_close($curl); 

    return json_decode($response, true);
}

// Vérifier si le paramètre 'action' est défini
if (!isset($_GET['action'])) {
    http_response_code(400); 
    echo json_encode([ 'error' => 'Aucune action spécifiée.' ]);
    exit;
}

$action = $_GET['action']; 

// Diriger la requête selon l'action spécifiée
switch ($action) {
    case 'current':
        if (!isset($_GET['location'])) {
            http_response_code(400); 
            echo json_encode([ 'error' => 'Le paramètre de localisation est requis.' ]);
            exit;
        }

        $location = $_GET['location']; 
        $data = fetch_weather_data('current.json', [ 'q' => $location ]); 
        echo json_encode($data); 
        break;

    case 'forecast':
        if (!isset($_GET['location']) || !isset($_GET['days'])) {
            http_response_code(400); 
            echo json_encode([ 'error' => 'Les paramètres de localisation et de jours sont requis.' ]);
            exit;
        }

        $location = $_GET['location']; 
        $days = (int) $_GET['days']; 
        $data = fetch_weather_data('forecast.json', [ 'q' => $location, 'days' => $days ]); 
        echo json_encode($data); 
        break;

    default:
        http_response_code(400); 
        echo json_encode([ 'error' => 'Action spécifiée invalide.' ]);
        break;
}

?>

